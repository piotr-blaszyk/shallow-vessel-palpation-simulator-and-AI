import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNetPlusPlus
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Tversky loss for imbalanced data
        :param alpha: weight of false negatives (default: 0.7)
        :param beta: weight of false positives (default: 0.3)
        :param smooth: smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Sigmoid activation
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        
        return 1 - tversky


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3, lr_patience=5, lr_factor=0.5, lr_min=1e-6, 
                 tversky_weight=0.3, focal_weight=0.7):
        super().__init__()
        self.model = BasicUNetPlusPlus(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            features=(16, 16, 32, 64, 128, 16),  # Increased feature depth for fine details
            deep_supervision=False,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            dropout=0.2,
            upsample="deconv"  # Using deconvolution for better upsampling quality
        )
        # Initialize loss functions
        self.tversky_loss = TverskyLoss()
        self.focal_loss = FocalLoss()
        
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        
        # Loss weights
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
        # Save hyperparameters for logging
        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.model(x)
        # If deep supervision is enabled, return only the final output
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    @staticmethod
    def iou_score(preds, targets, eps=1e-6):
        """
        Compute class-wise IoU scores for binary segmentation.
        
        Args:
            preds: Binary predictions (B, T, H, W)
            targets: Binary ground truth (B, T, H, W)
            eps: Small constant to avoid division by zero
            
        Returns:
            Dictionary containing various IoU metrics
        """
        # Compute frame-wise ground truth presence (B, T)
        gt_presence = targets.sum(dim=(2, 3)) > 0
        pred_presence = preds.sum(dim=(2, 3)) > 0
        
        # Split frames based on ground truth presence
        has_gt_mask = gt_presence
        no_gt_mask = ~gt_presence
        
        # Initialize metrics
        metrics = {
            'fg_iou': torch.tensor(0.0, device=preds.device),
            'bg_iou': torch.tensor(0.0, device=preds.device),
            'macro_iou': torch.tensor(0.0, device=preds.device),
            'detection_rate': torch.tensor(0.0, device=preds.device)
        }
        
        # Compute IoU for frames with foreground presence
        if has_gt_mask.sum() > 0:
            # Foreground IoU (class 1)
            fg_intersection = (preds[has_gt_mask] * targets[has_gt_mask]).sum(dim=(1, 2))
            fg_union = (preds[has_gt_mask] + targets[has_gt_mask]).sum(dim=(1, 2)) - fg_intersection
            fg_iou = (fg_intersection + eps) / (fg_union + eps)
            metrics['fg_iou'] = fg_iou.mean()
            
            # Background IoU (class 0)
            bg_preds = 1 - preds[has_gt_mask]
            bg_targets = 1 - targets[has_gt_mask]
            bg_intersection = (bg_preds * bg_targets).sum(dim=(1, 2))
            bg_union = (bg_preds + bg_targets).sum(dim=(1, 2)) - bg_intersection
            bg_iou = (bg_intersection + eps) / (bg_union + eps)
            metrics['bg_iou'] = bg_iou.mean()
            
            # Macro IoU (mean of class IoUs)
            metrics['macro_iou'] = (metrics['fg_iou'] + metrics['bg_iou']) / 2
        
        # Compute detection rate for empty frames
        if no_gt_mask.sum() > 0:
            # For empty GT frames: IoU=1 if prediction also empty, 0 if not empty
            correct_empty = (~pred_presence[no_gt_mask]).float()
            metrics['detection_rate'] = correct_empty.mean()
        
        return metrics

    def shared_step(self, batch, stage):
        x, y = batch
        outputs = self(x)
        
        # Handle deep supervision outputs
        if isinstance(outputs, (list, tuple)):
            # Calculate loss for each deep supervision output
            tversky_losses = [self.tversky_loss(out, y) for out in outputs]
            focal_losses = [self.focal_loss(out, y) for out in outputs]
            
            # Average the losses across deep supervision outputs
            tversky_loss = sum(tversky_losses) / len(tversky_losses)
            focal_loss = sum(focal_losses) / len(focal_losses)
            
            # Use the final output for metrics
            logits = outputs[-1]  # Final output is typically the most refined
        else:
            # Single output case
            logits = outputs
            tversky_loss = self.tversky_loss(logits, y)
            focal_loss = self.focal_loss(logits, y)
        
        # Combine losses using weights
        loss = self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        preds = preds.squeeze(1)
        y = y.squeeze(1)
        metrics = SegmentationModel.iou_score(preds, y)
        
        # Log all metrics
        self.log(f"{stage}_tversky_loss", tversky_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_focal_loss", focal_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_combined_loss", loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_fg_iou", metrics['fg_iou'], prog_bar=True, on_epoch=True)
        self.log(f"{stage}_bg_iou", metrics['bg_iou'], prog_bar=True, on_epoch=True)
        self.log(f"{stage}_macro_iou", metrics['macro_iou'], prog_bar=True, on_epoch=True)
        self.log(f"{stage}_detection_rate", metrics['detection_rate'], prog_bar=True, on_epoch=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',           # Minimize the monitored quantity (validation loss)
                factor=0.5,          # Multiply LR by this factor when reducing
                patience=3,          # Number of epochs with no improvement after which LR will be reduced
                verbose=True,        # Print message when LR is reduced
                min_lr=1e-6,        # Don't reduce LR below this value
                cooldown=1,         # Number of epochs to wait before resuming normal operation after LR has been reduced
            ),
            "monitor": "val_fg_iou",   # Quantity to monitor
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
