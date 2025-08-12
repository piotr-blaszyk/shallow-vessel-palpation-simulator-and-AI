import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3, lr_patience=5, lr_factor=0.5, lr_min=1e-6, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),  # Increased number and size of channels
            strides=(2, 2, 2),  # Added another downsampling level
            num_res_units=3,    # Increased residual units
            act='PRELU',
            norm='INSTANCE',
            dropout=0.2,        # Added some dropout for regularization
            bias=True,
            adn_ordering='NDA',
        )
        # Initialize both loss functions
        self.dice_loss = DiceLoss(sigmoid=True, batch=False)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.alpha = 0.5

    def forward(self, x):
        return self.model(x)

    @classmethod
    def iou_score(cls, preds, targets, eps=1e-6):
        """
        Compute class-wise IoU scores for binary segmentation.
        
        Args:
            preds: Binary predictions (B, C, H, W) or (B, C, T, H, W)
            targets: Binary ground truth (B, C, H, W) or (B, C, T, H, W)
            eps: Small constant to avoid division by zero
            
        Returns:
            Dictionary containing various IoU metrics
        """
        # Handle 5D tensors (B, C, T, H, W) -> (B*T, C, H, W)
        if preds.ndim == 5:
            B, C, T, H, W = preds.shape
            preds = preds.transpose(1, 2).reshape(-1, C, H, W)
            targets = targets.transpose(1, 2).reshape(-1, C, H, W)
            
        # Ensure inputs have the right shape (B, H, W)
        preds = preds.squeeze(1)
        targets = targets.squeeze(1)
            
        # Compute frame-wise ground truth presence (B)
        gt_presence = targets.sum(dim=(1, 2)) > 0
        pred_presence = preds.sum(dim=(1, 2)) > 0
        
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
        logits = self(x)
        
        # Calculate both losses
        dice_loss = self.dice_loss(logits, y)
        bce_loss = self.bce_loss(logits, y)
        
        # Combine losses using weights
        loss = self.alpha * dice_loss + (1 - self.alpha) * bce_loss
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        preds = preds.squeeze(1)
        y = y.squeeze(1)
        metrics = self.iou_score(preds, y)
        
        # Log all metrics
        self.log(f"{stage}_dice_loss", dice_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_bce_loss", bce_loss, prog_bar=False, on_epoch=True)
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
