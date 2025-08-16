from monai.networks.nets import BasicUNetPlusPlus, UNet
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from difftactile.cnn.common import *


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3, lr_patience=5, lr_factor=0.5, lr_min=1e-6, 
                 tversky_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),  # Increased number and size of channels
            strides=(2, 2),  # Added another downsampling level
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,    # Increased residual units
            act='PRELU',
            norm='INSTANCE',
            dropout=0.2,        # Added some dropout for regularization
            bias=True,
            adn_ordering='NDA',
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

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        tversky_loss = self.tversky_loss(logits, y)
        focal_loss = self.focal_loss(logits, y)
        
        # Combine losses using weights
        loss = self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        preds = preds.squeeze(1)
        y = y.squeeze(1)
        metrics = Common.iou_score(preds, y)
        
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
