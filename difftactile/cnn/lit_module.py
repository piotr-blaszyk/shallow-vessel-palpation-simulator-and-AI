import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceLoss


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2,
            act='PRELU',
            norm='INSTANCE',
            dropout=0.0,
            bias=True,
            adn_ordering='NDA',
        )
        self.loss_fn = DiceLoss(sigmoid=True, batch=True)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        preds = preds.squeeze(1)
        y = y.squeeze(1)
        iou = self.iou_score(preds, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def iou_score(self, preds, targets, eps=1e-6):
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.mean()
