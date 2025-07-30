import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="timm-mobilenetv3_small_075",
            encoder_weights=None,
            in_channels=1,
            classes=1,
        )
        self.loss_fn = smp.losses.DiceLoss(mode="binary")
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
        intersection = (preds * targets).sum(dim=(1, 2))
        union = (preds + targets).sum(dim=(1, 2)) - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.mean()
