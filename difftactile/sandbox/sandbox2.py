import torch
import pytorch_lightning as pl

class GNNModule(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # Accumulators per stage
        self.area_pred = {"train": torch.zeros(num_classes),
                          "val": torch.zeros(num_classes),
                          "test": torch.zeros(num_classes)}
        self.area_true = {"train": torch.zeros(num_classes),
                          "val": torch.zeros(num_classes),
                          "test": torch.zeros(num_classes)}
        self.area_inter = {"train": torch.zeros(num_classes),
                           "val": torch.zeros(num_classes),
                           "test": torch.zeros(num_classes)}

    def update_accumulators(self, preds: torch.Tensor, y: torch.Tensor, stage: str):
        for class_ix in range(self.num_classes):
            pred_mask = (preds == class_ix)
            true_mask = (y == class_ix)

            self.area_pred[stage][class_ix] += pred_mask.sum().float()
            self.area_true[stage][class_ix] += true_mask.sum().float()
            self.area_inter[stage][class_ix] += (pred_mask & true_mask).sum().float()

    def compute_ious(self, stage: str):
        ious = {}
        mean_ious = []

        for class_ix in range(self.num_classes):
            ap = self.area_pred[stage][class_ix].item()
            at = self.area_true[stage][class_ix].item()
            ai = self.area_inter[stage][class_ix].item()

            if ap == 0 and at == 0:
                iou = 1.0
            elif ap == 0 or at == 0:
                iou = 0.0
            else:
                iou = ai / (ap + at - ai)

            ious[class_ix] = iou
            mean_ious.append(iou)

        ious["mean_iou"] = sum(mean_ious) / self.num_classes
        return ious

    # ----------- Hooks for clearing accumulators -----------

    def on_train_epoch_end(self):
        train_ious = self.compute_ious("train")
        self.log_dict({f"train_iou/{k}": v for k, v in train_ious.items()})
        # clear accumulators
        self.area_pred["train"].zero_()
        self.area_true["train"].zero_()
        self.area_inter["train"].zero_()

    def on_validation_epoch_end(self):
        val_ious = self.compute_ious("val")
        self.log_dict({f"val_iou/{k}": v for k, v in val_ious.items()})
        self.area_pred["val"].zero_()
        self.area_true["val"].zero_()
        self.area_inter["val"].zero_()

    def on_test_epoch_end(self):
        test_ious = self.compute_ious("test")
        self.log_dict({f"test_iou/{k}": v for k, v in test_ious.items()})
        self.area_pred["test"].zero_()
        self.area_true["test"].zero_()
        self.area_inter["test"].zero_()
