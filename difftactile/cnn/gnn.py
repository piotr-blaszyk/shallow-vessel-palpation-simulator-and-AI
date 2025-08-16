import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from difftactile.cnn.dataset import *
from difftactile.cnn.common import *


class GNN(pl.LightningModule):
    def __init__(
            self, 
            in_channels=2, 
            hidden_channels=16, 
            out_channels=1,  # Changed to 1 for binary classification
            lr=1e-2,
            tversky_weight=0.5,
            focal_weight=0.5
        ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Initialize loss functions
        self.tversky_loss = TverskyLoss()
        self.focal_loss = FocalLoss()
        
        # Loss weights
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
        # Learning rate
        self.lr = lr
        
        # Save hyperparameters for logging
        self.save_hyperparameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def shared_step(self, batch, stage):
        batch, _ = batch
        # Forward pass
        out = self(batch.x, batch.edge_index)
        out = out.squeeze(-1)  # Remove the channel dimension
        
        # Calculate losses
        tversky_loss = self.tversky_loss(out, batch.y.float())
        focal_loss = self.focal_loss(out, batch.y.float())
        
        # Combine losses using weights
        loss = self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
        
        # Calculate predictions
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        
        # Reshape predictions and targets for IoU calculation
        # Assuming batch.y contains node-level binary labels
        B = batch.num_graphs  # number of graphs in batch
        max_nodes = max(batch.batch.bincount())  # maximum number of nodes in any graph
        
        # Initialize tensors with padding
        preds_reshaped = torch.zeros(B, max_nodes, device=preds.device)
        targets_reshaped = torch.zeros(B, max_nodes, device=batch.y.device)
        
        # Fill in the actual values
        for i in range(B):
            mask = batch.batch == i
            num_nodes = mask.sum()
            preds_reshaped[i, :num_nodes] = preds[mask]
            targets_reshaped[i, :num_nodes] = batch.y[mask]
        
        # Add a dummy H,W dimension of size 1 to match the expected shape
        preds_reshaped = preds_reshaped.unsqueeze(-1).unsqueeze(-1)
        targets_reshaped = targets_reshaped.unsqueeze(-1).unsqueeze(-1)
        
        # Calculate IoU metrics
        metrics = Common.iou_score(preds_reshaped, targets_reshaped)
        
        # Log all metrics
        self.log(f"{stage}_tversky_loss", tversky_loss, prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_focal_loss", focal_loss, prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_combined_loss", loss, prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_fg_iou", metrics['fg_iou'], prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_bg_iou", metrics['bg_iou'], prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_macro_iou", metrics['macro_iou'], prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_detection_rate", metrics['detection_rate'], prog_bar=False, batch_size=batch.num_graphs)
        
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
                mode='max',           # Maximize the monitored quantity (validation IoU)
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


def main():
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    NUM_WORKERS = 16
    LR = 1e-3

    logger = TensorBoardLogger("lightning_logs", name="segmentation_model")
    full_dataset = MyDataset(
        data_dir=SYSTEM_PARAMS.files.dataset_root
    )
    train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
        full_dataset, train_size=0.70, val_size=0.15, test_size=0.15
    )
    # Using PyTorch Geometric's DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    test_data = {
        'dataset': test_dataset,
        'num_workers': NUM_WORKERS
    }
    with open(SYSTEM_PARAMS.files.test_loader_gnn, 'wb') as f:
        pickle.dump(test_data, f)

    model = GNN()

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS, 
        accelerator="auto",
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=1
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), SYSTEM_PARAMS.files.final_segmentation_model_gnn)


if __name__ == "__main__":
    main()
