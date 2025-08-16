import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from difftactile.cnn.dataset import *



class GNN(pl.LightningModule):
    def __init__(
            self, 
            in_channels=2, 
            hidden_channels=16, 
            out_channels=2
        ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


if __name__ == "__main__":
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
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

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

