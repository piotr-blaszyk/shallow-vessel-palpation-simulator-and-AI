import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

from exploratory_data_analysis import *

# Preprocessing: Convert marker displacements to heatmaps
def displacements_to_heatmaps(displacements, default_positions, grid_size=32):
    """
    Convert marker displacements to 2-channel heatmaps
    Channel 0: x-displacement magnitude
    Channel 1: y-displacement magnitude
    """
    heatmaps = np.zeros((len(displacements), 2, grid_size, grid_size))
    
    # Normalize positions to [0, 1]
    min_vals = default_positions.min(axis=0)
    max_vals = default_positions.max(axis=0)
    norm_pos = (default_positions - min_vals) / (max_vals - min_vals)
    
    for i, disp in enumerate(displacements):
        # Compute displacement magnitude
        mag = np.linalg.norm(disp, axis=1)
        
        # Convert positions to grid coordinates
        grid_pos = (norm_pos * (grid_size - 1)).astype(int)
        
        # Assign to heatmap channels
        for j, (x, y) in enumerate(grid_pos):
            heatmaps[i, 0, x, y] = disp[j, 0]  # x-displacement
            heatmaps[i, 1, x, y] = disp[j, 1]  # y-displacement
            
    return heatmaps

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (2, 32, 32)
            nn.Conv2d(2, 16, 3, padding=1),  # 16x32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x16
            
            nn.Conv2d(16, 32, 3, padding=1),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x8x8
            
            nn.Conv2d(32, 64, 3, padding=1),  # 64x8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x4x4
            
            nn.Flatten(),
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
        )
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_hat, y.int()), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

def main():
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()

    # Convert to heatmaps
    X = displacements_to_heatmaps(displacements, virtual_markers_snapshots[0], grid_size=32)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, ground_truth_labels, test_size=0.2, random_state=42)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train the model
    model = CNNClassifier()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()