import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

from exploratory_data_analysis import *

class TumorClassifier(pl.LightningModule):
    def __init__(self, input_dim=254):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.layers(x))

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
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()
    X = displacements.reshape(displacements.shape[0], -1)  # Shape: [n, 254]
    y = ground_truth_labels  # Shape: [n]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Train the model
    model = TumorClassifier()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
