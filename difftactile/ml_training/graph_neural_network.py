import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import numpy as np

from exploratory_data_analysis import *

class GNNClassifier(pl.LightningModule):
    def __init__(self, node_dim=4):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.val_accuracy = Accuracy(task="binary")

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(F.relu(self.conv3(x, edge_index)))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification head
        return torch.sigmoid(self.classifier(x))

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = F.binary_cross_entropy(y_hat, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = F.binary_cross_entropy(y_hat, batch.y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_hat, batch.y.int()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Preprocessing (run once before training)
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()
    # Build graph dataset:
    graphs = []
    for i in range(len(predict_markers_snapshots)):
        # Node features: [displacement_x, displacement_y] + [default_x, default_y]
        node_features = np.hstack([
            displacements[i], 
            virtual_markers_snapshots[0],
        ])
        
        # Create kNN graph (k=8)
        pos_tensor = torch.tensor(virtual_markers_snapshots[0], dtype=torch.float)
        dist_matrix = torch.cdist(pos_tensor, pos_tensor)
        k = 8
        _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
        
        # Create edge_index
        src = torch.repeat_interleave(torch.arange(len(pos_tensor)), k)
        dst = indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        
        # Create PyG Data object
        graphs.append(Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor([ground_truth_labels[i]], dtype=torch.float)
        ))

    # Split dataset
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)

    # Initialize and train model
    model = GNNClassifier()
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()