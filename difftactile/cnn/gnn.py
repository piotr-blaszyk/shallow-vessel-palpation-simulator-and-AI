import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np


class GNN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
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
    import numpy as np

    # Fake example data
    num_nodes = 10
    num_node_features = 5
    num_neighbours = 3
    num_edge_features = 2

    node_features = np.random.randn(num_nodes, num_node_features)
    adjacency = np.random.randint(0, num_nodes, size=(num_nodes, num_neighbours))
    edge_features = np.random.randn(num_nodes, num_neighbours, num_edge_features)
    ground_truth_labels = np.random.randint(0, 2, size=(num_nodes,))

    data = numpy_to_pyg(node_features, adjacency, edge_features, ground_truth_labels)

    model = GNN(in_channels=num_node_features, hidden_channels=16, out_channels=2)

    trainer = pl.Trainer(max_epochs=50, enable_checkpointing=False, logger=False)
    trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader([data], batch_size=1))
