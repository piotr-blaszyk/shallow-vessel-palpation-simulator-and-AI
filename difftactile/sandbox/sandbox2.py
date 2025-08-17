import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv

# ---------------------------
# 1. Synthetic dataset
# ---------------------------
class SpatioTemporalToyDataset(Dataset):
    def __init__(self, num_samples=100, num_frames=3, num_points=5):
        super().__init__()
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_points = num_points

    def len(self):
        return self.num_samples

    def get(self, idx):
        T, N = self.num_frames, self.num_points
        num_nodes = T * N

        # Node positions (x,y) random walk
        pos = torch.randn(num_nodes, 2).cumsum(dim=0)
        x = pos  # node features

        # Spatial + temporal edges
        edge_list, edge_attr = [], []
        for t in range(T):
            offset = t * N
            for i in range(N):
                for j in range(N):
                    if i != j:  # spatial edge
                        src, dst = offset + i, offset + j
                        edge_list.append([src, dst])
                        disp = pos[dst] - pos[src]
                        edge_attr.append(torch.cat([disp, torch.norm(disp).unsqueeze(0)]))

        for t in range(T - 1):  # temporal edges
            for i in range(N):
                src, dst = t * N + i, (t + 1) * N + i
                edge_list.append([src, dst])
                disp = pos[dst] - pos[src]
                edge_attr.append(disp)

        edge_index = torch.tensor(edge_list).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        # Labels: only for central frame nodes
        labels = torch.randint(0, 2, (N,))  # binary
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[(T // 2) * N : (T // 2 + 1) * N] = True

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels, mask=mask)


# ---------------------------
# 2. GNN model
# ---------------------------
class SpatioTemporalGNN(pl.LightningModule):
    def __init__(self, in_dim=2, hidden_dim=32, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)  # binary classification
        self.lr = lr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        out = self.lin(x).squeeze(-1)  # [num_nodes]
        return out

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        mask = batch.mask
        loss = F.binary_cross_entropy_with_logits(logits[mask], batch.y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        mask = batch.mask
        loss = F.binary_cross_entropy_with_logits(logits[mask], batch.y.float())
        preds = (torch.sigmoid(logits[mask]) > 0.5).long()
        acc = (preds == batch.y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------
# 3. Train
# ---------------------------
if __name__ == "__main__":
    dataset = SpatioTemporalToyDataset(num_samples=200, num_frames=3, num_points=5)
    train_loader = DataLoader(dataset[:150], batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset[150:], batch_size=16)

    model = SpatioTemporalGNN()

    trainer = pl.Trainer(max_epochs=5, accelerator="cpu", devices=1)
    trainer.fit(model, train_loader, val_loader)
