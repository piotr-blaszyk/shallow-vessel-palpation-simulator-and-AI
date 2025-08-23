import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric_temporal.nn.recurrent import GConvGRU

class BiDirectionalWaveFrontNetWithProjections(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__()

        # Explicit node and edge projections to latent space
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Spatial encoder (GINEConv uses projected node + edge features)
        self.gine = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=hidden_dim)

        # Bidirectional temporal GRUs
        self.gru_fwd = GConvGRU(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.gru_bwd = GConvGRU(hidden_dim, hidden_dim, edge_dim=hidden_dim)

        # Classifier head on concatenated forward+backward states
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        """
        Args:
            x_seq: [T, N, F_node] node features
            edge_index_seq: list of T edge_index tensors [2, E]
            edge_attr_seq: list of T edge_attr tensors [E, F_edge]
        Returns:
            out_seq: [T, N] per-node logits for each timestep
        """
        T, N, _ = x_seq.shape

        # Project all nodes and edges into latent space
        x_seq_latent = [self.node_proj(x_seq[t]) for t in range(T)]
        edge_seq_latent = [self.edge_proj(edge_attr_seq[t]) for t in range(T)]

        # Forward GRU
        h_fwd = None
        out_fwd = []
        for t in range(T):
            x_lat = x_seq_latent[t]
            edge_lat = edge_seq_latent[t]
            x_lat = self.gine(x_lat, edge_index_seq[t], edge_lat)
            h_fwd = self.gru_fwd(x_lat, edge_index_seq[t], h_fwd, edge_lat)
            out_fwd.append(h_fwd)

        # Backward GRU
        h_bwd = None
        out_bwd = []
        for t in reversed(range(T)):
            x_lat = x_seq_latent[t]
            edge_lat = edge_seq_latent[t]
            x_lat = self.gine(x_lat, edge_index_seq[t], edge_lat)
            h_bwd = self.gru_bwd(x_lat, edge_index_seq[t], h_bwd, edge_lat)
            out_bwd.append(h_bwd)
        out_bwd = list(reversed(out_bwd))

        # Concatenate forward + backward hidden states
        h_cat = [torch.cat([f, b], dim=-1) for f, b in zip(out_fwd, out_bwd)]
        h_cat = torch.stack(h_cat, dim=0)  # [T, N, 2*hidden_dim]

        # Per-node classification logits
        out_seq = self.classifier(h_cat).squeeze(-1)  # [T, N]
        return out_seq
