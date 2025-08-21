"""
PyTorch-Geometric skeleton for per-node wavefront classification from marker positions only.
Assumptions:
 - Input: a sliding window of node positions for each sample: positions shape [W, N, 2]
 - Node identities are stable inside a window (so we can form per-node temporal sequences)
 - No amplitude readings; the model uses displacement/velocity derived from positions

Key components:
 - Node encoder: encodes velocity / short displacement history -> feature vector
 - EGNN-like spatial block: equivariant message passing using relative positions
 - Temporal encoder: GRU per node over the window of spatial embeddings
 - Classifier head: per-node binary prediction (near foreign object or far)

This is a minimal, readable skeleton to adapt to your dataset and training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected
from torch_cluster import knn_graph


# ------------------ Helper utilities ------------------

def compute_velocities(positions):
    # positions: [W, N, 2]
    # velocities: [W, N, 2] where v[t] = p[t] - p[t-1], v[0] = 0
    W, N, _ = positions.shape
    v = torch.zeros_like(positions)
    v[1:] = positions[1:] - positions[:-1]
    return v


def build_spatial_graph(positions_t, k=8, batch=None):
    # positions_t: [N,2] or [B*N,2] if batched (we'll assume single sample per call here)
    # returns edge_index [2, E] (undirected) for the k-NN graph
    # using torch_cluster.knn_graph (expects x: [num_nodes, dim])
    # if batch is provided (tensor of node->batch idx), pass to knn_graph
    if batch is None:
        edge_index = knn_graph(positions_t, k=k, loop=False)
    else:
        edge_index = knn_graph(positions_t, k=k, batch=batch, loop=False)
    edge_index = to_undirected(edge_index)
    return edge_index


# ------------------ EGNN-like layer ------------------
class EGNNLayer(nn.Module):
    """A simplified EGNN layer.
    - updates node features h and optionally updates coordinates p
    - messages are based on (h_i, h_j, ||p_i - p_j||^2)
    - coordinate update is optional and small
    """
    def __init__(self, in_feats, out_feats, coord_update=True):
        super().__init__()
        self.coord_update = coord_update
        self.phi_m = nn.Sequential(
            nn.Linear(2 * in_feats + 1, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats),
            nn.ReLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(out_feats, 1),
            nn.Tanh(),
        ) if coord_update else None
        self.node_mlp = nn.Sequential(
            nn.Linear(in_feats + out_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats),
        )

    def forward(self, h, pos, edge_index):
        # h: [N, F]
        # pos: [N, 2]
        # edge_index: [2, E]
        row, col = edge_index  # message from col -> row (i <- j)
        # relative positional scalar: squared distance
        rel = pos[row] - pos[col]  # [E, 2]
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # message input: [h_i, h_j, dist2]
        m_in = torch.cat([h[row], h[col], dist2], dim=-1)  # [E, 2F+1]
        m_ij = self.phi_m(m_in)  # [E, out_feats]

        # aggregate messages (sum)
        N = h.size(0)
        agg = torch.zeros(N, m_ij.size(-1), device=h.device)
        agg = agg.index_add(0, row, m_ij)

        # update node features
        h_out = self.node_mlp(torch.cat([h, agg], dim=-1))

        if self.coord_update:
            # coordinate update: compute a small displacement for pos
            # scalar gating s_ij from message features
            s = self.phi_x(m_ij)  # [E,1]
            # dir = (p_i - p_j) normalized-ish
            dir = rel  # use raw direction
            # weighted displacement per-edge
            disp_per_edge = dir * s
            disp = torch.zeros_like(pos)
            disp = disp.index_add(0, row, disp_per_edge)
            pos_out = pos + 0.01 * disp  # small step size
            return h_out, pos_out
        else:
            return h_out, pos


# ------------------ Model components ------------------
class NodeEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        # default in_dim = 4 -> [vx, vy, |v|, speed_dir?] but flexible
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, node_feat):
        return self.net(node_feat)


class SpatialEncoder(nn.Module):
    def __init__(self, F=64, n_layers=3, coord_update=False):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(F, F, coord_update=coord_update) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(F) for _ in range(n_layers)])

    def forward(self, h, pos, edge_index):
        for layer, norm in zip(self.layers, self.norms):
            h_res = h
            h, pos = layer(h, pos, edge_index)
            h = norm(h + h_res)
        return h, pos


class TemporalEncoderGRU(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=128, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

    def forward(self, seq):
        # seq: [N, W, F]
        out, h_n = self.gru(seq)  # out: [N, W, H]
        # return last time-step feature per node
        return out[:, -1, :]


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits [N]


# ------------------ Full Model ------------------
class TactileEGNNClassifier(nn.Module):
    def __init__(self, window=8, node_in_dim=4, F=64, spatial_layers=3, gru_hidden=128, coord_update=False, k=8):
        """
        window: number of frames in input window
        node_in_dim: dimensionality of per-node input (e.g., vx,vy,|v|,speed_mag)
        F: latent feature dim
        spatial_layers: number of equivariant spatial layers per frame
        gru_hidden: hidden dim for per-node GRU temporal encoder
        coord_update: whether EGNN layers update coordinates internally
        k: k-NN for spatial graph
        """
        super().__init__()
        self.window = window
        self.k = k
        self.node_encoder = NodeEncoder(in_dim=node_in_dim, hidden=F)
        self.spatial = SpatialEncoder(F=F, n_layers=spatial_layers, coord_update=coord_update)
        self.temporal = TemporalEncoderGRU(feat_dim=F, hidden_dim=gru_hidden)
        self.classifier = ClassifierHead(in_dim=gru_hidden)

    def forward(self, positions_window, node_mask=None):
        """
        positions_window: tensor [W, N, 2]
        node_mask: optional [N] boolean mask for active nodes
        returns: logits [N] for nodes at last time step
        """
        W, N, _ = positions_window.shape
        device = positions_window.device

        # compute velocities (or short displacement features)
        velocities = compute_velocities(positions_window)  # [W, N, 2]
        speed = torch.norm(velocities, dim=-1, keepdim=True)  # [W,N,1]
        # Node input features per time: [vx, vy, speed, maybe |vx-v_local| etc.]
        node_inputs = torch.cat([velocities, speed], dim=-1)  # [W, N, 3]
        # if you want a fixed size (e.g., 4), you can append zero column
        if node_inputs.shape[-1] < 4:
            pad = torch.zeros(W, N, 4 - node_inputs.shape[-1], device=device)
            node_inputs = torch.cat([node_inputs, pad], dim=-1)

        # Process each time-slice spatially
        spatial_embeddings = []
        pos_t = positions_window.clone()
        for t in range(W):
            pos = pos_t[t]  # [N,2]
            node_feat = node_inputs[t]  # [N, node_in_dim]
            h0 = self.node_encoder(node_feat)  # [N, F]
            # build spatial graph on current positions
            edge_index = build_spatial_graph(pos, k=self.k)
            h_sp, pos = self.spatial(h0, pos, edge_index)
            spatial_embeddings.append(h_sp)
            pos_t[t] = pos  # if coords updated inside model; otherwise unchanged

        # Stack to sequence per node: [N, W, F]
        seq = torch.stack(spatial_embeddings, dim=1).transpose(0, 1)  # orig: list [W][N,F] -> [W,N,F]
        seq = seq.transpose(0, 1)  # -> [N,W,F]

        # Temporal encoder per node
        h_final = self.temporal(seq)  # [N, gru_hidden]

        logits = self.classifier(h_final)  # [N]
        return logits


# ------------------ Training skeleton ------------------

def train_step(model, optimizer, positions_window, labels, pos_weight=1.0):
    # positions_window: [W, N, 2]
    # labels: [N] binary (0/1)
    model.train()
    optimizer.zero_grad()
    logits = model(positions_window)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=logits.device))
    loss = loss_fn(logits, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


# ------------------ Example usage ------------------
if __name__ == '__main__':
    # toy example
    W = 8
    N = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TactileEGNNClassifier(window=W, node_in_dim=4, F=64, spatial_layers=2, gru_hidden=128, coord_update=False, k=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # random toy positions (replace with your data loader)
    positions_window = torch.randn(W, N, 2, device=device)  # replace with actual trajectories
    labels = torch.zeros(N, device=device)
    # set some random nodes as positives
    labels[10:25] = 1.0

    loss = train_step(model, opt, positions_window, labels, pos_weight=5.0)
    print('toy loss', loss)

# ------------------ Notes & TODOs ------------------
# - Replace build_spatial_graph with a batched version when training multiple samples per batch.
# - node_mask and handling of missing nodes: if nodes can disappear/appear, handle padding and masks.
# - You can add data augmentations (global rotation / translation) during training.
# - If coordinates are noisy, consider enabling coord_update in EGNN and add a coordinate-denoising loss.
# - For imbalanced labels, use focal loss or sampling strategies.

