import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.loader import DataLoader
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from difftactile.cnn.dataset import *
from difftactile.cnn.common import *


class LossWeightScheduler(pl.Callback):
    def __init__(self, max_epochs):
        self.tversky_alpha = [0.5, 0.8]
        self.tversky_beta = [0.5, 0.2]
        self.focal_alpha = [0.5, 0.8]
        self.focal_gamma = [0.0, 2.0]
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        ratio = epoch/(self.max_epochs-1)
        pl_module.set_loss_weights(
            self.compute_val(*self.tversky_alpha, ratio),
            self.compute_val(*self.tversky_beta, ratio),
            self.compute_val(*self.focal_alpha, ratio),
            self.compute_val(*self.focal_gamma, ratio),
        )
    
    def compute_val(self, min, max, ratio):
        diff = max - min
        return min + diff * ratio


class GNNTverskyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.25, smooth=1e-6):
        """Tversky Loss for GNN node classification with imbalanced data.
        
        Specifically designed for node-level binary classification where each node
        is classified as either part of a vein (1) or not (0).
        
        Args:
            alpha (float): Weight of false negatives, range [0,1].
                         Higher alpha puts more emphasis on finding vein nodes.
                         Default 0.9 to prioritize vein detection.
            beta (float): Weight of false positives, range [0,1].
                         Lower beta allows more false positives.
                         Default 0.1 to be more permissive of false positives.
            smooth (float): Smoothing constant to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw logits from GNN of shape (num_nodes,)
            targets: Binary target labels of shape (num_nodes,)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute per-node metrics
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        
        # Compute Tversky index
        numerator = TP + self.smooth
        denominator = TP + self.alpha * FN + self.beta * FP + self.smooth
        tversky = numerator / denominator
        
        return 1 - tversky


class GNNFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.0):
        """Focal Loss for GNN node classification.
        
        Specifically designed for node-level binary classification where the focus
        is on hard examples and handling class imbalance.
        
        Args:
            alpha (float): Weight factor for the positive class (vein nodes), range [0,1].
                         Default 0.75 to give more weight to vein nodes.
            gamma (float): Focusing parameter that reduces loss for well-classified nodes.
                         Default 2.0 as per the original paper.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw logits from GNN of shape (num_nodes,)
            targets: Binary target labels of shape (num_nodes,)
        """
        # Compute binary cross entropy loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        # Compute probabilities for focusing
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply class weights
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class GNN(pl.LightningModule):
    def __init__(
            self, 
            node_channels=SYSTEM_PARAMS.gnn.num_node_features,
            edge_channels=SYSTEM_PARAMS.gnn.num_edge_features,
            hidden_channels=128,
            out_channels=1,
            num_heads=4,  # Number of attention heads
            lr=1e-2,
            tversky_weight=0.5,
            focal_weight=0.5
        ):
        super().__init__()
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        internal_edge_channels = hidden_channels
        
        # GATv2 layers with edge features and multi-head attention
        self.conv1 = GATv2Conv(
            in_channels=node_channels,
            out_channels=hidden_channels // num_heads,  # Split channels across heads
            heads=num_heads,
            edge_dim=internal_edge_channels,  # Edge features dimension
            add_self_loops=False  # We'll handle temporal connections explicitly
        )
        
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            edge_dim=internal_edge_channels,
            add_self_loops=False
        )
        
        # Final prediction layer
        self.conv3 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            heads=1,  # Single head for final prediction
            edge_dim=internal_edge_channels,
            add_self_loops=False
        )
        
        # Layer norms for better training stability
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        # Initialize loss functions
        self.tversky_loss = GNNTverskyLoss()
        self.focal_loss = GNNFocalLoss()
        
        # Loss weights
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
        # Learning rate
        self.lr = lr
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
    
    def set_loss_weights(
            self,
            tversky_alpha,
            tversky_beta,
            focal_alpha,
            focal_gamma
    ):
        self.tversky_loss.alpha = tversky_alpha
        self.tversky_loss.beta = tversky_beta
        self.focal_loss.alpha = focal_alpha
        self.focal_loss.gamma = focal_gamma

    @staticmethod
    def iou_score(preds, targets, eps=1e-6):
        """
        Compute IoU scores for binary node classification.
        
        Args:
            preds: Binary predictions tensor of shape (num_nodes,)
            targets: Binary ground truth tensor of shape (num_nodes,)
            eps: Small constant to avoid division by zero
            
        Returns:
            Dictionary containing foreground and background IoU scores.
            Returns 0 for IoU when union is 0 (no overlap and no predictions/targets).
        """
        # Convert inputs to float for calculations
        preds = preds.float()
        targets = targets.float()
        
        # Compute foreground IoU (class 1)
        fg_intersection = (preds * targets).sum()
        fg_union = (preds + targets).sum() - fg_intersection
        if fg_union > eps:
            fg_iou = fg_intersection / fg_union
        else:
            fg_iou = torch.tensor(0., device=preds.device)
        
        # Compute background IoU (class 0)
        bg_preds = 1 - preds
        bg_targets = 1 - targets
        bg_intersection = (bg_preds * bg_targets).sum()
        bg_union = (bg_preds + bg_targets).sum() - bg_intersection
        if bg_union > eps:
            bg_iou = bg_intersection / bg_union
        else:
            bg_iou = torch.tensor(0., device=preds.device)
        
        return {
            'fg_iou': fg_iou,
            'bg_iou': bg_iou,
            'macro_iou': (fg_iou + bg_iou) / 2
        }

    def forward(self, x, edge_index, edge_attr):
        # Embed edge features
        edge_features = self.edge_embedding(edge_attr)
        # edge_features = edge_attr
        
        # First conv layer
        h1 = self.conv1(x, edge_index, edge_attr=edge_features)
        h1 = F.relu(self.norm1(h1))
        
        # Second conv layer with residual connection
        h2 = self.conv2(h1, edge_index, edge_attr=edge_features)
        h2 = F.relu(self.norm2(h2 + h1))  # Residual connection
        
        # Final conv layer
        out = self.conv3(h2, edge_index, edge_attr=edge_features)
        return out

    def shared_step(self, batch, stage):
        batch, _ = batch
        # Forward pass with edge features
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        out = out.squeeze(-1)  # Remove the channel dimension
        mask = batch.mask
        out = out[mask]
        
        # Calculate losses
        tversky_loss = self.tversky_loss(out, batch.y.float())
        focal_loss = self.focal_loss(out, batch.y.float())
        
        # Combine losses using weights
        loss = self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
        
        # Calculate predictions
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        
        # Calculate IoU metrics per graph in batch
        B = batch.num_graphs
        batch_metrics = {
            'fg_iou': torch.zeros(B, device=preds.device),
            'bg_iou': torch.zeros(B, device=preds.device),
            'macro_iou': torch.zeros(B, device=preds.device)
        }
        
        # Calculate IoU for each graph separately
        for i in range(B):
            iou_mask = batch.batch[mask] == i
            graph_preds = preds[iou_mask]
            graph_targets = batch.y[iou_mask]
            metrics = self.iou_score(graph_preds, graph_targets)
            batch_metrics['fg_iou'][i] = metrics['fg_iou']
            batch_metrics['bg_iou'][i] = metrics['bg_iou']
            batch_metrics['macro_iou'][i] = metrics['macro_iou']
        
        # Average metrics across batch
        metrics = {k: v.mean() for k, v in batch_metrics.items()}
        
        # Log all metrics
        self.log(f"{stage}_tversky_loss", tversky_loss, prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_focal_loss", focal_loss, prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_combined_loss", loss, prog_bar=False, batch_size=batch.num_graphs)
        self.log(f"{stage}_fg_iou", metrics['fg_iou'], prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_bg_iou", metrics['bg_iou'], prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_macro_iou", metrics['macro_iou'], prog_bar=False, batch_size=batch.num_graphs)
        
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
                min_lr=1e-6,        # Don't reduce LR below this value
                cooldown=1,         # Number of epochs to wait before resuming normal operation after LR has been reduced
            ),
            "monitor": "val_fg_iou",   # Quantity to monitor
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    NUM_WORKERS = 16
    LR = 1e-4

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

    model = GNN(lr=LR)


    checkpoint_cb = ModelCheckpoint(
        monitor="val_fg_iou",
        mode="max",
        save_top_k=1,
        filename="best-model",
    )
    early_stopping = EarlyStopping(
        monitor="val_fg_iou",
        mode="max",
        patience=5,
        min_delta=1e-4,
        verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS, 
        accelerator="auto",
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            LossWeightScheduler(NUM_EPOCHS)
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), SYSTEM_PARAMS.files.final_segmentation_model_gnn)


if __name__ == "__main__":
    main()
