import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, Linear
from torch_geometric.loader import DataLoader
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import SubsetRandomSampler
import time
from tqdm import tqdm

from difftactile.cnn.dataset import *
from difftactile.cnn.common import *


class CurriculumCallback(pl.Callback):
    def __init__(self, data_module, all_stats):
        self.data_module = data_module
        self.all_stats = all_stats

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'execute CurriculumCallback.on_train_epoch_start')
        epoch = trainer.current_epoch
        n = 2
        k = 10
        if epoch < n:
            difficulty = 0.0
        elif epoch >= n and epoch < (n+k):
            difficulty = (epoch+1-n) / k
        else:
            difficulty = 1.0
        datasets = self.data_module.get_datasets()
        for i in range(len(datasets)):
            datasets[i].set_difficulty_level(difficulty)
        pl_module.set_stats(self.all_stats[difficulty])

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
    def __init__(self, alpha=0.5, gamma=3.0):
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
            lr,
            node_channels=SYSTEM_PARAMS.gnn.num_node_features,
            # node_channels=1,
            edge_channels=SYSTEM_PARAMS.gnn.num_edge_features,
            # edge_channels=2,
            hidden_channels=SYSTEM_PARAMS.gnn.num_hidden_channels,
            out_channels=1,
            num_layers=2,
            tversky_weight=0.0,
            focal_weight=1.0,
        ):
        super().__init__()
        
        # Add previous_lr attribute to track changes
        self._previous_lr = None
        
        # Node networks for GINEConv
        self.node_net1 = nn.Sequential(
            nn.Linear(node_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.node_net2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # GINEConv layers
        self.conv1 = GINEConv(self.node_net1, edge_dim=edge_channels)
        self.conv2 = GINEConv(self.node_net2, edge_dim=edge_channels)
        
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
    
    def set_stats(self, stats):
        self.focal_loss.alpha = stats['alpha_pos']
        print(f'focal_loss.alpha={self.focal_loss.alpha}')
    
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
        # First GINEConv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Second GINEConv layer
        out = self.conv2(x, edge_index, edge_attr)
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
        
        is_val_stage = f'{stage}' == 'val'

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
                factor=0.1,          # Multiply LR by this factor when reducing
                patience=3,          # Number of epochs with no improvement after which LR will be reduced
                min_lr=1e-6,        # Don't reduce LR below this value
                cooldown=1,         # Number of epochs to wait before resuming normal operation after LR has been reduced
                threshold=1e-4,
                threshold_mode='rel'
            ),
            "monitor": "val_fg_iou",   # Quantity to monitor
            "interval": "epoch",
            "frequency": 1
        }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def on_train_epoch_start(self):
        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        # Print only if learning rate has changed
        if self._previous_lr != current_lr:
            print(f"\nLearning rate changed from {self._previous_lr} to {current_lr:.2e}")
            self._previous_lr = current_lr
            
        # Still log to tensorboard but don't show in progress bar
        self.log('learning_rate', current_lr, prog_bar=False)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, train_subset_size, val_subset_size, batch_size, num_workers, seed=42):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.current_train_indices = None
        self.current_val_indices = None
    
    def get_datasets(self):
        return [
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        ]

    def setup(self, stage=None):
        # Initialize current_indices for the first epoch
        self._select_new_subset()

    def _select_new_subset(self):
        len_train = len(self.train_dataset)
        train_subset_size = min(
            len_train,
            self.train_subset_size
        )
        self.current_train_indices = np.random.choice(
            len_train,
            train_subset_size,
            replace=True
        )
        len_val = len(self.val_dataset)
        val_subset_size = min(
            len_val,
            self.val_subset_size
        )
        self.current_val_indices = np.random.choice(
            len_val,
            val_subset_size,
            replace=False
        )

    def train_dataloader(self):
        if self.current_train_indices is None:
            self._select_new_subset()
        sampler = SubsetRandomSampler(self.current_train_indices, generator=self.generator)
        print(f'train dataset difficulty: {self.train_dataset.difficulty_fyi}')
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False
        )

    def val_dataloader(self):
        if self.current_val_indices is None:
            self._select_new_subset()
        sampler = SubsetRandomSampler(self.current_val_indices, generator=self.generator)
        print(f'val dataset difficulty: {self.train_dataset.difficulty_fyi}')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False
        )

    def test_dataloader(self):
        print(f'test dataset difficulty: {self.train_dataset.difficulty_fyi}')
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False
        )

    def on_train_epoch_start(self):
        self._select_new_subset()


def main():
    BATCH_SIZE = 512
    NUM_EPOCHS = 8
    NUM_WORKERS = 16
    TRAIN_EPOCH_SUBSET_SIZE = BATCH_SIZE * 64
    VAL_EPOCH_SUBSET_SIZE = BATCH_SIZE * 8
    LR = 1e-3

    logger = TensorBoardLogger("lightning_logs", name="gnn", version=f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    full_dataset = MyDataset(
        data_dir=SYSTEM_PARAMS.files.dataset_root_today_reordered
    )
    train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
        full_dataset, train_size=0.70, val_size=0.15, test_size=0.15
    )
    all_stats = {}
    for i in range(11):
        if i != 10:
            continue
        difficulty = i / 10
        train_dataset.set_difficulty_level(difficulty)
        stats = compute_stats(train_dataset, BATCH_SIZE)
        all_stats[difficulty] = stats

        alpha_neg = stats['alpha_neg']
        alpha_pos = stats['alpha_pos']
        print(f'difficulty: {difficulty}; pos:neg = {alpha_neg:.2f}:{alpha_pos:.2f}')
    
    init_difficulty = 0.0
    final_difficulty = 1.0
    train_dataset.set_difficulty_level(final_difficulty)
    val_dataset.set_difficulty_level(final_difficulty)
    test_dataset.set_difficulty_level(final_difficulty)
    train_dataset.set_stats(all_stats[final_difficulty])
    val_dataset.set_stats(all_stats[final_difficulty])
    test_dataset.set_stats(all_stats[final_difficulty])

    # Create a single datamodule for training, validation and testing
    data_module = MyDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_subset_size=TRAIN_EPOCH_SUBSET_SIZE,
        val_subset_size=VAL_EPOCH_SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    test_data = {
        'dataset': test_dataset,
        'num_workers': NUM_WORKERS,
        'dataset_stats': all_stats
    }
    with open(SYSTEM_PARAMS.files.test_loader_gnn, 'wb') as f:
        pickle.dump(test_data, f)
    

    model = GNN(lr=LR)
    model.set_stats(all_stats[final_difficulty])

    checkpoint_cb = ModelCheckpoint(
        monitor="val_fg_iou",
        mode="max",
        save_top_k=1,
        filename="best-model",
    )
    early_stopping = EarlyStopping(
        monitor="val_fg_iou",
        mode="max",
        patience=NUM_EPOCHS*2,
        min_delta=1e-4,
        verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS, 
        accelerator="auto",
        enable_checkpointing=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            checkpoint_cb, 
            # early_stopping,
            # CurriculumCallback(data_module, all_stats)
        ],
        reload_dataloaders_every_n_epochs=1
    )

    start = time.perf_counter()
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    end = time.perf_counter()
    duration = end - start
    print(f"Training and testing completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), SYSTEM_PARAMS.files.final_segmentation_model_gnn)


def choose_optimal_threshold():
    BATCH_SIZE = 512
    NUM_WORKERS = 16
    VAL_EPOCH_SUBSET_SIZE = BATCH_SIZE * 8  # Using same size as in main()
    
    # Load test data from file
    with open(SYSTEM_PARAMS.files.test_loader_gnn, 'rb') as f:
        test_data = pickle.load(f)
    test_dataset = test_data['dataset']
    test_dataset.set_difficulty_level(1.0)
    
    # Create data module using test dataset as validation dataset
    data_module = MyDataModule(
        train_dataset=test_dataset,  # Not used but needed for initialization
        val_dataset=test_dataset,    # We'll use this for threshold optimization
        test_dataset=test_dataset,   # Not used but needed for initialization
        train_subset_size=VAL_EPOCH_SUBSET_SIZE,
        val_subset_size=VAL_EPOCH_SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Load model
    model = GNN(lr=-1)  # Dummy values since we're not training
    model.load_state_dict(torch.load(SYSTEM_PARAMS.files.final_segmentation_model_gnn))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Test different thresholds
    thresholds = np.linspace(0.4, 0.6, 5)  # Test thresholds from 0.1 to 0.9 in steps of 0.05
    best_threshold = 0.5  # Default threshold
    best_fg_iou = 0.0
    
    print("\nTesting different thresholds:")
    print("Threshold | Foreground IoU")
    print("-" * 25)
    # Get a fresh validation loader for each threshold to ensure fair comparison
    val_loader = data_module.val_dataloader()
    
    with torch.no_grad():
        for threshold in thresholds:
            total_fg_iou = 0.0
            num_batches = 0
            
            for batch, _ in val_loader:  # Note: our loader returns (batch, _)
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                logits = logits.squeeze(-1)  # Remove the channel dimension
                mask = batch.mask
                logits = logits[mask]
                
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                iou_scores = GNN.iou_score(preds, batch.y)
                total_fg_iou += iou_scores['fg_iou'].item()
                num_batches += 1
            
            avg_fg_iou = total_fg_iou / num_batches
            print(f"{threshold:.2f}     | {avg_fg_iou:.4f}")
            
            if avg_fg_iou > best_fg_iou:
                best_fg_iou = avg_fg_iou
                best_threshold = threshold
    
    print("\nBest results:")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Best foreground IoU: {best_fg_iou:.4f}")


def compute_stats(dataset, batch_size):
    len_train = len(dataset)
    train_subset_size = batch_size
    ixs = np.random.choice(
        len_train,
        train_subset_size,
        replace=False
    )
    num_pos = 0
    num_neg = 0
    edge_attr_all = []
    x_all = []
    pos_all = []
    for ix in tqdm(ixs, desc="Computing stats"):
        pyg, _ = dataset[ix]
        y = pyg.y.cpu().numpy()
        pos = y.sum()
        neg = y.shape[0] - pos
        num_pos += pos
        num_neg += neg

        x = pyg.x.cpu().numpy()
        edge_attr = pyg.edge_attr.cpu().numpy()
        pos = pyg.pos.cpu().numpy()
        x_all.append(x)
        edge_attr_all.append(edge_attr)
        pos_all.append(pos)

    alpha_pos = num_neg / (num_neg + num_pos)
    alpha_neg = num_pos / (num_neg + num_pos)

    edge_attr_all = np.array(edge_attr_all)
    x_all = np.array(x_all)
    pos_all = np.array(pos_all)

    edge_attr_all = edge_attr_all.reshape((-1, edge_attr_all.shape[-1]))
    x_all = x_all.reshape((-1, x_all.shape[-1]))
    pos_all = pos_all.reshape((-1, pos_all.shape[-1]))

    edge_attr_mean = np.mean(edge_attr_all, axis=0)
    x_mean = np.mean(x_all, axis=0)
    pos_mean = np.mean(pos_all, axis=0)

    edge_attr_std = np.std(edge_attr_all, axis=0)
    x_std = np.std(x_all, axis=0)
    pos_std = np.std(pos_all, axis=0)

    return {
        'alpha_pos': alpha_pos,
        'alpha_neg': alpha_neg,

        'edge_attr_mean': edge_attr_mean,
        'x_mean': x_mean,
        'pos_mean': pos_mean,

        'edge_attr_std': edge_attr_std,
        'x_std': x_std,
        'pos_std': pos_std,
    }

