import os
import pickle
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from tqdm import tqdm
from torch_geometric.nn import global_add_pool
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import matplotlib
from difftactile.main.display import is_headless, show_plots
# Pick a non-interactive backend BEFORE pyplot is imported when there is no
# display. show_plots() already guards the blocking plt.show(), but that is too
# late for plt.figure(), which instantiates a Tk window as soon as it is called
# and would otherwise raise TclError on a headless/SSH/container run.
if is_headless():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from difftactile.cnn.common import *
from difftactile.cnn.dataset import *
from difftactile.cnn.curve_plots import DECISION_THRESHOLD, plot_pr, plot_roc
from difftactile.main.seeding import resolve_seed, seed_everything, seed_worker
from difftactile.main.paths import repo_path


# Configuration currently being run, set by run_scenario() and read by
# _retrained_path() so each configuration's training artifacts get their own
# filenames. None when a training function is called directly rather than
# through the dispatcher.
_ACTIVE_CONFIG = None


# Directory a training run should write its artifacts into, instead of the
# default `*_retrained_<config>` names beside the published ones. Set by the seed
# sweep (see cnn/seed_sweep.py) so each seed's checkpoint is preserved rather
# than overwritten by the next. Read from the environment because the sweep runs
# each seed in a separate process.
ARTIFACT_DIR_ENV_VAR = "DIFFTACTILE_ARTIFACT_DIR"


def _retrained_path(rel):
    """Path for an artifact produced by a *new* training run.

    Training would otherwise overwrite the published checkpoint and test-loader
    pickle that the evaluation scenarios read, so by default a `_retrained`
    suffix is inserted before the extension. Set
    DIFFTACTILE_OVERWRITE_PUBLISHED=1 to write over the published files instead.

    The configuration name is included in the suffix because `train_on_sim()`
    backs both A-to-B and A-to-C and writes to the same `*_sim` artifacts:
    without it, running the two in sequence would leave the first one's
    checkpoint silently replaced by the second's. Re-running the *same*
    configuration still overwrites in place, which is the intended behaviour.

    $DIFFTACTILE_ARTIFACT_DIR overrides all of that: the artifact keeps its
    basename but lands in that directory. The seed sweep uses this to give each
    seed its own folder, since the in-place overwrite above would otherwise
    leave only the last seed's weights. The BASENAME IS PRESERVED so that
    everything reading these artifacts - `_load_stats()`, the viewer, the
    scenario table - recognises them by the same `*_sim` / `*_meat` naming it
    already uses; only the directory changes.
    """
    artifact_dir = os.environ.get(ARTIFACT_DIR_ENV_VAR)
    if artifact_dir:
        base, ext = os.path.splitext(os.path.basename(rel))
        suffix = f"_retrained_{_ACTIVE_CONFIG}" if _ACTIVE_CONFIG else "_retrained"
        return os.path.join(artifact_dir, f"{base}{suffix}{ext}")
    if os.environ.get("DIFFTACTILE_OVERWRITE_PUBLISHED", "0") == "1":
        return repo_path(rel)
    base, ext = os.path.splitext(rel)
    suffix = f"_retrained_{_ACTIVE_CONFIG}" if _ACTIVE_CONFIG else "_retrained"
    return repo_path(f"{base}{suffix}{ext}")


# Kept as a thin alias so existing call sites read unchanged; the policy itself
# now lives in difftactile/main/display.py, which defaults to *not* blocking
# unless DIFFTACTILE_INTERACTIVE=1 is set.
_show_plots = show_plots


class CurriculumCallback(pl.Callback):
    def __init__(self, data_module, all_stats):
        self.data_module = data_module
        self.all_stats = all_stats

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"execute CurriculumCallback.on_train_epoch_start")
        epoch = trainer.current_epoch
        n = 2
        k = 10
        if epoch < n:
            difficulty = 0.0
        elif epoch >= n and epoch < (n + k):
            difficulty = (epoch + 1 - n) / k
        else:
            difficulty = 1.0
        datasets = self.data_module.get_datasets()
        for i in range(len(datasets)):
            datasets[i].set_difficulty_level(difficulty)
        pl_module.set_stats(self.all_stats[difficulty])


class MyTverskyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.25, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        numerator = TP + self.smooth
        denominator = TP + self.alpha * FN + self.beta * FP + self.smooth
        tversky = numerator / denominator
        return 1 - tversky


class MyFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class MyBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        out_dim,
    ):
        super(MyBlock, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.ReLU(),
                self.dropout,
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                self.dropout,
                nn.Linear(latent_dim, out_dim),
            )
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        
        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, edge_index, edge_attr):
        res = self.residual(x)
        
        out = self.conv(x, edge_index, edge_attr)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = out + res
        return out


class GNN(pl.LightningModule):
    def __init__(self, arch="compact"):
        """Graph neural network for subsurface feature segmentation.

        Two architectures are published, and a checkpoint only loads into the one
        it was trained with:

          arch="compact"  small model  (latent_dim 64,  skip_dim 32)  -> final_segmentation_model_gnn_meat.pt
          arch="large"    large model  (latent_dim 256, skip_dim 128) -> final_segmentation_model_gnn_sim.pt

        The large variant's sizes live under the `*_large` keys of the `gnn`
        section in system-params.json, so both are described by one config file
        (previously the two lived on separate git branches).
        """
        super().__init__()
        self.arch = arch
        self.save_hyperparameters()

        def gnn_param(name):
            """Read `name_large` for the large architecture, else `name`."""
            if arch == "large":
                large_name = f"{name}_large"
                if hasattr(SYSTEM_PARAMS.gnn, large_name):
                    return getattr(SYSTEM_PARAMS.gnn, large_name)
            return getattr(SYSTEM_PARAMS.gnn, name)

        self.num_regular_nodes = SYSTEM_PARAMS.vitactip.num_markers
        self.num_classes = SYSTEM_PARAMS.gnn.num_classes
        self.stages_str = [
            'train',
            'val',
            'test',
        ]
        self.init_accumulators()
        self.num_nodes = SYSTEM_PARAMS.vitactip.num_markers
        self.clip_len = SYSTEM_PARAMS.gnn.clip_len
        node_dim = 2+3+self.num_nodes+self.clip_len
        spatial_edge_dim = 5
        temporal_edge_dim = 5+2
        global_temporal_edge_dim = 2
        entity_type_embedding_dim = gnn_param("entity_type_embedding_dim")
        small_input_dim = gnn_param("small_input_dim")
        latent_dim = gnn_param("latent_dim")
        self.skip_dim = gnn_param("skip_dim")
        cat_out_dim = self.skip_dim * 4
        input_dim = latent_dim
        output_dim = SYSTEM_PARAMS.gnn.output_dim
        self.tversky_weight = SYSTEM_PARAMS.gnn.tversky_weight
        self.focal_weight = SYSTEM_PARAMS.gnn.focal_weight
        self.connectivity_weight = SYSTEM_PARAMS.gnn.connectivity_weight
        self.lr = SYSTEM_PARAMS.gnn.learning_rate
        self._previous_lr = None
        self.hidden_channels = latent_dim
        self.tversky_loss = MyTverskyLoss()
        self.focal_loss = MyFocalLoss()
        self.num_entity_types = SYSTEM_PARAMS.gnn.num_entity_types

        self.global_node = nn.Parameter(torch.randn(1, small_input_dim))
        self.edge_global_spatial = nn.Parameter(torch.randn(1, small_input_dim))
        self.entity_tag_embedding = nn.Embedding(
            num_embeddings=self.num_entity_types, 
            embedding_dim=entity_type_embedding_dim,
        )

        self.regular_node_mlp = nn.Sequential(
            nn.Linear(node_dim, small_input_dim), 
            nn.ReLU(), 
            nn.Linear(small_input_dim, small_input_dim)
        )
        self.spatial_edge_mlp = nn.Sequential(
            nn.Linear(spatial_edge_dim, small_input_dim),
            nn.ReLU(),
            nn.Linear(small_input_dim, small_input_dim),
        )
        self.temporal_edge_mlp = nn.Sequential(
            nn.Linear(temporal_edge_dim, small_input_dim),
            nn.ReLU(),
            nn.Linear(small_input_dim, small_input_dim),
        )
        self.global_temporal_edge_mlp = nn.Sequential(
            nn.Linear(global_temporal_edge_dim, small_input_dim),
            nn.ReLU(),
            nn.Linear(small_input_dim, small_input_dim),
        )

        self.block1 = MyBlock(
            in_dim=input_dim,
            latent_dim=latent_dim,
            out_dim=latent_dim,
        )
        self.block2 = MyBlock(
            in_dim=latent_dim,
            latent_dim=latent_dim,
            out_dim=latent_dim,
        )
        self.block3 = MyBlock(
            in_dim=latent_dim,
            latent_dim=latent_dim,
            out_dim=output_dim,
        )

        self.dropout = nn.Dropout(p=0.3)
        self.input_dropout = nn.Dropout(p=0.1)

        self.skip0 = self.skip_layer(input_dim)
        self.skip1 = self.skip_layer(latent_dim)
        self.skip2 = self.skip_layer(latent_dim)
        self.skip3 = self.skip_layer(output_dim)

        self.mlp_output_head = nn.Sequential(
            nn.Linear(cat_out_dim, cat_out_dim // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(cat_out_dim // 2, output_dim),
        )

        self.save_hyperparameters()
    
    def compute_connectivity_loss(
        self,
        batch,
        edge_index_regular_nodes,
        probs_unmasked,
        probs_masked,
    ):
        n = edge_index_regular_nodes.shape[1]
        k = batch.num_graphs
        assert n % k == 0
        x = n // k
        loss = torch.tensor(0.0, device=self.device)
        for i in range(batch.num_graphs):
            src, dst = edge_index_regular_nodes[:, x*i:x*(i+1)]
            diff = probs_unmasked[src] - probs_unmasked[dst]
            numerator = torch.sum(diff ** 2) / x
            denominator = (probs_masked.sum() / self.num_nodes) + 1e-8
            loss_single = numerator / denominator
            loss += loss_single
        loss /= k
        return loss
    
    def compute_connectivity_loss_single(self, pred, edge_index):
        src, dst = edge_index
        diff = pred[src] - pred[dst]
        numerator = torch.sum(diff ** 2)
        denominator = pred.sum() + 1e-8
        return numerator / denominator
    
    def skip_layer(self, in_dim):
        return nn.Linear(in_dim, self.skip_dim)
    
    def skip_layer_dropout(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.skip_dim),
            self.dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h0 = x
        h1 = self.block1(h0, edge_index, edge_attr)
        h2 = self.block2(h1, edge_index, edge_attr)
        h3 = self.block3(h2, edge_index, edge_attr)

        h0 = self.skip0(h0)
        h1 = self.skip1(h1)
        h2 = self.skip2(h2)
        h3 = self.skip3(h3)

        concat_features = torch.cat([h0, h1, h2, h3], dim=-1)

        out = self.mlp_output_head(concat_features)
        return out

    def shared_step(self, getitem_output, stage):
        batch, empty_visualisation_tensor, poses, metadata, frame_ix = getitem_output
        x, x_mask, edge_index, edge_index_regular_nodes, edge_attr = self.my_prepare_data(batch, batch.num_graphs)
        out = self(x, edge_index, edge_attr, batch.batch)
        out = out.squeeze(-1)
        out = out[x_mask]
        if stage == 'val' or stage == 'test':
            mask = batch.mask
            out_unmasked = out
            out_masked = out[mask]
            y_masked = batch.y[mask]
        else:
            out_unmasked = out
            out_masked = out
            y_masked = batch.y
        probs_unmasked = torch.sigmoid(out_unmasked)
        probs_masked = torch.sigmoid(out_masked)
        focal_loss = self.focal_loss(out_masked, y_masked.float())
        loss = (
            self.focal_weight*focal_loss
        )
        self.focal_loss_acc[stage] += focal_loss.detach()
        self.total_loss_acc[stage] += loss.detach()
        self.num_batches[stage] += 1
        # Only the logged IoU depends on this cut; the reported AUROC and AP are
        # threshold-free, so nothing in the paper moves if it changes.
        preds_masked = (probs_masked > DECISION_THRESHOLD).to(y_masked)

        self.update_iou_acc(
            preds_masked,
            y_masked,
            stage,
        )

        self.log(f"{stage}/focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
        self.log_per_batch_iou(batch, stage, preds_masked, y_masked)

        return loss

    def merge_tensors_unvectorised(self, x, y, n, k):
        b = x.shape[0] // n
        assert b == y.shape[0] // k, "Batch sizes must match"
        
        x_reshaped = x.reshape(b, n, -1)
        y_reshaped = y.reshape(b, k, -1)
        
        merged = []
        mask = []
        
        for i in range(b):
            merged.append(x_reshaped[i])
            mask.append(torch.ones(n, dtype=torch.bool))
            
            merged.append(y_reshaped[i])
            mask.append(torch.zeros(k, dtype=torch.bool))
        
        merged = torch.cat(merged, dim=0)
        mask = torch.cat(mask, dim=0)
        
        return merged, mask

    def my_prepare_data(self, batch, batch_size):
        pos = batch.pos
        mask = batch.mask
        y = batch.y
        regular_nodes = batch.regular_nodes

        edge_index_spatial = batch.edge_index_spatial
        edge_attr_spatial = batch.edge_attr_spatial

        edge_index_temporal = batch.edge_index_temporal
        edge_attr_temporal = batch.edge_attr_temporal

        edge_index_global_spatial = batch.edge_index_global_spatial
        edge_attr_global_spatial = batch.edge_attr_global_spatial

        edge_index_global_temporal = batch.edge_index_global_temporal
        edge_attr_global_temporal = batch.edge_attr_global_temporal

        regular_nodes = self.regular_node_mlp(regular_nodes)
        global_nodes = self.global_node.expand(batch_size*self.clip_len, -1)
        spatial_edges = self.spatial_edge_mlp(edge_attr_spatial)
        temporal_edges = self.temporal_edge_mlp(edge_attr_temporal)
        n = edge_index_global_spatial.shape[1]
        global_spatial_edges = self.edge_global_spatial.expand(n, -1)
        global_temporal_edges = self.global_temporal_edge_mlp(edge_attr_global_temporal)

        regular_nodes = self.cat_entity_tag(regular_nodes, 0)
        global_nodes = self.cat_entity_tag(global_nodes, 1)
        spatial_edges = self.cat_entity_tag(spatial_edges, 2)
        temporal_edges = self.cat_entity_tag(temporal_edges, 3)
        global_spatial_edges = self.cat_entity_tag(global_spatial_edges, 4)
        global_temporal_edges = self.cat_entity_tag(global_temporal_edges, 5)

        edge_index = torch.cat([
            edge_index_spatial,
            edge_index_temporal,
            edge_index_global_spatial,
            edge_index_global_temporal,
        ], dim=1)

        edge_index_regular_nodes = torch.cat([
            edge_index_spatial,
            edge_index_temporal,
        ], dim=1)

        edge_attr = torch.cat([
            spatial_edges,
            temporal_edges,
            global_spatial_edges,
            global_temporal_edges,
        ], dim=0)

        x, x_mask = self.merge_tensors_unvectorised(
            regular_nodes,
            global_nodes,
            self.num_regular_nodes*self.clip_len,
            self.clip_len,
        )

        edge_attr = self.input_dropout(edge_attr)
        x = self.input_dropout(x)

        return x, x_mask, edge_index, edge_index_regular_nodes, edge_attr
    
    def cat_entity_tag(self, features, entity_ix):
        ix_tensor = torch.tensor([entity_ix], dtype=torch.long, device=features.device)
        entity_tag = self.entity_tag_embedding(ix_tensor)
        entity_tag = entity_tag.expand(features.shape[0], -1)
        res = torch.cat([features, entity_tag], dim=1)
        return res
    
    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'bn' in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        optimizer = torch.optim.Adam([
            {'params': decay, 'weight_decay': 1e-4},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=64,
                gamma=0.5,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
        }
    
    def update_iou_acc(
        self, 
        preds: torch.Tensor, 
        y: torch.Tensor, 
        stage: str
    ):
        for class_ix in range(self.num_classes):
            pred_mask = (preds == class_ix)
            true_mask = (y == class_ix)

            self.area_pred_acc[stage][class_ix] += pred_mask.sum()
            self.area_true_acc[stage][class_ix] += true_mask.sum()
            self.area_inter_acc[stage][class_ix] += (pred_mask & true_mask).sum()
    
    def compute_ious_acc(self, stage: str):
        ious = {}

        for class_ix in range(self.num_classes):
            ap = self.area_pred_acc[stage][class_ix].float().item()
            at = self.area_true_acc[stage][class_ix].float().item()
            ai = self.area_inter_acc[stage][class_ix].float().item()

            if ap == 0 and at == 0:
                iou = 1.0
            elif ap == 0 or at == 0:
                iou = 0.0
            else:
                iou = ai / (ap + at - ai)

            ious[class_ix] = iou

        return ious
    
    def on_train_epoch_end(self):
        self.my_on_epoch_end('train')

    def on_validation_epoch_end(self):
        self.my_on_epoch_end('val')

    def on_test_epoch_end(self):
        self.my_on_epoch_end('test')
    
    def my_on_epoch_end(self, stage: str):
        log_on_prog_bar = stage == 'val'

        ious = self.compute_ious_acc(stage)
        self.log_dict(
            {f"{stage}_iou/{k}": v for k, v in ious.items()},
            prog_bar=log_on_prog_bar,
        )
        self.area_pred_acc[stage].zero_()
        self.area_true_acc[stage].zero_()
        self.area_inter_acc[stage].zero_()

        total_loss = self.total_loss_acc[stage] / self.num_batches[stage]
        focal_loss = self.focal_loss_acc[stage] / self.num_batches[stage]
        connectivity_loss = self.connectivity_loss_acc[stage] / self.num_batches[stage]
        hinge_loss = self.hinge_loss_acc[stage] / self.num_batches[stage]
        self.log(f"{stage}/loss", total_loss, prog_bar=False)
        self.log(f"{stage}/focal_loss", focal_loss, prog_bar=False)
        self.log(f"{stage}/connectivity_loss", connectivity_loss, prog_bar=False)
        self.log(f"{stage}/hinge_loss", hinge_loss, prog_bar=False)

        self.total_loss_acc[stage].zero_()
        self.focal_loss_acc[stage].zero_()
        self.connectivity_loss_acc[stage].zero_()
        self.hinge_loss_acc[stage].zero_()
        self.num_batches[stage] = 0

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_train_epoch_start(self):
        current_lr = self.optimizers().param_groups[0]["lr"]
        if self._previous_lr != current_lr:
            print(
                f"\nLearning rate changed from {self._previous_lr} to {current_lr:.2e}"
            )
            self._previous_lr = current_lr
        self.log("learning_rate", current_lr, prog_bar=False)

    def set_stats(self, stats):
        self.focal_loss.alpha = stats["alpha_pos"]
        print(f"focal_loss.alpha={self.focal_loss.alpha}")

    def set_loss_weights(self, tversky_alpha, tversky_beta, focal_alpha, focal_gamma):
        self.tversky_loss.alpha = tversky_alpha
        self.tversky_loss.beta = tversky_beta
        self.focal_loss.alpha = focal_alpha
        self.focal_loss.gamma = focal_gamma
    
    def log_per_batch_iou(self, batch, stage, preds, y):
        metrics = GNN.iou_score(preds, y)
        log_loss = stage == 'train'
        self.log_dict(
            {f"{stage}_iou/{k}": v for k, v in metrics.items()},
            prog_bar=log_loss,
            batch_size=batch.num_graphs,
        )

    @staticmethod
    def iou_score(preds, y):
        preds = preds.float()
        y = y.float()
        iou_0 = GNN.iou_score_single_class(preds, y, 0).item()
        iou_1 = GNN.iou_score_single_class(preds, y, 1).item()
        return {1: iou_1, 0: iou_0}
    
    @staticmethod
    def iou_score_single_class(
        preds: torch.Tensor, 
        y: torch.Tensor, 
        class_ix: int
    ) -> torch.Tensor:
        pred_mask = (preds == class_ix)
        true_mask = (y == class_ix)

        area_pred = pred_mask.sum().float().item()
        area_true = true_mask.sum().float().item()
        area_inter = (pred_mask & true_mask).sum().float().item()

        if area_pred == 0 and area_true == 0:
            return torch.tensor(float('nan'))

        if area_pred == 0 or area_true == 0:
            return torch.tensor(0.0)

        area_union = area_pred + area_true - area_inter
        return torch.tensor(area_inter / area_union)
    
    def get_iou_accumulator(self):
        return self.get_accumulator(shape=(self.num_classes,), dtype=torch.int64)
    
    def get_scalar_float64_accumulator(self):
        return self.get_accumulator(shape=(), dtype=torch.float64)
    
    def get_scalar_int32_accumulator(self):
        return self.get_accumulator(shape=(), dtype=torch.int32)
    
    def get_accumulator(self, shape, dtype):
        return {k: torch.zeros(shape, dtype=dtype, device='cuda:0') for k in self.stages_str}
    
    def init_accumulators(self):
        self.area_pred_acc = self.get_iou_accumulator()
        self.area_true_acc = self.get_iou_accumulator()
        self.area_inter_acc = self.get_iou_accumulator()
        self.total_loss_acc = self.get_scalar_float64_accumulator()
        self.focal_loss_acc = self.get_scalar_float64_accumulator()
        self.connectivity_loss_acc = self.get_scalar_float64_accumulator()
        self.hinge_loss_acc = self.get_scalar_float64_accumulator()
        self.num_batches = self.get_scalar_int32_accumulator()


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        train_subset_size,
        val_subset_size,
        batch_size,
        num_workers,
        seed=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        # `None` means "resolve centrally", so DIFFTACTILE_SEED reaches the
        # batch-order generator too. The resolved default is still 42, which is
        # what this argument was hardcoded to before.
        self.seed = resolve_seed(seed)
        seed = self.seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.current_train_indices = None
        self.current_val_indices = None

    def get_datasets(self):
        return [self.train_dataset, self.val_dataset, self.test_dataset]

    def setup(self, stage=None):
        self._select_new_subset()

    def _select_new_subset(self):
        len_train = len(self.train_dataset)
        train_subset_size = self.train_subset_size
        foo = min(len_train, train_subset_size)
        self.current_train_indices = NP_RNG.choice(
            len_train, foo, replace=False
        )
        len_val = len(self.val_dataset)
        val_subset_size = min(len_val, self.val_subset_size)
        self.current_val_indices = NP_RNG.choice(
            len_val, val_subset_size, replace=False
        )

    def train_dataloader(self):
        if self.current_train_indices is None:
            self._select_new_subset()
        sampler = SubsetRandomSampler(
            self.current_train_indices, generator=self.generator
        )
        # print(f"train dataset difficulty: {self.train_dataset.difficulty_fyi}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
            # Augmentation runs in the worker processes, and each forked worker
            # inherits a COPY of NP_RNG - so without this they would all draw the
            # identical rotation sequence. seed_worker gives each a distinct but
            # reproducible stream. See difftactile/main/seeding.py.
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def val_dataloader(self):
        if self.current_val_indices is None:
            self._select_new_subset()
        sampler = SubsetRandomSampler(
            self.current_val_indices, generator=self.generator
        )
        # print(f"val dataset difficulty: {self.train_dataset.difficulty_fyi}")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def test_dataloader(self):
        # print(f"test dataset difficulty: {self.train_dataset.difficulty_fyi}")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def on_train_epoch_start(self):
        self._select_new_subset()


def main():
    """Train the GNN on the real meat trials and test on the silicone phantom.

    This is the `sim-to-meat` / cross-domain training entrypoint. The bare
    `return` that used to sit here (disabling training on the pre-rename submission branch) has
    been replaced by the explicit scenario dispatcher in `run_scenario()`, so the
    behaviour is now selected by name rather than by editing the source.
    """
    # Before any dataset or model is built: the weights are drawn at
    # construction time, so seeding afterwards would not reach them.
    seed = seed_everything()
    print(f"seed: {seed} (override with DIFFTACTILE_SEED)")

    BATCH_SIZE = SYSTEM_PARAMS.gnn.batch_size
    NUM_EPOCHS = SYSTEM_PARAMS.gnn.num_epochs
    NUM_WORKERS = SYSTEM_PARAMS.gnn.num_workers
    TRAIN_EPOCH_SUBSET_SIZE = 10_000
    VAL_EPOCH_SUBSET_SIZE = 10_000
    tensor_board_root_dir = 'lightning_logs'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    tensor_board_experiment_dir = f"gnn"
    tensor_board_full_dir = f'{tensor_board_root_dir}/{tensor_board_experiment_dir}'
    print(f'tensorboard directory: {tensor_board_full_dir}')
    # logger = TensorBoardLogger(
    #     save_dir=tensor_board_root_dir,
    #     name=tensor_board_experiment_dir,
    #     version=f"run_{timestamp}",
    # )
    logger = CSVLogger(
        save_dir="logs",
        name="my_experiment",
        version=f"run_{timestamp}",
    )
    # C-to-B trains on meat, so the random-rotation augmentation stays ON here.
    # getitem_meat() additionally gates on mode == "train", so the val and test
    # splits created below are unaugmented even though they inherit this flag.
    meat_dataset = MyDataset(
        scheme="meat",
        sim_exp='apple',
        data_dir='banana',
        apply_augmentations=True,
        name='meat',
    )
    train_dataset, val_dataset, test_dataset = meat_dataset.create_splits()
    silicone_dataset = MyDataset(
        scheme="single_dataset",
        sim_exp="exp",
        data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
        apply_augmentations=False,
        name='silicone',
    )
    stats = compute_stats(train_dataset, BATCH_SIZE)
    alpha_neg = stats["alpha_neg"]
    alpha_pos = stats["alpha_pos"]
    print(f"pos:neg = {alpha_neg:.2f}:{alpha_pos:.2f}")
    train_dataset.set_stats(stats)
    val_dataset.set_stats(stats)
    test_dataset.set_stats(stats)
    silicone_dataset.set_stats(stats)
    data_module = MyDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_subset_size=TRAIN_EPOCH_SUBSET_SIZE,
        val_subset_size=VAL_EPOCH_SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    test_data = {
        "dataset": test_dataset,
        "num_workers": NUM_WORKERS,
        "dataset_stats": stats,
        # Marks this loader as coming from the meat scheme, whose `dataset_stats`
        # is a single stats dict rather than the curriculum's {difficulty: stats}
        # mapping. Read back via `has_flat_stats()`.
        MEAT_LOADER_FLAG: True,
    }
    # Write to *_retrained paths, NOT over the published artifacts.
    # evaluate_and_plot_roc() (the sim-to-silicone scenario) reads the bundle's
    # checkpoint and test-loader pickle; overwriting them here would silently
    # change the reported AUC on the next evaluation run, with no way back short
    # of re-restoring the Zenodo bundle.
    test_loader_path = _retrained_path(SYSTEM_PARAMS.files.test_loader_gnn_meat)
    os.makedirs(os.path.dirname(test_loader_path), exist_ok=True)
    with open(test_loader_path, "wb") as f:
        pickle.dump(test_data, f)
    print(f"test loader written to: {test_loader_path}")
    model = GNN()
    model.set_stats(stats)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_iou/1",
        mode="max",
        save_top_k=1,
        filename="best-model-meat",
    )
    early_stopping = EarlyStopping(
        monitor="val_iou/1",
        mode="max",
        patience=NUM_EPOCHS * 2,
        min_delta=1e-4,
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto',
        enable_checkpointing=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            checkpoint_cb,
        ],
        reload_dataloaders_every_n_epochs=1,
    )
    silicone_loader = DataLoader(
        silicone_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
    )
    start = time.perf_counter()
    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_cb.best_model_path
    if not best_model_path:
        raise RuntimeError("No best checkpoint was saved; cannot test best model.")
    best_model = GNN.load_from_checkpoint(best_model_path)
    print("\nTesting on silicone dataset:")
    trainer.test(best_model, dataloaders=silicone_loader)
    # Threshold-free companions to the IoU above, so C-to-B reports the same
    # metrics as the other two configurations. Returned at the end of this
    # function so the seed sweep can collect them without parsing stdout.
    metrics = score_ranking_metrics(best_model, silicone_loader, "C-to-B")
    end = time.perf_counter()
    duration = end - start
    print(
        f"\nTraining and testing completed in {duration:.2f} seconds ({duration / 60:.2f} minutes)"
    )
    model_path = _retrained_path(SYSTEM_PARAMS.files.final_segmentation_model_gnn_meat)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model.state_dict(), model_path)
    print(f"checkpoint written to: {model_path}")
    print(
        "  (the published checkpoint is left untouched; set "
        "DIFFTACTILE_OVERWRITE_PUBLISHED=1 to overwrite it instead)"
    )
    return metrics


def compute_stats(dataset, batch_size):
    len_train = len(dataset)
    train_subset_size = batch_size
    ixs = NP_RNG.choice(len_train, train_subset_size, replace=False)
    res = compute_alpha(dataset, ixs)
    keys = [
        "pos",
        "regular_nodes",

        "edge_attr_spatial",
        "edge_attr_temporal",

        "edge_attr_global_spatial",
        "edge_attr_global_temporal",
    ]
    for key in keys:
        res |= compute_mean_std(dataset, ixs, key)
    return res


def compute_alpha(dataset, ixs):
    num_pos = 0
    num_neg = 0
    for ix in tqdm(ixs, desc="Computing stats for alpha"):
        pyg, veins, poses, metadata, frame_ix = dataset[ix]
        y = pyg.y.cpu().numpy()
        pos = y.sum()
        neg = y.shape[0] - pos
        num_pos += pos
        num_neg += neg
    alpha_pos = num_neg / (num_neg + num_pos)
    alpha_neg = num_pos / (num_neg + num_pos)
    return {
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg,
    }


def compute_mean_std(dataset, ixs, key):
    vals = []
    for ix in tqdm(ixs, desc=f"Computing stats for {key}"):
        pyg, veins, poses, metadata, frame_ix = dataset[ix]
        val = pyg[key].cpu().numpy()
        vals.append(val)
    vals = np.array(vals)
    if vals.size == 0:
        vals_mean = None
        vals_std = None
    else:
        vals = vals.reshape((-1, vals.shape[-1]))
        vals_mean = np.mean(vals, axis=0)
        vals_std = np.std(vals, axis=0)
    return {
        f"{key}_mean": vals_mean,
        f"{key}_std": vals_std,
    }

def evaluate_on_sim():
    """A-to-A: score the published sim checkpoint on held-out SIMULATED data.

    The in-domain result - train on dataset A, test on a disjoint part of
    dataset A. It is the ceiling the cross-domain configurations are measured
    against: the gap between this and A-to-B is the sim-to-real transfer cost.

    The test split comes straight out of the published test-loader pickle, which
    stores the very `MyDataset` (mode='test') that `train_on_sim()` held back,
    rather than re-deriving it. Re-deriving would risk a different split from the
    one the checkpoint was actually held out from - which would quietly turn this
    into a partly-seen-data number.

    Uses the simulated test split, NOT the validation split: validation drives
    early stopping and checkpoint selection, so a number read off it is
    optimistic by construction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(arch="large")
    ckpt = repo_path(SYSTEM_PARAMS.files.final_segmentation_model_gnn_sim)
    print(f"loading simulation-trained checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    model = model.to(device)

    stats_path = repo_path(SYSTEM_PARAMS.files.test_loader_gnn_sim)
    with open(stats_path, "rb") as f:
        test_data = pickle.load(f)
    test_dataset = test_data["dataset"]
    print(f"held-out simulated test split: {len(test_dataset)} clips ({stats_path})")

    all_stats = test_data["dataset_stats"]
    if has_flat_stats(test_data):
        stats = all_stats
    else:
        # Curriculum-keyed stats; 1.0 is the difficulty the checkpoint trained at.
        target_difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
        test_dataset.set_difficulty_level(target_difficulty)
        stats = all_stats[target_difficulty]
    test_dataset.set_stats(stats)
    test_dataset.eval()

    batch_size = getattr(SYSTEM_PARAMS.gnn, "batch_size_large", SYSTEM_PARAMS.gnn.batch_size)
    num_workers = getattr(SYSTEM_PARAMS.gnn, "num_workers_large", SYSTEM_PARAMS.gnn.num_workers)
    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    return score_ranking_metrics(model, data_loader, "A-to-A", device=device)


def evaluate_and_plot_roc():
    """Evaluate the SIMULATION-trained checkpoint on silicone (B); ROC and PR curves.

    This is the evaluate-only half of A-to-B, so it must load dataset A's
    checkpoint. It previously loaded `final_segmentation_model_gnn_meat` (the
    small, MEAT-trained model) with a default-architecture `GNN()`, which made
    this path compute C-to-B and silently duplicate the C-to-B configuration.
    The reported AUC was therefore the meat-trained model on silicone, not the
    sim-to-real result A-to-B is meant to report.

    The checkpoint is the LARGE architecture, and its normalisation statistics
    come from the matching sim test-loader pickle, exactly as `silicone_to_meat()`
    (the A-to-C evaluate path) does.

    Reports AUROC *and* average precision, with a figure for each; the scoring
    itself lives in `score_ranking_metrics()`, shared with the other two
    configurations. Returns those metrics as a dict (callers currently ignore it).
    """
    if True:
        model = GNN(arch="large")
        ckpt = repo_path(SYSTEM_PARAMS.files.final_segmentation_model_gnn_sim)
        print(f"loading simulation-trained checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with open(repo_path(SYSTEM_PARAMS.files.test_loader_gnn_sim), 'rb') as f:
            test_data = pickle.load(f)

        full_dataset = MyDataset(
            scheme="single_dataset",
            sim_exp="exp",
            data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
            apply_augmentations=False,
            name='silicone',
        )
        train_dataset, _, _ = full_dataset.create_splits(
            train_size=1.0,
            val_size=0.0,
            test_size=0.0
        )
        if has_flat_stats(test_data):
            stats = test_data['dataset_stats']
        else:
            # The simulated pipeline carries a difficulty curriculum, so its
            # stats are keyed by difficulty. `train_on_sim()` trains at 1.0 and
            # stores {1.0: stats}; fall back to the sole entry if a loader was
            # written at a different setting.
            all_stats = test_data['dataset_stats']
            target_difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
            train_dataset.set_difficulty_level(target_difficulty)
            stats = all_stats[target_difficulty]
        train_dataset.set_stats(stats)
        train_dataset.eval()
        data_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=16,
            pin_memory=False,
            persistent_workers=False,
        )

    # Scoring and figure-writing are shared with the other two configurations, so
    # all three report the same threshold-free metrics from one implementation.
    # Figures are named for the configuration, not the checkpoint: the legacy
    # `script_gnn` entrypoint in cnn/gnn.py already writes roc_curve_sim.pdf.
    return score_ranking_metrics(model, data_loader, "A-to-B", device=device)


# =============================================================================
# Scenario dispatcher
#
# The paper trains three distinct models over three datasets:
#
#   (A) simulated dataset collected in the differentiable-tactile simulator
#   (B) real silicone phantom, shallow veins             ("easy")
#   (C) real meat phantom, veins at varying depths       ("difficult")
#
# giving three (train -> test) configurations, each with a train and an
# evaluate mode so that no branch switching is needed to reproduce any of them:
#
#   A-to-B   train on sim,  test on silicone   (large architecture)
#   C-to-B   train on meat, test on silicone   (small "compact" architecture)
#   A-to-C   train on sim,  test on meat       (large architecture)
#
# These used to live on separate git branches, selected by editing source
# (commenting call lists, toggling `if False:` blocks, and a bare `return` at
# the top of main()). They are unified here and chosen by name instead.
#
# NOTE on the legacy alias `silicone-to-meat`: despite its name it loads
# `final_segmentation_model_gnn_sim.pt`, which is the SIMULATION-trained
# checkpoint (the large architecture also used for A-to-B). The configuration it
# actually runs is therefore sim -> meat, i.e. A-to-C, which is how the paper
# describes it. The old name is kept working, but A-to-C is the accurate one.
#
# Usage:
#   python -m difftactile.scripts.script_segmentation_gnn <config> [--train|--eval]
# =============================================================================

# Canonical paper configurations. Each maps to (train_fn, eval_fn).
PAPER_CONFIGS = ("A-to-B", "C-to-B", "A-to-C")

# Backwards-compatible aliases for the names used before the paper notation was
# adopted. `silicone-to-meat` is a misnomer - see the NOTE above.
SCENARIO_ALIASES = {
    "sim-to-silicone": "A-to-B",
    "sim-to-meat": "C-to-B",
    "silicone-to-meat": "A-to-C",
    # Descriptive spellings, accepted for convenience.
    "meat-to-silicone": "C-to-B",
    "sim-to-meat-eval": "A-to-C",
}

# Kept so that `from difftactile.cnn.segmentation_gnn import *` still exposes a list of
# every accepted name (the old constant only held the three legacy names).
SCENARIOS = PAPER_CONFIGS + tuple(SCENARIO_ALIASES)


def train_on_sim(test_on="silicone"):
    """Train the GNN on the SIMULATED dataset (A), then test it on real data.

    This is the training half of both A-to-B (test_on="silicone") and A-to-C
    (test_on="meat"). Only the held-out test set differs; the model, the
    training data and the normalisation statistics are identical, which is why
    one function covers both.

    Uses the LARGE ("large") architecture, matching
    final_segmentation_model_gnn_sim.pt - the published simulation-trained
    checkpoint that the evaluate-only paths load.

    Previously this lived in `cnn/gnn.py::main()`, disabled by a bare `return`
    on every branch, so the simulation-trained models could not be reproduced
    without editing source.
    """
    # Before any dataset or model is built: the weights are drawn at
    # construction time, so seeding afterwards would not reach them.
    seed = seed_everything()
    print(f"seed: {seed} (override with DIFFTACTILE_SEED)")

    BATCH_SIZE = getattr(SYSTEM_PARAMS.gnn, "batch_size_large", SYSTEM_PARAMS.gnn.batch_size)
    NUM_EPOCHS = getattr(SYSTEM_PARAMS.gnn, "num_epochs_large", SYSTEM_PARAMS.gnn.num_epochs)
    NUM_WORKERS = getattr(SYSTEM_PARAMS.gnn, "num_workers_large", SYSTEM_PARAMS.gnn.num_workers)
    NUM_TRAIN_BATCHES = getattr(SYSTEM_PARAMS.gnn, "num_train_batches_large", -1)
    NUM_VAL_BATCHES = getattr(SYSTEM_PARAMS.gnn, "num_val_batches_large", -1)
    # MyDataModule takes an epoch subset *size* and clamps it with
    # min(len(dataset), size), so a very large number means "use everything".
    # A batch count of -1 is the config's way of saying exactly that.
    UNLIMITED = sys.maxsize
    TRAIN_EPOCH_SUBSET_SIZE = BATCH_SIZE * NUM_TRAIN_BATCHES if NUM_TRAIN_BATCHES > 0 else UNLIMITED
    VAL_EPOCH_SUBSET_SIZE = BATCH_SIZE * NUM_VAL_BATCHES if NUM_VAL_BATCHES > 0 else UNLIMITED

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    logger = CSVLogger(save_dir="logs", name="my_experiment", version=f"run_{timestamp}")

    # Dataset A - the simulated trajectories.
    sim_dataset = MyDataset(
        scheme="single_dataset",
        sim_exp="sim",
        data_dir=SYSTEM_PARAMS.files.sim_data,
        apply_augmentations=True,
        name="sim",
    )
    train_dataset, val_dataset, test_dataset = sim_dataset.create_splits(
        train_size=0.7, val_size=0.15, test_size=0.15
    )

    # The dataset this configuration is evaluated against. For A-to-A that is
    # the held-out simulated test split itself - there is no second domain - so
    # the "real" evaluation below simply becomes the in-domain one, and the
    # cross-domain section is skipped.
    if test_on == "sim":
        real_dataset = None
        real_eval_dataset = None
    elif test_on == "silicone":
        real_dataset = MyDataset(
            scheme="single_dataset",
            sim_exp="exp",
            data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
            apply_augmentations=False,
            name="silicone",
        )
        real_eval_dataset = real_dataset
    elif test_on == "meat":
        real_dataset = MyDataset(
            scheme="meat",
            sim_exp="apple",
            data_dir="banana",
            apply_augmentations=False,
            name="meat",
        )
        # Pure transfer: nothing is held back for fitting.
        _, _, real_eval_dataset = real_dataset.create_splits(all_to_test=True)
    else:
        raise ValueError(
            f"test_on must be 'sim', 'silicone' or 'meat', got {test_on!r}"
        )

    # The simulated pipeline carries a difficulty curriculum; 1.0 is the setting
    # the published checkpoint was trained at.
    target_difficulty = 1.0
    train_dataset.set_difficulty_level(target_difficulty)
    stats = compute_stats(train_dataset, BATCH_SIZE)
    print(f"pos:neg = {stats['alpha_neg']:.2f}:{stats['alpha_pos']:.2f}")
    for ds in (val_dataset, test_dataset):
        ds.set_difficulty_level(target_difficulty)
    for ds in (train_dataset, val_dataset, test_dataset, real_eval_dataset):
        # real_eval_dataset is None for A-to-A, which evaluates on the simulated
        # test split already covered above.
        if ds is not None:
            ds.set_stats(stats)
    # set_difficulty_level only applies to the simulated "single_dataset"
    # scheme; the meat scheme has no curriculum.
    if test_on == "silicone":
        real_eval_dataset.set_difficulty_level(target_difficulty)

    data_module = MyDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_subset_size=TRAIN_EPOCH_SUBSET_SIZE,
        val_subset_size=VAL_EPOCH_SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Persist the stats alongside the test split so the evaluate-only paths can
    # normalise exactly as training did. Written to a *_retrained path so the
    # published artifacts survive (see _retrained_path).
    test_loader_path = _retrained_path(SYSTEM_PARAMS.files.test_loader_gnn_sim)
    os.makedirs(os.path.dirname(test_loader_path), exist_ok=True)
    with open(test_loader_path, "wb") as f:
        pickle.dump(
            {
                "dataset": test_dataset,
                "num_workers": NUM_WORKERS,
                "dataset_stats": {target_difficulty: stats},
            },
            f,
        )
    print(f"test loader written to: {test_loader_path}")

    model = GNN(arch="large")
    model.set_stats(stats)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_iou/1", mode="max", save_top_k=1, filename="best-model-sim"
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        enable_checkpointing=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
    )
    real_loader = None
    if real_eval_dataset is not None:
        real_loader = DataLoader(
            real_eval_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=False,
            shuffle=False,
        )

    start = time.perf_counter()
    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_cb.best_model_path
    if not best_model_path:
        raise RuntimeError("No best checkpoint was saved; cannot test best model.")
    best_model = GNN.load_from_checkpoint(best_model_path)
    config_name = config_name_for(test_on)

    print("\nTesting on simulation data:")
    trainer.test(best_model, datamodule=data_module)
    if real_loader is not None:
        print(f"\nTesting on the real {test_on} dataset:")
        trainer.test(best_model, dataloaders=real_loader)

    # The held-out 15% of dataset A, same distribution as the training data.
    #
    # For A-to-A this IS the result. For A-to-B and A-to-C it is the in-domain
    # reference reported beside the cross-domain number, and the gap between the
    # two is the sim-to-real transfer cost.
    #
    # The TEST split, not the val split: val drives early stopping and
    # ModelCheckpoint's `save_top_k`, so a number read off it is optimistic by
    # construction. The test split is untouched by both.
    if real_loader is None:
        print("\nHeld-out simulation (A-to-A):")
        metrics = score_ranking_metrics(
            best_model, data_module.test_dataloader(), config_name,
        )
    else:
        print("\nIn-domain reference (held-out simulation, same distribution as training):")
        in_domain = score_ranking_metrics(
            best_model, data_module.test_dataloader(), f"{config_name}-sim",
        )
        print(f"\nCross-domain result on the real {test_on} dataset:")
        metrics = score_ranking_metrics(best_model, real_loader, config_name)
        # Carried under an `in_domain_` prefix so one flat dict holds both, which
        # is what lets the seed sweep summarise them side by side.
        for key, value in in_domain.items():
            metrics[f"in_domain_{key}"] = value
    duration = time.perf_counter() - start
    print(f"\nTraining and testing completed in {duration:.2f} s ({duration / 60:.2f} min)")

    model_path = _retrained_path(SYSTEM_PARAMS.files.final_segmentation_model_gnn_sim)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model.state_dict(), model_path)
    print(f"checkpoint written to: {model_path}")
    print(
        "  (the published checkpoint is left untouched; set "
        "DIFFTACTILE_OVERWRITE_PUBLISHED=1 to overwrite it instead)"
    )
    return metrics


def config_name_for(test_on):
    """Configuration name for a `train_on_sim(test_on=...)` run.

    A-to-A, A-to-B and A-to-C all train on simulation; the target dataset is what
    tells them apart.
    """
    return {"sim": "A-to-A", "silicone": "A-to-B", "meat": "A-to-C"}[test_on]


def marker_iou(probs, labels, threshold=DECISION_THRESHOLD):
    """Foreground and background IoU over marker nodes, as the paper's table reports.

    WHAT THE TWO NUMBERS MEAN. Each is an ordinary per-class intersection-over-
    union, computed over every scored marker node pooled together (not averaged
    per frame):

        foreground IoU = |predicted vessel AND truly vessel|
                         --------------------------------------
                         |predicted vessel OR  truly vessel|

        background IoU = the same with "vessel absent" as the class.

    So "foreground" is NOT "ground truth says vessel" nor "prediction says
    vessel" - it is the class label, and the metric is the agreement between the
    two over that class. A prediction that finds every vessel marker but also
    flags many empty ones scores a low foreground IoU despite perfect recall,
    because the union grows.

    Background IoU is always the flattering one here: the negative class is ~89%
    of nodes on silicone and ~93% on meat, so even a poor model overlaps heavily
    with it. Report the pair, and read the foreground number as the real one.

    UNLIKE AUROC AND AP, THIS DEPENDS ON A DECISION THRESHOLD, so it inherits
    every caveat about that choice (see cnn/curve_plots.py). It is reported
    because the manuscript's table uses it and it is the intuitive quantity, not
    because it is the most trustworthy one.

    These are frame-by-frame marker predictions - NOT the reprojected bird's-eye
    phantom map, whose separate IoU comes from `vessel_map.sh`.
    """
    preds = (probs > threshold).astype(int)
    labels = labels.astype(int)

    ious = {}
    for class_ix, name in ((1, "foreground"), (0, "background")):
        pred_mask = preds == class_ix
        true_mask = labels == class_ix
        intersection = int((pred_mask & true_mask).sum())
        union = int((pred_mask | true_mask).sum())
        # A class absent from both prediction and truth is perfectly agreed on;
        # present in one but not the other is total disagreement. Matches
        # `compute_ious_acc()`, which is what the Lightning test table reports.
        if union == 0:
            ious[name] = 1.0
        else:
            ious[name] = intersection / union
    ious["macro"] = (ious["foreground"] + ious["background"]) / 2
    return ious


def score_ranking_metrics(model, loader, config, device=None):
    """AUROC and AP for a loaded model over a dataloader, with both figures.

    Factored out so every configuration reports the same threshold-free numbers.
    `evaluate_and_plot_roc()` (A-to-B) computed them inline; the other two ran
    only `trainer.test()` and so reported IoU at a fixed threshold and nothing
    else - which is exactly the metric that does not survive a domain shift,
    since it confounds ranking quality with output calibration.

    Scores the same masked central-frame nodes the val/test stages do (see
    `shared_step`), so these numbers describe the reported predictions.

    Writes `roc_curve_<config>.pdf` and `pr_curve_<config>.pdf` under
    difftactile/output/, and returns the metrics as a dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch, labels_images, poses, metadata, frame_ix in loader:
            batch = batch.to(device)
            x, x_mask, edge_index, _, edge_attr = model.my_prepare_data(
                batch, batch.num_graphs
            )
            out = model(x, edge_index, edge_attr, batch.batch)
            out = out.squeeze(-1)[x_mask]
            # Central-frame mask: the same one the reported metrics use.
            mask = batch.mask
            out = out[mask]
            all_probs.append(torch.sigmoid(out).cpu())
            all_labels.append(batch.y[mask].cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)

    roc_path = repo_path(f"difftactile/output/roc_curve_{config}.pdf")
    plot_roc(plt, fpr, tpr, all_probs, all_labels, auc, roc_path,
             thresholds_roc=roc_thresholds)
    pr_path = repo_path(f"difftactile/output/pr_curve_{config}.pdf")
    plot_pr(plt, precision, recall, all_probs, all_labels, ap, pr_path,
            thresholds_pr=pr_thresholds)

    # IoU at the decision threshold, which is what the manuscript's table
    # reports. Computed here so all three configurations produce it from one
    # implementation - previously A-to-B --eval reported no IoU at all, and the
    # other two only emitted it inside a Lightning table as `test_iou/0` and
    # `test_iou/1`, which does not say which class is which.
    ious = marker_iou(all_probs, all_labels)

    # Persist the raw (probability, label) pairs next to the checkpoint when a
    # sweep is directing artifacts there. Every curve and every scalar above is
    # derivable from these two arrays, so keeping them is what lets the sweep
    # draw mean ROC/PR curves across seeds without re-running inference. A few
    # hundred KB per seed.
    artifact_dir = os.environ.get(ARTIFACT_DIR_ENV_VAR)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(artifact_dir, f"scores_{config}.npz"),
            probs=all_probs, labels=all_labels,
        )

    baseline = float(all_labels.mean())
    lift = ap / baseline if baseline > 0 else float("nan")
    n_pos = int(all_labels.sum())
    print(f"\n=== Metrics for {config} ===")
    print(f"  nodes scored: {len(all_labels)} ({n_pos} vessel, {baseline:.1%})")
    print("\n  Threshold-free (what the paper leads with):")
    print(f"    AUROC = {auc:.4f}")
    print(f"    AP    = {ap:.4f}   (chance = {baseline:.4f}, i.e. {lift:.2f}x lift)")
    print(f"\n  IoU at threshold {DECISION_THRESHOLD} "
          f"(frame-by-frame marker predictions):")
    print(f"    foreground IoU (vessel present) = {ious['foreground']:.4f}")
    print(f"    background IoU (vessel absent)  = {ious['background']:.4f}")
    print(f"    macro IoU (mean of the two)     = {ious['macro']:.4f}")
    print(f"\n  ROC curve: {roc_path}")
    print(f"  PR curve:  {pr_path}")
    return {
        "auroc": float(auc), "ap": float(ap), "chance": baseline,
        "lift": float(lift),
        "iou_foreground": ious["foreground"],
        "iou_background": ious["background"],
        "iou_macro": ious["macro"],
        "iou_threshold": DECISION_THRESHOLD,
        "num_nodes": int(len(all_labels)),
        "num_positive": n_pos,
    }


def silicone_to_meat():
    """Test the simulation-trained (large) checkpoint on the real meat trials.

    No training happens: the checkpoint is loaded as-is and evaluated, which is
    what makes this the cross-domain generalisation result.
    """
    # The published checkpoint is the LARGE architecture, so both the model and the
    # batch size come from the `*_large` config keys.
    BATCH_SIZE = getattr(SYSTEM_PARAMS.gnn, "batch_size_large", SYSTEM_PARAMS.gnn.batch_size)
    NUM_WORKERS = getattr(SYSTEM_PARAMS.gnn, "num_workers_large", SYSTEM_PARAMS.gnn.num_workers)

    # The meat trials are indexed by dataset.py's "meat" scheme; the sim_exp and
    # data_dir arguments are unused for that scheme. apply_augmentations is NOT
    # unused: this is evaluation, so the random rotation must stay off.
    meat_dataset = MyDataset(
        scheme="meat",
        sim_exp="apple",
        data_dir="banana",
        apply_augmentations=False,
        name="meat",
    )
    # Pure transfer: every trial is evaluated, none held back for fitting.
    _, _, test_dataset = meat_dataset.create_splits(all_to_test=True)

    model = GNN(arch="large")
    ckpt = repo_path(SYSTEM_PARAMS.files.final_segmentation_model_gnn_sim)
    print(f"loading simulation-trained checkpoint: {ckpt}")
    model.load_state_dict(
        torch.load(ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
    )

    # Normalisation statistics must match those the checkpoint was trained with,
    # so they are taken from the saved test-loader pickle rather than recomputed.
    with open(repo_path(SYSTEM_PARAMS.files.test_loader_gnn_sim), "rb") as f:
        test_data = pickle.load(f)
    if has_flat_stats(test_data):
        stats = test_data["dataset_stats"]
    else:
        # Curriculum-keyed stats. train_on_sim() trains at difficulty 1.0 and
        # stores {1.0: stats}; older pickles used 0.0. Prefer 1.0 and fall back
        # to whatever single entry is present, matching evaluate_and_plot_roc()
        # and evaluate_on_sim().
        all_stats = test_data["dataset_stats"]
        target_difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
        stats = all_stats[target_difficulty]
    model.set_stats(stats)
    # NOTE — deliberate behaviour change from the old `sim-to-meat-test` branch,
    # and NOT a cosmetic one.
    # There, this call sat inside a dead `if False:` block, so the dataset's
    # `warmup` flag stayed True and MyDataset.normalise() was a no-op: the
    # checkpoint was evaluated on UNNORMALISED inputs despite having been
    # trained on normalised ones. Applying the simulation statistics the checkpoint
    # expects raises the vein IoU from 0.034 to 0.198 (background 0.888 ->
    # 0.809) — measured, see REPRODUCTION_TEST.md. The old path understated the
    # cross-domain result roughly six-fold.
    test_dataset.set_stats(stats)

    meat_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
    )
    trainer = pl.Trainer(accelerator="auto", logger=False, enable_checkpointing=False)
    print("\nTesting simulation-trained model on the meat dataset:")
    start = time.perf_counter()
    trainer.test(model, dataloaders=meat_loader)
    # trainer.test() reports IoU at a fixed threshold, which confounds ranking
    # quality with output calibration - and calibration is the first thing to
    # shift across a domain gap, which is precisely what this configuration
    # measures. The threshold-free metrics are the ones to read here.
    metrics = score_ranking_metrics(model, meat_loader, "A-to-C")
    print(f"\nDone in {time.perf_counter() - start:.2f} s")
    return metrics


# For each paper configuration: what it does in --train and in --eval mode.
#
#   train: reproduces the model from scratch and writes a *_retrained checkpoint
#   eval:  loads the published checkpoint and reports the paper's numbers
#
# C-to-B has no separate eval entry because its published training run ends by
# testing on silicone; `main()` therefore serves as both.
CONFIG_ACTIONS = {
    "A-to-A": {
        "description": "train on simulation (A), test on held-out simulation (A)",
        "train": lambda: train_on_sim(test_on="sim"),
        "eval": lambda: evaluate_on_sim(),
    },
    "A-to-B": {
        "description": "train on simulation (A), test on silicone (B)",
        "train": lambda: train_on_sim(test_on="silicone"),
        "eval": lambda: evaluate_and_plot_roc(),
    },
    "C-to-B": {
        "description": "train on meat (C), test on silicone (B)",
        "train": lambda: main(),
        "eval": lambda: main(),
    },
    "A-to-C": {
        "description": "train on simulation (A), test on meat (C)",
        "train": lambda: train_on_sim(test_on="meat"),
        "eval": lambda: silicone_to_meat(),
    },
}

# Which mode each configuration runs when none is given. Evaluation is the
# cheaper, reproduce-the-published-number path, so it is the default wherever a
# published checkpoint exists.
DEFAULT_MODES = {"A-to-A": "eval", "A-to-B": "eval", "C-to-B": "train", "A-to-C": "eval"}

_USAGE = (
    "Usage: python -m difftactile.scripts.script_segmentation_gnn <config> [--train|--eval]\n"
    "\n"
    "Configurations (paper notation; A=simulation, B=silicone, C=meat):\n"
    "  A-to-A   train on simulation, test on held-out simulation   [default: --eval]\n"
    "  A-to-B   train on simulation, test on silicone              [default: --eval]\n"
    "  C-to-B   train on meat,       test on silicone              [default: --train]\n"
    "  A-to-C   train on simulation, test on meat                  [default: --eval]\n"
    "\n"
    "Modes:\n"
    "  --train  reproduce the model from scratch (writes a *_retrained checkpoint)\n"
    "  --eval   load the published checkpoint and report its numbers\n"
)


def run_scenario(scenario=None, mode=None):
    """Entrypoint used by scripts/script_segmentation_gnn.py.

    Selects one of the three paper configurations (A-to-B, C-to-B, A-to-C) and
    runs it in either training or evaluation mode, so all three models can be
    trained and tested from this one branch.

    `scenario` falls back to the first non-flag CLI argument, then to the
    DIFFTACTILE_SCENARIO environment variable, then to A-to-B. `mode` falls back
    to a --train/--eval flag, then to DIFFTACTILE_MODE, then to the
    configuration's default in DEFAULT_MODES.
    """
    argv = sys.argv[1:]
    flags = [a for a in argv if a.startswith("--")]
    positional = [a for a in argv if not a.startswith("--")]

    if scenario is None:
        scenario = positional[0] if positional else None
    if scenario is None:
        scenario = os.environ.get("DIFFTACTILE_SCENARIO", "A-to-B")

    if mode is None:
        if "--train" in flags:
            mode = "train"
        elif "--eval" in flags:
            mode = "eval"
    unknown = [f for f in flags if f not in ("--train", "--eval", "--help", "-h")]
    if "--help" in flags or "-h" in flags:
        print(_USAGE)
        return None
    if unknown:
        raise SystemExit(f"Unknown option(s): {', '.join(unknown)}\n\n{_USAGE}")

    # Accept the pre-paper names so existing commands and docs keep working.
    canonical = SCENARIO_ALIASES.get(scenario, scenario)
    if canonical not in CONFIG_ACTIONS:
        raise SystemExit(
            f"Unknown configuration {scenario!r}.\n\n{_USAGE}"
        )
    if canonical != scenario:
        print(f"note: {scenario!r} is a legacy alias for {canonical!r}")

    if mode is None:
        mode = os.environ.get("DIFFTACTILE_MODE") or DEFAULT_MODES[canonical]
    if mode not in ("train", "eval"):
        raise SystemExit(f"Unknown mode {mode!r}; expected 'train' or 'eval'.\n\n{_USAGE}")

    action = CONFIG_ACTIONS[canonical]
    print(f"=== {canonical} ({action['description']}) | mode: {mode} ===")
    # Tag any artifacts this run writes with the configuration name, so the
    # three configurations do not overwrite each other's retrained checkpoints.
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = canonical
    try:
        return action[mode]()
    finally:
        _ACTIVE_CONFIG = None
