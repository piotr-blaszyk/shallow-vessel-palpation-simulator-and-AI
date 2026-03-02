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
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from difftactile.cnn.common import *
from difftactile.cnn.dataset import *


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
    def __init__(self):
        super().__init__()
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
        entity_type_embedding_dim = SYSTEM_PARAMS.gnn.entity_type_embedding_dim
        small_input_dim = SYSTEM_PARAMS.gnn.small_input_dim
        latent_dim = SYSTEM_PARAMS.gnn.latent_dim
        self.skip_dim = SYSTEM_PARAMS.gnn.skip_dim
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
        preds_masked = (probs_masked > 0.5).to(y_masked)

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
        seed=42,
    ):
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
    meat_dataset = MyDataset(
        scheme="iros",
        sim_exp='apple',
        data_dir='banana',
        apply_augmentations='cherry',
        name='meat',
    )
    train_dataset, val_dataset, test_dataset = meat_dataset.create_splits()
    silicone_dataset = MyDataset(
        scheme="single_dataset",
        sim_exp="exp",
        data_dir=SYSTEM_PARAMS.files.exp_data_endgame,
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
        "iros": True,
    }
    with open(SYSTEM_PARAMS.files.test_loader_gnn_iros, "wb") as f:
        pickle.dump(test_data, f)
    model = GNN()
    model.set_stats(stats)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_iou/1",
        mode="max",
        save_top_k=1,
        filename="best-model-iros",
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
    end = time.perf_counter()
    duration = end - start
    print(
        f"\nTraining and testing completed in {duration:.2f} seconds ({duration / 60:.2f} minutes)"
    )
    os.makedirs("saved_models_iros", exist_ok=True)
    torch.save(best_model.state_dict(), SYSTEM_PARAMS.files.final_segmentation_model_gnn_iros)


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

def evaluate_and_plot_roc():
    if True:
        model = GNN()
        model.load_state_dict(torch.load(SYSTEM_PARAMS.files.final_segmentation_model_gnn_iros))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with open(SYSTEM_PARAMS.files.test_loader_gnn_iros, 'rb') as f:
            test_data = pickle.load(f)
        all_stats = test_data['dataset_stats']

        full_dataset = MyDataset(
            scheme="single_dataset",
            sim_exp="exp",
            data_dir=SYSTEM_PARAMS.files.exp_data_endgame,
            apply_augmentations=False,
            name='silicone',
        )
        train_dataset, _, _ = full_dataset.create_splits(
            train_size=1.0,
            val_size=0.0,
            test_size=0.0
        )
        if 'iros' in test_data:
            stats = test_data['dataset_stats']
        else:
            all_stats = test_data['dataset_stats']
            target_difficulty = 1.0
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

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch, labels_images, poses, metadata, frame_ix in data_loader:
                batch = batch.to(device)

                x, x_mask, edge_index, edge_index_regular_nodes, edge_attr = model.my_prepare_data(batch, batch.num_graphs)
                out = model(x, edge_index, edge_attr, batch.batch)
                out = out.squeeze(-1)
                out = out[x_mask]
                mask = batch.mask
                out = out[mask]
                probs = torch.sigmoid(out)
                labels = batch.y[mask]
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        thresholds = np.linspace(0.4, 0.6, 5)
        tpr_list, fpr_list = [], []
        for thr in thresholds:
            preds = (all_probs >= thr).astype(int)
            tp = np.sum((preds == 1) & (all_labels == 1))
            fp = np.sum((preds == 1) & (all_labels == 0))
            tn = np.sum((preds == 0) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))
            tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    
    if False:
        fpr_list = np.linspace(0, 1, 11)
        tpr_list = fpr_list ** 0.5
        fpr = fpr_list
        tpr = tpr_list
        auc = 0.75
        thresholds = np.linspace(0, 1, 11)

    fontsize = 20
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve", alpha=0.8, linewidth=6.0)
    plt.scatter(fpr_list, tpr_list, color="red", s=200, label="thresholds")

    for thr, x, y in zip(thresholds, fpr_list, tpr_list):
        plt.text(x, y, f"{thr:.2f}", fontsize=fontsize, ha="left", va="bottom", fontweight="bold")
    
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")

    plt.plot([0, 1], [0, 1], "k-", alpha=0.5, linewidth=6.0)
    plt.xlabel("False Positive Rate", fontsize=fontsize, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=fontsize, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=3.0)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3.0)
    plt.tight_layout()
    plt.savefig('difftactile/output/roc_curve.pdf', format="pdf", dpi=300)
    plt.show()

    return auc
