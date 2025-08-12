import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler, PassThroughProfiler
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from difftactile.cnn.dataset import *
from difftactile.cnn.lit_module import *
from difftactile.main.constants import *


def train():
    start_time = time.perf_counter()
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    NUM_WORKERS = 1
    LR = 1e-3

    logger = TensorBoardLogger("lightning_logs", name="segmentation_model")
    full_dataset = MyDataset(
        data_dir=SYSTEM_PARAMS.files.dataset_root
    )
    train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
        full_dataset, train_size=0.70, val_size=0.15, test_size=0.15
    )
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
    with open(SYSTEM_PARAMS.files.test_loader, 'wb') as f:
        pickle.dump(test_data, f)

    model = SegmentationModel(lr=LR)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_fg_iou",  # Changed from val_iou to val_fg_iou
        mode="max",
        save_top_k=1,
        filename="best-model",
    )
    early_stopping = EarlyStopping(
        monitor="val_fg_iou",  # Changed from val_iou to val_fg_iou
        mode="max",
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        min_delta=1e-4,  # Minimum change in monitored quantity to qualify as an improvement
        verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stopping],
        logger=logger,
        log_every_n_steps=1,
        # profiler="pytorch"
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    total_time = time.perf_counter() - start_time
    print(f'total execution time: {total_time}')

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), SYSTEM_PARAMS.files.final_segmentation_model)
    if checkpoint_cb.best_model_path:
        best_model_save_path = SYSTEM_PARAMS.files.best_segmentation_model
        torch.save(torch.load(checkpoint_cb.best_model_path)["state_dict"], best_model_save_path)


def choose_optimal_threshold():
    # Load test dataset
    with open(SYSTEM_PARAMS.files.test_loader, 'rb') as f:
        test_data = pickle.load(f)
    test_dataset = test_data['dataset']
    num_workers = test_data['num_workers']
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,  # Larger batch size for faster evaluation
        shuffle=False,
        num_workers=num_workers
    )
    
    # Load model
    model = SegmentationModel()
    model.load_state_dict(torch.load(SYSTEM_PARAMS.files.final_segmentation_model))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define threshold range for grid search
    thresholds = np.linspace(0.1, 0.9, 17)  # [0.1, 0.15, 0.2, ..., 0.85, 0.9]
    threshold_scores = {}
    
    print("Starting grid search for optimal threshold...")
    
    with torch.no_grad():
        for threshold in thresholds:
            metrics_sum = {
                'fg_iou': 0.0,
                'bg_iou': 0.0,
                'macro_iou': 0.0,
                'detection_rate': 0.0,
                'num_fg_frames': 0,
                'num_empty_frames': 0
            }
            
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                # Get model predictions
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                # Remove channel dimension
                preds = preds.squeeze(1)
                y = y.squeeze(1)
                
                # Compute frame-wise ground truth presence
                gt_presence = y.sum(dim=(1, 2)) > 0
                pred_presence = preds.sum(dim=(1, 2)) > 0
                
                # Handle frames with foreground
                has_fg_mask = gt_presence
                if has_fg_mask.sum() > 0:
                    # Foreground IoU
                    fg_intersection = (preds[has_fg_mask] * y[has_fg_mask]).sum(dim=(1, 2))
                    fg_union = (preds[has_fg_mask] + y[has_fg_mask]).sum(dim=(1, 2)) - fg_intersection
                    fg_iou = (fg_intersection + 1e-6) / (fg_union + 1e-6)
                    
                    # Background IoU
                    bg_preds = 1 - preds[has_fg_mask]
                    bg_targets = 1 - y[has_fg_mask]
                    bg_intersection = (bg_preds * bg_targets).sum(dim=(1, 2))
                    bg_union = (bg_preds + bg_targets).sum(dim=(1, 2)) - bg_intersection
                    bg_iou = (bg_intersection + 1e-6) / (bg_union + 1e-6)
                    
                    metrics_sum['fg_iou'] += fg_iou.sum().item()
                    metrics_sum['bg_iou'] += bg_iou.sum().item()
                    metrics_sum['num_fg_frames'] += has_fg_mask.sum().item()
                
                # Handle empty frames
                empty_mask = ~gt_presence
                if empty_mask.sum() > 0:
                    correct_empty = (~pred_presence[empty_mask]).float()
                    metrics_sum['detection_rate'] += correct_empty.sum().item()
                    metrics_sum['num_empty_frames'] += empty_mask.sum().item()
            
            # Compute averages
            avg_metrics = {}
            if metrics_sum['num_fg_frames'] > 0:
                avg_metrics['fg_iou'] = metrics_sum['fg_iou'] / metrics_sum['num_fg_frames']
                avg_metrics['bg_iou'] = metrics_sum['bg_iou'] / metrics_sum['num_fg_frames']
                avg_metrics['macro_iou'] = (avg_metrics['fg_iou'] + avg_metrics['bg_iou']) / 2
            else:
                avg_metrics['fg_iou'] = 0.0
                avg_metrics['bg_iou'] = 0.0
                avg_metrics['macro_iou'] = 0.0
            
            if metrics_sum['num_empty_frames'] > 0:
                avg_metrics['detection_rate'] = metrics_sum['detection_rate'] / metrics_sum['num_empty_frames']
            else:
                avg_metrics['detection_rate'] = 0.0
            
            # Store results
            threshold_scores[threshold] = avg_metrics
            print(f"Threshold {threshold:.2f}:")
            print(f"  Foreground IoU: {avg_metrics['fg_iou']:.4f}")
            print(f"  Background IoU: {avg_metrics['bg_iou']:.4f}")
            print(f"  Macro IoU: {avg_metrics['macro_iou']:.4f}")
            print(f"  Detection Rate: {avg_metrics['detection_rate']:.4f}")
    
    # Find best threshold based on foreground IoU
    best_threshold = max(threshold_scores.items(), key=lambda x: x[1]['fg_iou'])
    print(f"\nBest threshold: {best_threshold[0]:.2f}")
    print(f"  Foreground IoU: {best_threshold[1]['fg_iou']:.4f}")
    print(f"  Background IoU: {best_threshold[1]['bg_iou']:.4f}")
    print(f"  Macro IoU: {best_threshold[1]['macro_iou']:.4f}")
    print(f"  Detection Rate: {best_threshold[1]['detection_rate']:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot all metrics
    thresholds_list = list(threshold_scores.keys())
    plt.plot(thresholds_list, [scores['fg_iou'] for scores in threshold_scores.values()], 'b-', marker='o', label='Foreground IoU')
    plt.plot(thresholds_list, [scores['bg_iou'] for scores in threshold_scores.values()], 'g-', marker='o', label='Background IoU')
    plt.plot(thresholds_list, [scores['macro_iou'] for scores in threshold_scores.values()], 'r-', marker='o', label='Macro IoU')
    plt.plot(thresholds_list, [scores['detection_rate'] for scores in threshold_scores.values()], 'y-', marker='o', label='Detection Rate')
    
    plt.axvline(x=best_threshold[0], color='k', linestyle='--', label=f'Best threshold: {best_threshold[0]:.2f}')
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Segmentation Metrics vs. Threshold Value')
    plt.legend()
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(SYSTEM_PARAMS.files.threshold_optimization)
    plt.close()
    
    return best_threshold[0]


def main():
    train()


if __name__ == "__main__":
    main()
