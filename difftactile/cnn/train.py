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


class CurriculumCallback(pl.Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch == 10:
            self.dataset.set_difficulty_level(1)
        elif epoch == 20:
            self.dataset.set_difficulty_level(2)

class LossWeightScheduler(pl.Callback):
    def __init__(self):
        self.min_alpha = [0.5, 0.6, 0.7]
        self.max_alpha = [0.8, 0.85, 0.9]

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        i = epoch // 10
        j = epoch % 10
        min_alpha = self.min_alpha[i]
        max_alpha = self.max_alpha[i]
        alpha = min_alpha + (max_alpha - min_alpha) * (j / 9)
        pl_module.alpha = alpha
        

def train():
    start_time = time.perf_counter()
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    NUM_WORKERS = 16
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
        callbacks=[
            checkpoint_cb, 
            early_stopping, 
            CurriculumCallback(train_dataset),
            LossWeightScheduler()
            ],
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
    thresholds = np.linspace(0.7, 0.9, 3)  # [0.1, 0.15, 0.2, ..., 0.85, 0.9]
    threshold_scores = {}
    
    print("Starting grid search for optimal threshold...")
    
    with torch.no_grad():
        for threshold in thresholds:
            batch_metrics = {
                'fg_iou': [],
                'bg_iou': [],
                'macro_iou': [],
                'detection_rate': []
            }
            
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                # Get model predictions
                logits = model(x)  # Shape: [B, 1, T, H, W]
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                # Remove the channel dimension to match expected shape [B, T, H, W]
                preds = preds.squeeze(1)
                y = y.squeeze(1)
                
                # Compute metrics using the classmethod
                metrics = SegmentationModel.iou_score(preds, y)  # Now handles 5D tensors properly
                
                # Store batch metrics
                for key in batch_metrics:
                    if torch.is_tensor(metrics[key]):
                        batch_metrics[key].append(metrics[key].item())
                    else:
                        batch_metrics[key].append(metrics[key])
            
            # Compute averages across batches
            avg_metrics = {
                key: np.mean(values) if values else 0.0 
                for key, values in batch_metrics.items()
            }
            
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
