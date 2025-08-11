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
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        filename="best-model",
    )
    early_stopping = EarlyStopping(
        monitor="val_iou",
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
            total_iou = 0.0
            num_batches = 0
            
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                # Get model predictions
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                # Calculate IoU score
                preds = preds.squeeze(1)  # Remove channel dimension
                y = y.squeeze(1)  # Remove channel dimension
                
                # Calculate IoU per frame and take mean
                intersection = (preds * y).sum(dim=(1, 2))  # Sum over spatial dimensions
                union = (preds + y).sum(dim=(1, 2)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero
                batch_iou = iou.mean().item()
                
                total_iou += batch_iou
                num_batches += 1
            
            avg_iou = total_iou / num_batches
            threshold_scores[threshold] = avg_iou
            print(f"Threshold {threshold:.2f}: Average IoU = {avg_iou:.4f}")
    
    # Find best threshold
    best_threshold = max(threshold_scores.items(), key=lambda x: x[1])
    print(f"\nBest threshold: {best_threshold[0]:.2f} with IoU: {best_threshold[1]:.4f}")
    
    # Plot results
    thresholds = list(threshold_scores.keys())
    scores = list(threshold_scores.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, 'b-', marker='o')
    plt.axvline(x=best_threshold[0], color='r', linestyle='--', label=f'Best threshold: {best_threshold[0]:.2f}')
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('IoU Score')
    plt.title('IoU Score vs. Threshold Value')
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
