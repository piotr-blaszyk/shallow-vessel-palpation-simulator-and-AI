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

from difftactile.cnn.dataset import *
from difftactile.cnn.lit_module import *
from difftactile.main.constants import *


def main():
    start_time = time.perf_counter()
    BATCH_SIZE = 32
    NUM_EPOCHS = 360
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
    torch.save(model.state_dict(), "saved_models/final_segmentation_model.pt")
    if checkpoint_cb.best_model_path:
        best_model_save_path = "saved_models/best_segmentation_model.pt"
        torch.save(torch.load(checkpoint_cb.best_model_path)["state_dict"], best_model_save_path)


if __name__ == "__main__":
    main()
