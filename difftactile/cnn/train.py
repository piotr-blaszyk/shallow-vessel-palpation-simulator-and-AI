# train.py

import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SegmentationDataset
from lit_module import SegmentationModel

# Parameters
IMG_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 25
NUM_WORKERS = 4
LR = 1e-3

# Albumentations Transforms
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

# Datasets & Dataloaders
train_dataset = SegmentationDataset("images/train", "masks/train", transforms=train_transforms)
val_dataset = SegmentationDataset("images/val", "masks/val", transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# Model
model = SegmentationModel(lr=LR)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    monitor="val_iou",
    mode="max",
    save_top_k=1,
    filename="best-model",
)

# Trainer
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator="auto",
    callbacks=[checkpoint_cb],
    log_every_n_steps=10,
)

# Train
trainer.fit(model, train_loader, val_loader)
