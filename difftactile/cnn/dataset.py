import os
import cv2
import torch
from torch.utils.data import Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import numpy as np

from difftactile.main.constants import *


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))[:100]
        
        self.target_size = SYSTEM_PARAMS.cnn.target_size
        self.pad_h = self.target_size - SYSTEM_PARAMS.cnn.input_size
        self.pad_w = self.target_size - SYSTEM_PARAMS.cnn.input_size
        self.pad_top = self.pad_h // 2
        self.pad_bottom = self.pad_h - self.pad_top
        self.pad_left = self.pad_w // 2
        self.pad_right = self.pad_w - self.pad_left

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.copyMakeBorder(
            image, 
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            cv2.BORDER_REFLECT
        )
        mask = cv2.copyMakeBorder(
            mask, 
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            cv2.BORDER_REFLECT
        )
        
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        return image, mask

    @staticmethod
    def create_splits(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
    ):
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, (
            "Split proportions must sum to 1"
        )
        indices = np.arange(len(dataset))
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_size, random_state=random_state
        )
        relative_val_size = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=relative_val_size, random_state=random_state
        )
        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx),
        )

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask