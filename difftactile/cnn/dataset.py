import os
import cv2
import torch
from torch.utils.data import Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import shutil
import random
from pathlib import Path

from difftactile.main.constants import *
from difftactile.main.main import SyntheticImageGenerator


class SegmentationDataset(Dataset):
    def __init__(self, markers_pickle_dir, vein_pickle_dir):
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.markers_pickle_dir = markers_pickle_dir
        self.vein_pickle_dir = vein_pickle_dir
        self.pickle_files = sorted([f for f in os.listdir(markers_pickle_dir) if f.endswith('.pickle')])[:1_000]
        
        self.target_size = SYSTEM_PARAMS.cnn.target_size
        self.pad_h = self.target_size - SYSTEM_PARAMS.cnn.input_size
        self.pad_w = self.target_size - SYSTEM_PARAMS.cnn.input_size
        self.pad_top = self.pad_h // 2
        self.pad_bottom = self.pad_h - self.pad_top
        self.pad_left = self.pad_w // 2
        self.pad_right = self.pad_w - self.pad_left

    def __len__(self):
        return len(self.pickle_files)

    def split_files(self):
        random.seed(42)
        xs = [
            'markers',
            'segmentation_mask'
        ]
        for x in xs:
            root = SYSTEM_PARAMS.files.dataset_root
            cur_dir = f'{root}/pickle/{x}'
            trajectory_dirs = sorted(os.listdir(cur_dir))
            random.shuffle(trajectory_dirs)

            n = len(trajectory_dirs)
            train_split = int(0.8 * n)
            val_split = int(0.9 * n)

            splits = {
                'train': trajectory_dirs[:train_split],
                'val': trajectory_dirs[train_split:val_split],
                'test': trajectory_dirs[val_split:]
            }

            for split, trajs in splits.items():
                os.makedirs(f'dataset/{split}', exist_ok=True)
                for traj in trajs:
                    shutil.copy(f'all_trajectories/{traj}', f'dataset/{split}/{traj}')

    def generate_image_from_points(self, points, w, h):
        if len(points) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        points = points.copy()
        points = points.astype(np.float32)
        img = np.zeros((h, w), dtype=np.uint8)
        if points.shape[0] > 3:
            try:
                contour = self.synthetic_image_generator.alpha_shape(points, alpha=0.02).astype(np.int32)
                contour_cv = contour.reshape((-1, 1, 2))
                cv2.fillPoly(img, [contour_cv], color=255)
            except:
                for point in points:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(img, (x, y), radius=1, color=255, thickness=-1)
        else:
            for point in points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (x, y), radius=1, color=255, thickness=-1)
        return img

    def __getitem__(self, idx):
        pickle_file = self.pickle_files[idx]
        markers_path = os.path.join(self.markers_pickle_dir, pickle_file)
        with open(markers_path, 'rb') as f:
            markers_data = pickle.load(f)
        vein_path = os.path.join(self.vein_pickle_dir, pickle_file)
        with open(vein_path, 'rb') as f:
            vein_data = pickle.load(f)
        markers = markers_data[-1] if markers_data else np.array([])
        vein = vein_data[-1] if vein_data else np.array([])
        w = h = SYSTEM_PARAMS.cnn.input_size
        markers_img = self.generate_image_from_points(markers, w, h)
        vein_img = self.generate_image_from_points(vein, w, h)
        markers_img = cv2.copyMakeBorder(
            markers_img,
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            cv2.BORDER_REPLICATE
        )
        vein_img = cv2.copyMakeBorder(
            vein_img,
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            cv2.BORDER_REPLICATE
        )
        markers_img = markers_img.astype(np.float32) / 255.0
        vein_img = (vein_img > 127).astype(np.float32)
        return markers_img, vein_img

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


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.split = split
        self.image_dir = Path(f'{SYSTEM_PARAMS.files.dataset_root}/splits/{split}/images')
        self.label_dir = Path(f'{SYSTEM_PARAMS.files.dataset_root}/splits/{split}/labels')
        self.trajectory_dirs = sorted([d for d in self.image_dir.iterdir() if d.is_dir()])

        self.samples = []
        for traj_dir in self.trajectory_dirs:
            traj_num = int(traj_dir.name.split('_')[1])
            image_files = sorted(traj_dir.glob('img_*.pkl'))
            
            for img_file in image_files:
                img_num = int(img_file.stem.split('_')[1])
                label_file = self.label_dir / f'trajectory_{traj_num:04d}' / f'img_{img_num:04d}.pkl'
                if label_file.exists():
                    self.samples.append((img_file, label_file))
        
        w = SYSTEM_PARAMS.fisheye_model.crop_width
        h = SYSTEM_PARAMS.fisheye_model.crop_height
        k = 4
        self.w_scaled = int(w / k)
        self.h_scaled = int(h / k)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]
        
        with open(image_path, 'rb') as f:
            image = pickle.load(f)
        
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
        
        image = self.generate_markers_image(image)
        label = self.generate_vein_image(label)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)
        
        return image, label

    def generate_markers_image(self, points):
        image = np.zeros((self.w_scaled, self.h_scaled), dtype=np.uint8)
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), radius=1, color=255, thickness=-1)
        return image

    def generate_vein_image(self, points):
        image = np.zeros((self.w_scaled, self.h_scaled), dtype=np.uint8)
        if len(points) > 0:
            contour_vein = self.synthetic_image_generator.alpha_shape(points, alpha=0.02).astype(np.int32)
            contour_vein_cv = contour_vein.reshape((-1, 1, 2))
            cv2.fillPoly(image, [contour_vein_cv], color=255)
        return image
