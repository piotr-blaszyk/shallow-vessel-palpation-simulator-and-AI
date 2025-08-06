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
import math
from pathlib import Path

from difftactile.main.constants import *
from difftactile.main.main import SyntheticImageGenerator
from difftactile.sensor_model.fisheye_model import *


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
    def __init__(self, split, apply_augmentation=False):
        super().__init__()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.fisheye_model = FisheyeModel()
        self.split = split
        self.apply_augmentation = apply_augmentation
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
        
        self.w = SYSTEM_PARAMS.fisheye_model.crop_width
        self.h = SYSTEM_PARAMS.fisheye_model.crop_height
        self.k = 4
        self.w_scaled = int(self.w / self.k)
        self.h_scaled = int(self.h / self.k)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]
        
        with open(image_path, 'rb') as f:
            image = pickle.load(f)
        
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
        
        if self.apply_augmentation and self.split == 'train':
            
            image = self.augmentation_rotation(image)
            label = self.augmentation_rotation(label)
            image = self.shift_radial(image)
            label = self.shift_radial(label)
            image = self.uniform_shift(image)
            label = self.uniform_shift(label)
            image = self.rotate_xy(image)
            label = self.rotate_xy(label)
            image = self.randomly_remove(image)
        
        image = self.downscale('markers', image)
        label = self.downscale('vein', label)
        
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

    def augmentation_rotation(self, points):
        discrete_angles = [0, 60, 120, 180, 240, 300]
        angle_degrees = random.choice(discrete_angles)
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        if len(points) == 0:
            return points
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        centered_points = points - np.array([cx, cy])
        rotated_points = np.zeros_like(centered_points)
        rotated_points[:, 0] = centered_points[:, 0] * cos_a - centered_points[:, 1] * sin_a
        rotated_points[:, 1] = centered_points[:, 0] * sin_a + centered_points[:, 1] * cos_a
        rotated_points = rotated_points + np.array([cx, cy])
        return rotated_points

    def uniform_shift(self, points):
        if len(points) == 0:
            return points
        angle_rad = random.uniform(0, 2 * math.pi)
        magnitude = random.uniform(0, 20)
        shift_x = magnitude * math.cos(angle_rad)
        shift_y = magnitude * math.sin(angle_rad)
        shifted_points = points + np.array([shift_x, shift_y])
        return shifted_points

    def shift_radial(self, points):
        if len(points) == 0:
            return points
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        radial_shift = random.uniform(-10, 10)
        shifted_points = np.zeros_like(points)
        for i, point in enumerate(points):
            dx = point[0] - cx
            dy = point[1] - cy
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 0:
                unit_x = dx / distance
                unit_y = dy / distance
                shifted_points[i, 0] = point[0] + radial_shift * unit_x
                shifted_points[i, 1] = point[1] + radial_shift * unit_y
            else:
                shifted_points[i] = point
        return shifted_points

    def rotate_xy(self, points):
        if len(points) == 0:
            return points
            
        points_3d = self.fisheye_model.project_pix_to_points_3d_plane(points)
        z_coord = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.trajectory.press_depth_1
        assert np.allclose(points_3d[:, 2], z_coord, atol=1e-6), f"All z-coordinates must be equal to {z_coord}"
        
        angle_x = math.radians(random.uniform(-10, 10))
        angle_y = math.radians(random.uniform(-10, 10))
        
        # Rotation matrices around axes through origin
        cos_x = math.cos(angle_x)
        sin_x = math.sin(angle_x)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # Apply rotations: translate to origin, rotate, translate back
        rotated_points_3d = np.zeros_like(points_3d)
        for i, point_3d in enumerate(points_3d):
            # Translate so that (0,0,z_coord) becomes origin
            translated_point = point_3d - np.array([0, 0, z_coord])
            
            # Apply x-axis rotation around origin
            point_rotated_x = Rx @ translated_point
            
            # Apply y-axis rotation around origin
            point_rotated_xy = Ry @ point_rotated_x
            
            # Translate back
            rotated_points_3d[i] = point_rotated_xy + np.array([0, 0, z_coord])
        
        # Project back to 2D
        rotated_points_2d = self.fisheye_model.project_3d_2d_np(rotated_points_3d)
        
        return rotated_points_2d
    
    def randomly_remove(self, points):
        if len(points) == 0:
            return points
        k = random.randint(0, min(20, len(points)))
        if k == 0:
            return points
        indices_to_remove = random.sample(range(len(points)), k)
        mask = np.ones(len(points), dtype=bool)
        mask[indices_to_remove] = False
        return points[mask]

    def downscale(self, mode, points):
        if mode == 'markers':
            markers = self.synthetic_image_generator.crop(points)
            markers /= self.k
            markers_img = np.zeros((self.w_scaled, self.h_scaled), dtype=np.uint8)
            for point in markers:
                x, y = int(point[0]), int(point[1])
                cv2.circle(markers_img, (x, y), radius=1, color=255, thickness=-1)
            return markers_img
        elif mode == 'vein':
            vein = self.synthetic_image_generator.crop(points)
            vein /= self.k
            vein_img = np.zeros((self.w_scaled, self.h_scaled), dtype=np.uint8)
            if len(vein) > 0:
                contour_vein = self.synthetic_image_generator.alpha_shape(vein, alpha=0.02).astype(np.int32)
                contour_vein_cv = contour_vein.reshape((-1, 1, 2))
                cv2.fillPoly(vein_img, [contour_vein_cv], color=255)
            return vein_img
        else:
            raise Exception()

