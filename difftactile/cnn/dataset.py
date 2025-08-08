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

class MarkerTrajectoryDataset(Dataset):
    def __init__(self, data_dir, clip_len=16, transform=None):
        self.data_dir = data_dir
        self.clip_len = clip_len
        self.transform = transform
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)

        markers = data["marker_positions"]  # shape: (T, N, 2)
        total_frames = markers.shape[0]

        if total_frames < self.clip_len:
            raise ValueError(f"Clip too short: {total_frames} < {self.clip_len}")

        # Random temporal cropping
        start = random.randint(0, total_frames - self.clip_len)
        clip = markers[start:start + self.clip_len]  # shape: (clip_len, N, 2)

        # Optional: convert to tensor
        clip = torch.tensor(clip, dtype=torch.float32)  # shape: (T, N, 2)

        # Optional: apply transform
        if self.transform:
            clip = self.transform(clip)

        return clip


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode, clip_len=16, apply_augmentation=False, clips_per_trajectory=8):
        super().__init__()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.fisheye_model = FisheyeModel()
        self.data_dir = data_dir
        self.clip_len = clip_len
        self.mode = mode
        self.apply_augmentation = apply_augmentation
        self.clips_per_trajectory = clips_per_trajectory
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
        
        self.w = SYSTEM_PARAMS.fisheye_model.crop_width
        self.h = SYSTEM_PARAMS.fisheye_model.crop_height
        self.k = 4
        self.w_scaled = int(self.w / self.k)
        self.h_scaled = int(self.h / self.k)

        # Pre-compute valid clips for each trajectory
        self.clips = []
        for file_path in self.files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            total_frames = data["markers"].shape[0]
            
            if total_frames >= self.clip_len:
                if mode != 'train':
                    # Deterministic, evenly spaced clips for val/test
                    stride = max(1, (total_frames - self.clip_len) // self.clips_per_trajectory)
                    start_indices = list(range(0, total_frames - self.clip_len + 1, stride))[:self.clips_per_trajectory]
                else:
                    # For training, pre-compute more start indices than needed
                    # This maintains randomness while allowing reproducible splits
                    num_possible_starts = total_frames - self.clip_len + 1
                    start_indices = sorted(random.sample(
                        range(num_possible_starts), 
                        min(self.clips_per_trajectory, num_possible_starts)
                    ))
                
                for start_idx in start_indices:
                    self.clips.append((file_path, start_idx))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        file_path, start = self.clips[idx]
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        images_all = data["markers"]  # shape: (T, N, 2)
        labels_all = data["labels"]  # shape: (T, N, 2)
        
        images = images_all[start:start + self.clip_len]  # shape: (clip_len, N, 2)
        labels = labels_all[start:start + self.clip_len]  # shape: (clip_len, N, 2)
        
        if self.apply_augmentation:
            images = self.augmentation_rotation(images)
            labels = self.augmentation_rotation(labels)
            images = self.shift_radial(images)
            labels = self.shift_radial(labels)
            images = self.uniform_shift(images)
            labels = self.uniform_shift(labels)
            images = self.rotate_xy(images)
            labels = self.rotate_xy(labels)
            images = self.randomly_remove(images)
        
        images = self.downscale('markers', images)
        labels = self.downscale('vein', labels)
        
        images = self.generate_markers_image(images)
        labels = self.generate_vein_image(labels)

        images = torch.tensor(images, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return images, labels

    @staticmethod
    def create_splits(
        dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
    ):
        """Split dataset while ensuring all clips from the same trajectory stay together"""
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, (
            "Split proportions must sum to 1"
        )
        
        # Group indices by trajectory
        trajectory_to_indices = {}
        for i, (file_path, _) in enumerate(dataset.clips):
            trajectory_to_indices.setdefault(file_path, []).append(i)
        
        # Split trajectories
        trajectories = list(trajectory_to_indices.keys())
        random.seed(random_state)
        random.shuffle(trajectories)
        
        n = len(trajectories)
        train_split = int(n * train_size)
        val_split = int(n * (train_size + val_size))
        
        train_trajectories = trajectories[:train_split]
        val_trajectories = trajectories[train_split:val_split]
        test_trajectories = trajectories[val_split:]
        
        # Convert trajectory splits to index splits
        train_indices = [i for traj in train_trajectories for i in trajectory_to_indices[traj]]
        val_indices = [i for traj in val_trajectories for i in trajectory_to_indices[traj]]
        test_indices = [i for traj in test_trajectories for i in trajectory_to_indices[traj]]
        
        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )

    def generate_markers_image(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
        # Initialize output array for all frames
        images = np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            for point in points[t]:
                x, y = int(point[0]), int(point[1])
                cv2.circle(images[t], (x, y), radius=1, color=255, thickness=-1)
                
        return images

    def generate_vein_image(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
        # Initialize output array for all frames
        images = np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            if len(points[t]) > 0:
                contour_vein = self.synthetic_image_generator.alpha_shape(points[t], alpha=0.02).astype(np.int32)
                contour_vein_cv = contour_vein.reshape((-1, 1, 2))
                cv2.fillPoly(images[t], [contour_vein_cv], color=255)
                
        return images

    def augmentation_rotation(self, points):
        discrete_angles = [0, 60, 120, 180, 240, 300]
        angle_degrees = random.choice(discrete_angles)
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        
        if points.shape[1] == 0:  # Check if there are any points
            return points
            
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Center all points across all frames at once
        centered_points = points - np.array([cx, cy])
        
        # Initialize output array with same shape as input
        rotated_points = np.zeros_like(centered_points)
        
        # Apply rotation to all frames at once
        rotated_points[..., 0] = centered_points[..., 0] * cos_a - centered_points[..., 1] * sin_a
        rotated_points[..., 1] = centered_points[..., 0] * sin_a + centered_points[..., 1] * cos_a
        
        # Translate back
        rotated_points = rotated_points + np.array([cx, cy])

        return rotated_points

    def uniform_shift(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            return points
            
        # Generate random shift parameters - same for all frames
        angle_rad = random.uniform(0, 2 * math.pi)
        magnitude = random.uniform(0, 20)
        shift_x = magnitude * math.cos(angle_rad)
        shift_y = magnitude * math.sin(angle_rad)
        
        # Apply the same shift to all frames at once using broadcasting
        shifted_points = points + np.array([shift_x, shift_y])
        
        return shifted_points

    def shift_radial(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            return points
            
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        
        # Same radial shift for all frames
        radial_shift = random.uniform(-10, 10)
        
        # Calculate distances and unit vectors for all points at once
        dx = points[..., 0] - cx  # Broadcasting handles all frames
        dy = points[..., 1] - cy
        distances = np.sqrt(dx**2 + dy**2)
        
        # Create mask for non-zero distances to avoid division by zero
        nonzero_mask = distances > 0
        
        # Initialize output array
        shifted_points = points.copy()
        
        # Calculate unit vectors where distance > 0
        unit_x = np.zeros_like(dx)
        unit_y = np.zeros_like(dy)
        unit_x[nonzero_mask] = dx[nonzero_mask] / distances[nonzero_mask]
        unit_y[nonzero_mask] = dy[nonzero_mask] / distances[nonzero_mask]
        
        # Apply radial shift
        shifted_points[..., 0] = points[..., 0] + radial_shift * unit_x
        shifted_points[..., 1] = points[..., 1] + radial_shift * unit_y
        
        return shifted_points

    def rotate_xy(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            return points
            
        # Project all points to 3D at once
        points_3d = self.fisheye_model.project_pix_to_points_3d_plane(points.reshape(-1, 2)).reshape(points.shape[0], points.shape[1], 3)
        z_coord = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.trajectory.press_depth_1
        assert np.allclose(points_3d[..., 2], z_coord, atol=1e-6), f"All z-coordinates must be equal to {z_coord}"
        
        # Same rotation angles for all frames
        angle_x = math.radians(random.uniform(-10, 10))
        angle_y = math.radians(random.uniform(-10, 10))
        
        # Rotation matrices around axes through origin
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # Translate all points to origin at once
        translated_points = points_3d - np.array([0, 0, z_coord])
        
        # Reshape for matrix multiplication
        points_reshaped = translated_points.reshape(-1, 3).T  # (3, N*T)
        
        # Apply rotations to all points at once
        points_rotated_x = Rx @ points_reshaped  # (3, N*T)
        points_rotated_xy = Ry @ points_rotated_x  # (3, N*T)
        
        # Reshape back and translate back
        rotated_points_3d = (points_rotated_xy.T).reshape(points_3d.shape)  # (T, N, 3)
        rotated_points_3d = rotated_points_3d + np.array([0, 0, z_coord])
        
        # Project all points back to 2D at once
        rotated_points_2d = self.fisheye_model.project_3d_2d_np(rotated_points_3d.reshape(-1, 3)).reshape(points.shape)
        
        return rotated_points_2d
    
    def randomly_remove(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            return points
            
        # Determine number of points to remove (same for all frames)
        k = random.randint(0, min(20, points.shape[1]))
        if k == 0:
            return points
            
        # Select which points to remove (same indices across all frames)
        indices_to_remove = random.sample(range(points.shape[1]), k)
        
        # Create boolean mask for points to keep
        mask = np.ones(points.shape[1], dtype=bool)
        mask[indices_to_remove] = False
        
        # Apply mask to all frames at once using broadcasting
        # The mask is automatically broadcast across the time dimension
        return points[:, mask, :]

    def downscale(self, mode, points):
        if points.shape[1] == 0:  # Check if there are any points
            # Return empty images for all frames
            return np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
        # Process all frames
        if mode == 'markers':
            # Crop and scale all points
            markers = self.synthetic_image_generator.crop(points.reshape(-1, 2)).reshape(points.shape)
            markers = markers / self.k
            
            # Initialize output array for all frames
            markers_imgs = np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
            # Generate image for each frame
            for t in range(points.shape[0]):
                for point in markers[t]:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(markers_imgs[t], (x, y), radius=1, color=255, thickness=-1)
            return markers_imgs
            
        elif mode == 'vein':
            # Crop and scale all points
            vein = self.synthetic_image_generator.crop(points.reshape(-1, 2)).reshape(points.shape)
            vein = vein / self.k
            
            # Initialize output array for all frames
            vein_imgs = np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
            # Generate image for each frame
            for t in range(points.shape[0]):
                if len(vein[t]) > 0:
                    contour_vein = self.synthetic_image_generator.alpha_shape(vein[t], alpha=0.02).astype(np.int32)
                    contour_vein_cv = contour_vein.reshape((-1, 1, 2))
                    cv2.fillPoly(vein_imgs[t], [contour_vein_cv], color=255)
            return vein_imgs
            
        else:
            raise Exception("Invalid mode")

