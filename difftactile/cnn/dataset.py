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
import time

from difftactile.main.constants import *
from difftactile.main.main import SyntheticImageGenerator
from difftactile.sensor_model.fisheye_model import *


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode=None, clip_len=16, apply_augmentation=True, clips_per_trajectory=4):
        super().__init__()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.fisheye_model = FisheyeModel()
        self.data_dir = data_dir
        self.clip_len = clip_len
        self.apply_augmentation = apply_augmentation
        self.clips_per_trajectory = clips_per_trajectory
        self.mode = mode
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        
        self.w = SYSTEM_PARAMS.fisheye_model.crop_width
        self.h = SYSTEM_PARAMS.fisheye_model.crop_height
        self.k = 4
        self.w_scaled = int(self.w / self.k)
        self.h_scaled = int(self.h / self.k)
        self.avg_call_time = 0.0
        self.num_calls = 0

        # Pre-compute valid clips for each trajectory
        self.clips = []
        # Possible dilation factors
        dilations = [1, 2, 3]
        
        for file_path in self.files:
            data = np.load(file_path)
            total_frames = len(data["markers"])
            
            # For each dilation factor
            for dilation in dilations:
                # Calculate required clip length for this dilation
                dilated_clip_len = self.clip_len * dilation
                
                if total_frames >= dilated_clip_len:
                    if self.mode != 'train':
                        # Deterministic, evenly spaced clips for val/test
                        stride = max(1, (total_frames - dilated_clip_len) // self.clips_per_trajectory)
                        start_indices = list(range(0, total_frames - dilated_clip_len + 1, stride))[:self.clips_per_trajectory]
                    else:
                        # For training, pre-compute more start indices than needed
                        num_possible_starts = total_frames - dilated_clip_len + 1
                        start_indices = sorted(random.sample(
                            range(num_possible_starts), 
                            min(self.clips_per_trajectory, num_possible_starts)
                        ))
                    
                    for start_idx in start_indices:
                        self.clips.append((file_path, start_idx, dilation))

    def __len__(self):
        return len(self.clips)

    @staticmethod
    def create_splits(
        dataset, train_size, val_size, test_size, random_state=42
    ):
        """Split dataset while ensuring all clips from the same trajectory stay together"""
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, (
            "Split proportions must sum to 1"
        )
        
        # Group indices by trajectory
        trajectory_to_indices = {}
        for i, (file_path, _, _) in enumerate(dataset.clips):
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
        
        res = (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )
        res[0].dataset.mode = 'train'
        res[1].dataset.mode = 'val'
        res[2].dataset.mode = 'test'
        print(f'split len: {[len(x) for x in res]}')
        return res

    def __getitem__(self, idx):
        file_path, start, dilation = self.clips[idx]

        data = np.load(file_path)
        # np.load returns a dict-like object whose keys we can access directly
        markers = data['markers']
        markers_mask = data['markers_mask']
        labels = data['labels']
        labels_mask = data['labels_mask']
        
        # Extract longer sequence and apply dilation
        dilated_clip_len = self.clip_len * dilation
        images = markers[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        labels = labels[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        images_mask = markers_mask[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        labels_mask = labels_mask[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        
        if self.mode == 'train' and self.apply_augmentation:
            images = self.augmentation_rotation(images)
            labels = self.augmentation_rotation(labels)
            images = self.shift_radial(images)
            labels = self.shift_radial(labels)
            images = self.uniform_shift(images)
            labels = self.uniform_shift(labels)
            images = self.rotate_xy(images)
            labels = self.rotate_xy(labels)
            images = self.randomly_remove(images)
        
        images, images_crop_mask = self.downscale(images)
        labels, labels_crop_mask = self.downscale(labels)

        images_mask &= images_crop_mask
        labels_mask &= labels_crop_mask
        
        images = self.generate_markers_image(images, images_mask)  # shape: (T, H, W)
        labels = self.generate_vein_image(labels, labels_mask)     # shape: (T, H, W)

        # Convert to float and normalize
        images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        labels = torch.tensor(labels, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        
        # Add channel dimension: (T, H, W) -> (C, T, H, W)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)
        
        return images, labels

    def generate_markers_image(self, points, points_mask):
        if points.shape[1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
        # Initialize output array for all frames
        images = np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            for i, point in enumerate(points[t]):
                # Only draw circle if the mask indicates this point is valid
                if points_mask[t, i]:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(images[t], (x, y), radius=1, color=255, thickness=-1)
                
        return images

    def generate_vein_image(self, points, points_mask):
        if points.shape[1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
        # Initialize output array for all frames
        images = np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            # Get only valid points according to mask
            valid_points = points[t][points_mask[t]]
            if len(valid_points) > 0:
                contour_vein = self.synthetic_image_generator.alpha_shape(valid_points, alpha=0.02).astype(np.int32)
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

    def downscale(self, points):
        if points.shape[1] == 0:  # Check if there are any points
            # Return empty images for all frames
            return np.zeros((points.shape[0], self.w_scaled, self.h_scaled), dtype=np.uint8)
            
        points, mask = self.synthetic_image_generator.crop(points)
        points = points / self.k
        return points, mask
