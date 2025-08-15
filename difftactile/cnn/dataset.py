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
from difftactile.main.main import *
from difftactile.sensor_model.fisheye_model import *


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode=None):
        super().__init__()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.fisheye_model = FisheyeModel()
        self.data_dir = data_dir
        self.clip_len = SYSTEM_PARAMS.cnn.clip_len
        self.clips_per_trajectory = 16
        self.mode = mode
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

        self.w_camera_big = int(SYSTEM_PARAMS.fisheye_model.target_image_width)
        self.h_camera_big = int(SYSTEM_PARAMS.fisheye_model.target_image_height)
        
        self.w_crop_big = SYSTEM_PARAMS.fisheye_model.crop_width
        self.h_crop_big = SYSTEM_PARAMS.fisheye_model.crop_height
        self.k = SYSTEM_PARAMS.fisheye_model.down_scaling_factor
        self.w_crop_small = int(self.w_crop_big / self.k)
        self.h_crop_small = int(self.h_crop_big / self.k)
        # self.w_scaled = 1920
        # self.h_scaled = 1080
        self.avg_call_time = 0.0
        self.num_calls = 0
        self.difficulty_level = 0

        self.randomly_remove_k = [0, 6, 13]
        self.avs_disp_c = [0.4, 0.3, 0.2]
        self.avs_decay_c = [1.5, 2.25, 3.0]

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
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    
                    if False and not self.file_contains_vein(file_path):
                        # For files without veins, keep original random sampling
                        start_indices = sorted(random.sample(
                            range(num_possible_starts), 
                            min(self.clips_per_trajectory, num_possible_starts)
                        ))
                        for start_idx in start_indices:
                            self.clips.append((file_path, start_idx, dilation))
                    else:
                        # For files with veins, keep sampling until we find clips with veins
                        clips_found = 0
                        max_attempts = num_possible_starts * 2  # Prevent infinite loop
                        attempts = 0
                        
                        while clips_found < self.clips_per_trajectory and attempts < max_attempts:
                            start_idx = random.randrange(num_possible_starts)
                            if self.clip_contains_vein(file_path, start_idx, dilation):
                                self.clips.append((file_path, start_idx, dilation))
                                clips_found += 1
                            attempts += 1

    def __len__(self):
        return len(self.clips)

    def set_difficulty_level(self, level):
        self.difficulty_level = level
        print(f'new difficulty: {self.difficulty_level}')

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
        # random.seed(random_state)
        # random.shuffle(trajectories)
        
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
        
        # Create new dataset instances for each split
        train_dataset = MyDataset(dataset.data_dir, mode='train')
        train_dataset.clips = [dataset.clips[i] for i in train_indices]
        
        val_dataset = MyDataset(dataset.data_dir, mode='val')
        val_dataset.clips = [dataset.clips[i] for i in val_indices]
        
        test_dataset = MyDataset(dataset.data_dir, mode='test')
        test_dataset.clips = [dataset.clips[i] for i in test_indices]
        
        res = (train_dataset, val_dataset, test_dataset)
        print(f'split len: {[len(x) for x in res]}')
        return res

    def file_contains_vein(self, file_path):
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        
        # Extract the trajectory number from the file name
        # Format is "trajectory_XXXX.npz" where XXXX is a 4-digit number
        trajectory_num = int(file_name.split('_')[1][:4])
        
        # Return True if the number is even, False if odd
        return trajectory_num % 2 == 0

    def clip_contains_vein(self, file_path, start, dilation):
        data = np.load(file_path)
        markers = data['markers']
        markers_mask = data['markers_mask']
        labels = data['labels']
        labels_mask = data['labels_mask']

        centre = MyDataset.compute_mean_marker_position(markers, markers_mask)
        
        dilated_clip_len = self.clip_len * dilation
        labels = labels[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        labels_mask = labels_mask[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame

        cx = centre[0]
        cy = centre[1]
        r = SYSTEM_PARAMS.fisheye_model.circle_radius / 3
        mask2 = SyntheticImageGenerator.filter_points_vectorised(cx, cy, r, labels)
        labels_mask &= mask2

        res = labels[labels_mask].sum() > 0
        return res

    @staticmethod
    def compute_mean_marker_position(markers, markers_mask):
        # Compute mean for each frame using only valid markers
        frame_means = []
        for frame_idx in range(markers.shape[0]):
            valid_markers = markers[frame_idx][markers_mask[frame_idx]]
            if len(valid_markers) > 0:  # Only compute mean if we have valid markers
                frame_mean = np.mean(valid_markers, axis=0)
                frame_means.append(frame_mean)
        
        # Compute overall mean across all frame means
        if frame_means:  # Check if we have any valid frames
            return np.mean(frame_means, axis=0)
        else:
            return np.array([-1., -1.])  # Return invalid marker position if no valid data

    def __getitem__(self, idx):
        file_path, start, dilation = self.clips[idx]

        # file_vein = self.file_contains_vein(file_path)
        # print(f'file contains vein: {file_vein}')

        data = np.load(file_path)
        # np.load returns a dict-like object whose keys we can access directly
        markers = data['markers']
        markers_mask = data['markers_mask']
        labels = data['labels']
        labels_mask = data['labels_mask']
        vein_endpoints = data['vein_endpoints']
        vein_endpoints_mask = data['vein_endpoints_mask']
        
        # Extract longer sequence and apply dilation
        dilated_clip_len = self.clip_len * dilation
        images = markers[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        labels = labels[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        images_mask = markers_mask[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        labels_mask = labels_mask[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        vein_endpoints = vein_endpoints[start:start + dilated_clip_len:dilation]
        vein_endpoints_mask = vein_endpoints_mask[start:start + dilated_clip_len:dilation]
        
        # images, labels_signal_mask = self.augmentation_artificial_vein_signal(
        #     images, vein_endpoints, vein_endpoints_mask
        # )
        # # Expand labels_signal_mask to broadcast with shape (num_video_frames, num_veins, num_points)
        # labels_mask &= labels_signal_mask[:, np.newaxis, np.newaxis]
        # discrete_angles = [0, 60, 120, 180, 240, 300]
        # rotation_angle_deg = random.choice(discrete_angles)
        # images = self.augmentation_rotation(images, rotation_angle_deg)
        # labels = self.augmentation_rotation(labels, rotation_angle_deg)
        # angle_uniform_shift_rad = random.uniform(0, 2 * math.pi)
        # magnitude = random.uniform(0, 10)
        # images = self.uniform_shift(images, angle_uniform_shift_rad, magnitude)
        # labels = self.uniform_shift(labels, angle_uniform_shift_rad, magnitude)
        # angle_x = math.radians(random.uniform(-5, 5))
        # angle_y = math.radians(random.uniform(-5, 5))
        # images = self.rotate_xy(images, angle_x, angle_y)
        # labels = self.rotate_xy(labels, angle_x, angle_y)
        # images_random_remove_mask = self.randomly_remove(images)
        # images_mask &= images_random_remove_mask
        
        
        images, images_crop_mask = MyDataset.downscale(self.k, images)
        labels, labels_crop_mask = MyDataset.downscale(self.k, labels)

        images_mask &= images_crop_mask
        labels_mask &= labels_crop_mask

        images, marker_masks = MyDataset.generate_markers_image(self.h_crop_small, self.w_crop_small, images, images_mask)  # shape: (T, H, W)
        labels = MyDataset.generate_vein_image(self.h_crop_small, self.w_crop_small, labels, labels_mask)     # shape: (T, H, W)
        labels[marker_masks != 255] = 0
        
        # Convert to float and normalize
        images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        labels = torch.tensor(labels, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        
        # Add channel dimension: (T, H, W) -> (C, T, H, W)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)
        
        return images, labels

    @staticmethod
    def get_clip(h, w, k, data, clip_len, dilation, start_ix):
        markers = data['markers']
        markers_mask = data['markers_mask']

        dilated_clip_len = clip_len * dilation
        images = markers[start_ix:start_ix + dilated_clip_len:dilation]  # Take every dilation-th frame
        images_mask = markers_mask[start_ix:start_ix + dilated_clip_len:dilation]  # Take every dilation-th frame
        
        images, images_crop_mask = MyDataset.downscale(k, images)
        images_mask &= images_crop_mask
        
        h_scaled = h // k
        w_scaled = w // k
        images, _ = MyDataset.generate_markers_image(h_scaled, w_scaled, images, images_mask)  # shape: (T, H, W)

        # Convert to float and normalize
        images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        
        # Add channel dimension: (T, H, W) -> (C, T, H, W)
        images = images.unsqueeze(0).unsqueeze(0)
        
        return images

    @staticmethod
    def generate_markers_image(h, w, points, points_mask):
        if points.shape[-1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], h, w), dtype=np.uint8)
            
        # Initialize output array for all frames
        images = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        marker_masks = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            filtered_points = []
            for i, point in enumerate(points[t]):
                # Only set pixel if the mask indicates this point is valid
                if points_mask[t, i]:
                    filtered_points.append(point)
                    x, y = point[0], point[1]
                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    x1, y1 = x0 + 1, y0 + 1
                    
                    # Calculate weights for bilinear interpolation
                    wx1, wx0 = x - x0, x1 - x
                    wy1, wy0 = y - y0, y1 - y
                    
                    # Ensure we don't write outside the image bounds
                    if 0 <= x0 < w and 0 <= y0 < h:
                        images[t, y0, x0] = min(255, images[t, y0, x0] + int(255 * wx0 * wy0))
                    if 0 <= x1 < w and 0 <= y0 < h:
                        images[t, y0, x1] = min(255, images[t, y0, x1] + int(255 * wx1 * wy0))
                    if 0 <= x0 < w and 0 <= y1 < h:
                        images[t, y1, x0] = min(255, images[t, y1, x0] + int(255 * wx0 * wy1))
                    if 0 <= x1 < w and 0 <= y1 < h:
                        images[t, y1, x1] = min(255, images[t, y1, x1] + int(255 * wx1 * wy1))
            markers_mask = Contact.compute_mask(h, w, filtered_points)
            marker_masks[t, :, :] = markers_mask
                
        return images, marker_masks

    @staticmethod
    def generate_vein_image(h, w, points, points_mask):
        if points.shape[-1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], h, w), dtype=np.uint8)
            
        # Initialize output array for all frames
        images = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            # Process each vein separately
            for v in range(points.shape[1]):
                # Get only valid points according to mask for this vein
                vein_points = points[t, v]
                vein_mask = points_mask[t, v]
                valid_points = vein_points[vein_mask]
                
                if len(valid_points) > 0:
                    # Remove duplicate points
                    contour_vein = SyntheticImageGenerator.alpha_shape(valid_points).astype(np.int32)
                    if len(contour_vein) > 0:
                        contour_vein_cv = contour_vein.reshape((-1, 1, 2))
                        cv2.fillPoly(images[t], [contour_vein_cv], color=255)
                
        return images

    def augmentation_rotation(self, points, angle_degrees):
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        
        if points.shape[-1] == 0:  # Check if there are any points
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

    def uniform_shift(self, points, angle_rad, magnitude):
        if points.shape[-1] == 0:  # Check if there are any points
            return points
            
        shift_x = magnitude * math.cos(angle_rad)
        shift_y = magnitude * math.sin(angle_rad)
        
        # Apply the same shift to all frames at once using broadcasting
        shifted_points = points + np.array([shift_x, shift_y])
        
        return shifted_points

    def shift_radial(self, points, radial_shift):
        if points.shape[-1] == 0:  # Check if there are any points
            return points
            
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        
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

    def rotate_xy(self, points, angle_x, angle_y):
        if points.shape[-1] == 0:  # Check if there are any points
            return points
            
        # Project all points to 3D at once
        points_3d = self.fisheye_model.project_pix_to_points_3d_plane(points)
        z_coord = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.trajectory.press_depth_1
        assert np.allclose(points_3d[..., 2], z_coord, atol=1e-6), f"All z-coordinates must be equal to {z_coord}"
        
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
        if points.shape[-1] == 0:  # Check if there are any points
            return np.ones((points.shape[0], 0), dtype=bool)  # Return empty mask matching input shape
            
        # Get max number of points that can be removed
        max_k = self.randomly_remove_k[self.difficulty_level]
        
        # Initialize mask array for all frames
        mask = np.ones(points.shape[:2], dtype=bool)  # Shape: (n, num_points)
        
        # For each frame, randomly remove points
        for frame_idx in range(points.shape[0]):
            # Determine number of points to remove for this frame
            k = random.randint(0, min(max_k, points.shape[1]))
            
            if k > 0:  # Only modify mask if we're removing points
                # Select which points to remove for this frame
                indices_to_remove = random.sample(range(points.shape[1]), k)
                mask[frame_idx, indices_to_remove] = False
        
        return mask  # Shape: (n, num_points)

    @staticmethod
    def downscale(k, points):
        if points.shape[-1] == 0:  # Check if there are any points
            return points, np.ones((points.shape[0], 0), dtype=bool)  # Return empty mask matching input shape
            
        points, mask = SyntheticImageGenerator.crop(points)
        points = points / k
        return points, mask

    def augmentation_artificial_vein_signal(self, clip_points, clip_vein_endpoints, clip_vein_endpoints_mask):
        valid_frames_mask = np.any(np.all(clip_vein_endpoints_mask, axis=2), axis=1)
        vein_visible_mask = np.ones(clip_points.shape[0], dtype=bool)
        if not np.any(valid_frames_mask):
            return clip_points, vein_visible_mask
        
        padded = np.concatenate(([False], valid_frames_mask, [False]))
        runs = np.where(np.diff(padded))[0]
        run_lengths = runs[1::2] - runs[::2]
        if len(run_lengths) > 0:
            longest_run_idx = np.argmax(run_lengths)
            start_idx = runs[::2][longest_run_idx]
            end_idx = runs[1::2][longest_run_idx]
        else:
            start_idx = 0
            end_idx = 0
        if self.difficulty_level == 2:
            max_length = end_idx - start_idx
            if max_length < 3:
                length = 0
            else:
                length = random.randint(3, max_length)
            end_idx -= length
            start_idx = random.randint(start_idx, end_idx)
            end_idx = start_idx + length

            vein_visible_mask = np.zeros(clip_points.shape[0], dtype=bool)
            vein_visible_mask[start_idx:end_idx] = True

        # lower_disp_c = self.avs_disp_c[self.difficulty_level]
        # disp_c = random.uniform(lower_disp_c, 0.5)
        disp_c = 1.0
        # upper_decay_c = self.avs_decay_c[self.difficulty_level]
        # decay_c = random.uniform(1, upper_decay_c)
        decay_c = 0.0

        for j in range(start_idx, end_idx):
            points = clip_points[j]
            centre = np.mean(points, axis=0)
            max_dist = np.max(np.linalg.norm(points - centre, axis=1))
            x_0 = SYSTEM_PARAMS.meta.px_dist_adjacent_markers / 2
            frame_vein_endpoints = clip_vein_endpoints[j] 
            frame_vein_endpoints_mask = clip_vein_endpoints_mask[j]
            valid_veins = np.all(frame_vein_endpoints_mask, axis=1)
            
            for k in range(frame_vein_endpoints.shape[0]):
                if not valid_veins[k]:
                    continue
                vein_endpoints = frame_vein_endpoints[k]
                a, b, c = Contact.line_equation(
                    vein_endpoints
                )
                a2, b2, c2 = Contact.line_point_to_line_pass_through_point(
                    a, b, c, centre
                )

                for i in range(len(points)):
                    vec_vein_trajectory = Contact.vector_point_to_line(a2, b2, c2, points[i])
                    dist_from_vein_trajectory = np.linalg.norm(vec_vein_trajectory)
                    centre_ratio = max(0, 1 - decay_c * (dist_from_vein_trajectory / max_dist))
                    vec = Contact.vector_point_to_line(a, b, c, points[i])
                    x = np.linalg.norm(vec)
                    displacement = 0.0
                    if 0 < x < x_0:
                        displacement = x
                    elif x_0 <= x < 2 * x_0:
                        displacement = x_0 - (x - x_0)
                    if displacement > 0:
                        vec_normalized = vec / x
                        points[i] = points[i] + vec_normalized * displacement * disp_c * centre_ratio

            clip_points[j] = points

        return clip_points, vein_visible_mask
    