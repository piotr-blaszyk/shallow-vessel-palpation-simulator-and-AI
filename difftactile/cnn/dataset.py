from difftactile.main.constants import *
if SYSTEM_PARAMS.meta.cnn_gnn == 0:
    import cv2

import os
import torch
import numpy as np
import random
import math
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import time
from tqdm import tqdm

from difftactile.main.synthetic_image_generator import *
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.data_analysis.experiment.adjacency import *
import re


class MyDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            scheme, 
            mode='root', 
            data_points_today=[], 
            data_points_yesterday=[],
            data_points_pos=[],
            data_points_neg=[],
        ):
        super().__init__()
        start_time = time.perf_counter()
        self.scheme = scheme
        self.clip_len = SYSTEM_PARAMS.cnn.clip_len
        self.data_points_per_trajectory = 1024
        self.mode = mode

        today_data_dir = SYSTEM_PARAMS.files.dataset_root_today_reordered
        yesterday_data_dir = SYSTEM_PARAMS.files.dataset_root_yesterday_reordered_smoothed
        self.files_today = MyDataset.get_folder_files(today_data_dir)
        self.files_yesterday = MyDataset.get_folder_files(yesterday_data_dir)

        new_data_dir = SYSTEM_PARAMS.files.dataset_root_2025_08_21_reordered
        self.new_files = MyDataset.get_folder_files(new_data_dir)
        
        self.w_camera_big = int(SYSTEM_PARAMS.fisheye_model.target_image_width)
        self.h_camera_big = int(SYSTEM_PARAMS.fisheye_model.target_image_height)
        self.k = SYSTEM_PARAMS.fisheye_model.down_scaling_factor

        self.min_disp_c_range = [1.0, 0.1]
        self.max_disp_c_range = [1.0, 0.4]
        self.min_vein_length_range = [1.0, 0.2]
        self.max_vein_length_range = [1.0, 0.5]
        self.min_disp_c = None
        self.max_disp_c = None
        self.min_vein_length = None
        self.max_vein_length = None

        self.randomly_remove_k = [0, 6, 13]
        self.cnn_gnn = SYSTEM_PARAMS.meta.cnn_gnn
        self.visualisation_mode = False
        self.data_points_per_epoch = SYSTEM_PARAMS.dataset.data_points_per_epoch
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        self.adjacency_matrix = base_graph_data['adjacency_matrix']
        self.warmup = True
        self.difficulty_fyi = None
        self.set_difficulty_level(0.0)

        self.data_points_today = data_points_today
        self.data_points_yesterday = data_points_yesterday
        self.data_points_pos = data_points_pos
        self.data_points_neg = data_points_neg
        self.data_points = []
        if mode == 'root':
            if scheme == 'old':
                self.populate_clips_old_scheme()
            elif scheme == 'new':
                self.populate_clips_new_scheme()
        elif mode != 'dummy':
            if scheme == 'old':
                self.compute_data_points_old_scheme()
            elif scheme == 'new':
                self.compute_data_points_new_scheme()
        end_time = time.perf_counter()
        if mode == 'root':
            print(f"Time taken to initialise dataset: {end_time - start_time:.2f} seconds")
            print(f"num data points: {len(self.data_points):,}")
    
    @staticmethod
    def get_folder_files(path):
        return sorted([os.path.join(path, f) for f in os.listdir(path)])

    def compute_data_points_old_scheme(self):
        # Shuffle both datasets independently
        random.shuffle(self.data_points_today)
        random.shuffle(self.data_points_yesterday)
        
        # Calculate how many elements we can take while maintaining 2:1 ratio
        # For every 3 elements, we need 2 from today and 1 from yesterday
        num_triplets = min(len(self.data_points_today) // 2, len(self.data_points_yesterday))
        
        # Select the elements
        today_elements = self.data_points_today[:num_triplets * 2]
        yesterday_elements = self.data_points_yesterday[:num_triplets]
        
        # Combine the elements
        self.data_points = []
        for i in range(num_triplets):
            # Add 2 elements from today
            self.data_points.append(today_elements[i * 2])
            self.data_points.append(today_elements[i * 2 + 1])
            # Add 1 element from yesterday
            self.data_points.append(yesterday_elements[i])
        
        # Final shuffle to mix today and yesterday elements
        random.shuffle(self.data_points)
    
    def compute_data_points_new_scheme(self):
        # Shuffle both datasets independently
        random.shuffle(self.data_points_pos)
        random.shuffle(self.data_points_neg)
        
        # Calculate how many elements we can take while maintaining 2:1 ratio
        # For every 3 elements, we need 2 from pos and 1 from neg
        num_triplets = min(len(self.data_points_pos), len(self.data_points_neg))
        
        # Select the elements
        pos_elements = self.data_points_pos[:num_triplets]
        neg_elements = self.data_points_neg[:num_triplets]
        
        # Combine the elements
        self.data_points = []
        for i in range(num_triplets):
            # Add 1 element from pos
            self.data_points.append(pos_elements[i])
            # Add 1 element from neg
            self.data_points.append(neg_elements[i])
        
        # Final shuffle to mix pos and neg elements
        random.shuffle(self.data_points)

    def set_stats(
            self,
            stats
    ):
        self.warmup = False
        bad_keys = {'alpha_pos', 'alpha_neg'}
        for key, value in stats.items():
            if key not in bad_keys:
                setattr(self, key, value)
        foo = 7

    def populate_clips_old_scheme(self):
        return
        self.file_today_vein_masks = []
        self.file_today_contains_vein = []
        for i in range(len(self.files_today)):
            self.file_today_vein_masks.append(
                self.video_contains_vein(self.files_today[i])
            )
            self.file_today_contains_vein.append(
                self.compute_file_contains_vein(self.files_today[i])
            )

        dilations = [1, 2, 4]
        
        for i in range(len(self.files_today)):
            file_path = self.files_today[i]
            data = np.load(file_path)
            points = data['markers']
            total_frames = points.shape[0]
            targets = data['target_id_array']
            targets = targets[:, 0]
            valid_frames = (targets >= 3) & np.isin((targets - 2) % 3, [1, 2])

            displacement_vectors = np.diff(points, axis=0)
            displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=2)
            mean_displacement = np.mean(displacement_magnitudes, axis=1)
            padded_displacement = np.pad(mean_displacement, (1, 1), mode='edge')
            kernel = np.ones(2) / 2.0
            smoothed_displacement = np.convolve(padded_displacement, kernel, mode='valid')
            threshold = np.percentile(smoothed_displacement, 10)
            static_frames = smoothed_displacement < threshold
            
            valid_frames &= static_frames

            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    clips_found = 0
                    max_attempts = num_possible_starts * 2
                    attempts = 0
                    while clips_found < self.data_points_per_trajectory and attempts < max_attempts:
                        start_idx = random.randrange(num_possible_starts)
                        if (
                            valid_frames[start_idx:start_idx + dilated_clip_len:dilation].all()
                            and self.clip_contains_vein(i, start_idx, dilation)
                        ):
                            self.data_points_today.append(('today', 'grid_search', file_path, start_idx, dilation))
                            clips_found += 1
                        attempts += 1
        
        for i in range(len(self.files_yesterday)):
            file_path = self.files_yesterday[i]
            file_num = MyDataset.extract_trajectory_number(file_path)
            if file_num % 2 == 0:
                traj_type = 'grid_search'
            else:
                traj_type = 'random'
            data = np.load(file_path)
            points = data['markers']
            total_frames = points.shape[0]
            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    start_indices = sorted(random.sample(
                        range(num_possible_starts), 
                        min(self.data_points_per_trajectory, num_possible_starts)
                    ))
                    for start_idx in start_indices:
                        self.data_points_yesterday.append(('yesterday', traj_type, file_path, start_idx, dilation))

        print("clips have now been populated!")

    def populate_clips_new_scheme(self):
        self.vein_masks_new_scheme = []
        for i in range(len(self.new_files)):
            self.vein_masks_new_scheme.append(
                self.video_contains_vein(self.new_files[i])
            )

        dilations = [1, 2, 4]
        
        for i in range(len(self.new_files)):
            file_path = self.new_files[i]
            data = np.load(file_path)
            points = data['markers']
            total_frames = points.shape[0]
            targets = data['target_id_array']
            targets = targets[:, 0]
            valid_frames = (targets >= 3) & np.isin((targets - 2) % 3, [1, 2])

            # displacement_vectors = np.diff(points, axis=0)
            # displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=2)
            # mean_displacement = np.mean(displacement_magnitudes, axis=1)
            # padded_displacement = np.pad(mean_displacement, (1, 1), mode='edge')
            # kernel = np.ones(2) / 2.0
            # smoothed_displacement = np.convolve(padded_displacement, kernel, mode='valid')
            # threshold = np.percentile(smoothed_displacement, 10)
            # static_frames = smoothed_displacement < threshold
            
            # valid_frames &= static_frames

            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    clips_found = 0
                    max_attempts = num_possible_starts * 2
                    attempts = 0
                    while clips_found < self.data_points_per_trajectory and attempts < max_attempts:
                        start_idx = random.randrange(num_possible_starts)
                        if (
                            valid_frames[start_idx:start_idx + dilated_clip_len:dilation].all()
                            and self.clip_contains_vein(i, start_idx, dilation)
                        ):
                            self.data_points_pos.append(('pos', file_path, start_idx, dilation))
                            clips_found += 1
                        attempts += 1
        
        dilations = [4, 8, 16, 32]

        for i in range(len(self.new_files)):
            file_path = self.new_files[i]
            file_num = MyDataset.extract_trajectory_number(file_path)
            if file_num % 2 == 0:
                traj_type = 'grid_search'
            else:
                traj_type = 'random'
            if traj_type == 'random':
                continue
            data = np.load(file_path)
            points = data['markers']
            total_frames = points.shape[0]
            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    start_indices = sorted(random.sample(
                        range(num_possible_starts), 
                        min(self.data_points_per_trajectory, num_possible_starts)
                    ))
                    for start_idx in start_indices:
                        self.data_points_neg.append(('neg', file_path, start_idx, dilation))

        print("clips have now been populated!")

    def __len__(self):
        return len(self.data_points)

    @staticmethod
    def interpolate(a, b, b_weight):
        return a * (1 - b_weight) + b * b_weight

    def set_difficulty_level(self, difficulty):
        # print(f'new difficulty: {difficulty}')
        self.min_disp_c = MyDataset.interpolate(
            *self.min_disp_c_range,
            difficulty
        )
        self.min_vein_length = MyDataset.interpolate(
            *self.min_vein_length_range,
            difficulty
        )
        self.max_disp_c = MyDataset.interpolate(
            *self.max_disp_c_range,
            difficulty
        )
        self.max_vein_length = MyDataset.interpolate(
            *self.max_vein_length_range,
            difficulty
        )
        self.difficulty_fyi = difficulty

    def create_splits(self, *args, **kwargs):
        if self.scheme == 'old':
            return self.create_splits_old_scheme(*args, **kwargs)
        elif self.scheme == 'new':
            return self.create_splits_new_scheme(*args, **kwargs)
    
    def create_splits_old_scheme(
        self, train_size, val_size, test_size
    ):
        """Split dataset while ensuring all clips from the same trajectory stay together"""
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, (
            "Split proportions must sum to 1"
        )

        all_data_points = [
            self.data_points_today,
            self.data_points_yesterday,
        ]
        all_indices = []
        for i in range(len(all_data_points)):
            data_points = all_data_points[i]

            trajectory_to_indices = {}
            for i, data_point in enumerate(data_points):
                file_path = data_point[2]
                trajectory_to_indices.setdefault(file_path, []).append(i)
            trajectories = list(trajectory_to_indices.keys())
            trajectories.sort()
            
            n = len(trajectories)
            train_split = int(n * train_size)
            val_split = int(n * (train_size + val_size))
            train_trajectories = trajectories[:train_split]
            val_trajectories = trajectories[train_split:val_split]
            test_trajectories = trajectories[val_split:]
            
            train_indices = [i for traj in train_trajectories for i in trajectory_to_indices[traj]]
            val_indices = [i for traj in val_trajectories for i in trajectory_to_indices[traj]]
            test_indices = [i for traj in test_trajectories for i in trajectory_to_indices[traj]]
            all_indices.append({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
            })
        
        today = all_indices[0]
        yesterday = all_indices[1]
        train_today = [self.data_points_today[i] for i in today['train_indices']]
        train_yesterday = [self.data_points_yesterday[i] for i in yesterday['train_indices']]
        val_today = [self.data_points_today[i] for i in today['val_indices']]
        val_yesterday = [self.data_points_yesterday[i] for i in yesterday['val_indices']]
        test_today = [self.data_points_today[i] for i in today['test_indices']]
        test_yesterday = [self.data_points_yesterday[i] for i in yesterday['test_indices']]

        train_dataset = MyDataset(
            scheme=self.scheme,
            mode='train',
            data_points_today=train_today,
            data_points_yesterday=train_yesterday
        )
        val_dataset = MyDataset(
            scheme=self.scheme,
            mode='val',
            data_points_today=val_today,
            data_points_yesterday=val_yesterday
        )
        test_dataset = MyDataset(
            scheme=self.scheme,
            mode='test',
            data_points_today=test_today,
            data_points_yesterday=test_yesterday
        )
        
        res = (train_dataset, val_dataset, test_dataset)
        print(f'split len: {[len(x) for x in res]}')
        return res
    
    def create_splits_new_scheme(
        self, train_size, val_size, test_size
    ):
        """Split dataset while ensuring all clips from the same trajectory stay together"""
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, (
            "Split proportions must sum to 1"
        )

        all_data_points = [
            self.data_points_pos,
            self.data_points_neg,
        ]
        all_indices = []
        for i in range(len(all_data_points)):
            data_points = all_data_points[i]

            trajectory_to_indices = {}
            for i, data_point in enumerate(data_points):
                file_path = data_point[2]
                trajectory_to_indices.setdefault(file_path, []).append(i)
            trajectories = list(trajectory_to_indices.keys())
            trajectories.sort()
            
            n = len(trajectories)
            train_split = int(n * train_size)
            val_split = int(n * (train_size + val_size))
            train_trajectories = trajectories[:train_split]
            val_trajectories = trajectories[train_split:val_split]
            test_trajectories = trajectories[val_split:]
            
            train_indices = [i for traj in train_trajectories for i in trajectory_to_indices[traj]]
            val_indices = [i for traj in val_trajectories for i in trajectory_to_indices[traj]]
            test_indices = [i for traj in test_trajectories for i in trajectory_to_indices[traj]]
            all_indices.append({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
            })
        
        pos = all_indices[0]
        neg = all_indices[1]
        train_pos = [self.data_points_pos[i] for i in pos['train_indices']]
        train_neg = [self.data_points_neg[i] for i in neg['train_indices']]
        val_pos = [self.data_points_pos[i] for i in pos['val_indices']]
        val_neg = [self.data_points_neg[i] for i in neg['val_indices']]
        test_pos = [self.data_points_pos[i] for i in pos['test_indices']]
        test_neg = [self.data_points_neg[i] for i in neg['test_indices']]

        train_dataset = MyDataset(
            scheme=self.scheme,
            mode='train',
            data_points_pos=train_pos,
            data_points_neg=train_neg
        )
        val_dataset = MyDataset(
            scheme=self.scheme,
            mode='val',
            data_points_pos=val_pos,
            data_points_neg=val_neg
        )
        test_dataset = MyDataset(
            scheme=self.scheme,
            mode='test',
            data_points_pos=test_pos,
            data_points_neg=test_neg
        )
        
        res = (train_dataset, val_dataset, test_dataset)
        print(f'split len: {[len(x) for x in res]}')
        return res

    def compute_file_contains_vein(self, file_path):
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        
        # Extract the trajectory number from the file name
        # Format is "trajectory_XXXX.npz" where XXXX is a 4-digit number
        file_num = int(file_name.split('_')[1][:4])
        
        return (file_num % 4) < 2

    def clip_contains_vein(self, file_ix, start, dilation):
        if self.scheme == 'old':
            markers, veins, contains_vein = self.file_today_vein_masks[file_ix]
        elif self.scheme == 'new':
            markers, veins, contains_vein = self.vein_masks_new_scheme[file_ix]
        dilated_clip_len = self.clip_len * dilation
        
        # Get the clip masks and vein data
        clip_vein_mask = contains_vein[start : start+dilated_clip_len : dilation]  # shape: (clip_len, num_veins)
        clip_veins = veins[start : start+dilated_clip_len : dilation]  # shape: (clip_len, num_veins, num_points, 2)
        
        # Find veins present in all frames
        veins_present = np.all(clip_vein_mask, axis=0)  # shape: (num_veins,)
        
        if not np.any(veins_present):
            return False
            
        # For veins present in all frames, compute displacement
        first_frame = clip_veins[0]    # shape: (num_veins, num_points, 2)
        last_frame = clip_veins[-1]    # shape: (num_veins, num_points, 2)
        
        # Compute displacement magnitude for each point in each vein
        displacement = np.linalg.norm(last_frame - first_frame, axis=2)  # shape: (num_veins, num_points)
        
        # Compute mean displacement for each vein
        mean_displacement = np.mean(displacement, axis=1)  # shape: (num_veins,)
        
        # Check which veins meet the displacement threshold
        meets_displacement = mean_displacement >= 168
        
        # A vein must be both present throughout and meet displacement threshold
        valid_veins = veins_present & meets_displacement
        
        return np.any(valid_veins)

    def video_contains_vein(self, file_path):
        data = np.load(file_path)
        markers = data['markers']  # shape: (num_video_frames, num_markers, 2)
        labels = data['vein_polyline']  # shape: (num_video_frames, num_veins, num_points, 2)
        labels_mask = data['vein_polyline_mask']  # shape: (num_video_frames, num_veins, num_points)
        
        # Compute mean position for each frame
        mean_positions = np.mean(markers, axis=1)  # shape: (num_video_frames, 2)
        r = SYSTEM_PARAMS.fisheye_model.circle_radius / 3

        # Reshape mean_positions to broadcast against labels
        # (num_video_frames, 1, 1, 2) to broadcast with (num_video_frames, num_veins, num_points, 2)
        centres = mean_positions[:, np.newaxis, np.newaxis, :]
        
        # Extract x and y coordinates
        cx = centres[..., 0]
        cy = centres[..., 1]
        x = labels[..., 0]
        y = labels[..., 1]
        
        # Check circle condition for all points
        in_circle = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2  # shape: (num_video_frames, num_veins, num_points)
        
        labels_mask &= in_circle
        
        # Reduce to (num_video_frames, num_veins) by checking if any point in each vein is True
        contains_vein = np.any(labels_mask, axis=2)  # shape: (num_video_frames, num_veins)
        
        return (markers, labels, contains_vein)

    def frame_contains_vein(self, data, ix):
        markers = data['markers'][ix]
        markers_mask = data['markers_mask'][ix]
        labels = data['vein_polyline'][ix]
        labels_mask = data['vein_polyline_mask'][ix]

        valid_markers = markers[markers_mask]
        centre = np.mean(valid_markers, axis=0)
        cx = centre[0]
        cy = centre[1]
        r = SYSTEM_PARAMS.fisheye_model.circle_radius / 3

        mask_circle = SyntheticImageGenerator.filter_points_vectorised(cx, cy, r, labels)
        labels_mask &= mask_circle

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
        if self.scheme == 'old':
            day, traj_type, file_path, frame_ix, dilation = self.data_points[idx]
            if day == 'today':
                rng = random.uniform(0, 1)
                spawn_vein = rng < 0.75
            else:
                spawn_vein = False
        elif self.scheme == 'new':
            pos_neg_str, file_path, frame_ix, dilation = self.data_points[idx]
            spawn_vein = pos_neg_str == 'pos'
        
        data = np.load(file_path)
        markers = data['markers']
        markers_mask = data['markers_mask']
        veins = data['vein_polyline']
        veins_mask = data['vein_polyline_mask']
        if not spawn_vein:
            veins_mask = np.zeros_like(veins_mask, dtype=bool)
        
        dilated_clip_len = self.clip_len * dilation
        markers = markers[frame_ix:frame_ix + dilated_clip_len:dilation]
        markers_mask = markers_mask[frame_ix:frame_ix + dilated_clip_len:dilation]
        veins = veins[frame_ix:frame_ix + dilated_clip_len:dilation]
        veins_mask = veins_mask[frame_ix:frame_ix + dilated_clip_len:dilation]
        
        # Find veins that are present in all frames
        # A vein is "present" if it has at least one valid point in a frame
        vein_present_per_frame = np.any(veins_mask, axis=2)  # shape: (clip_len, num_veins)
        vein_present_all_frames = np.all(vein_present_per_frame, axis=0)  # shape: (num_veins,)
        
        # If multiple veins are present in all frames, randomly select one
        if np.sum(vein_present_all_frames) > 1:
            present_vein_indices = np.where(vein_present_all_frames)[0]
            selected_vein_idx = np.random.choice(present_vein_indices)
            vein_present_all_frames[:] = False
            vein_present_all_frames[selected_vein_idx] = True
        
        # Mask out veins that aren't present in all frames
        veins_mask &= vein_present_all_frames[np.newaxis, :, np.newaxis]
        
        if spawn_vein:
            freeze_markers = True
        else:
            freeze_markers = random.uniform(0, 1) < 0.2

        if freeze_markers:
            # Compute mean marker displacement relative to first frame
            first_frame_markers = markers[0]  # shape: (num_markers, 2)
            # Compute displacement for each marker relative to first frame
            marker_displacements = markers - first_frame_markers[np.newaxis, :, :]  # shape: (num_frames, num_markers, 2)
            # Compute mean displacement across all markers for each frame
            mean_marker_displacement = np.mean(marker_displacements, axis=1)  # shape: (num_frames, 2)
            
            # Set all marker positions to their positions in first frame
            markers = np.tile(first_frame_markers[np.newaxis, :, :], (markers.shape[0], 1, 1))
            
            # Adjust vein positions by subtracting mean marker displacement
            veins -= mean_marker_displacement[:, np.newaxis, np.newaxis, :]  # Broadcasting to (num_frames, num_veins, num_points, 2)

        markers, labels_signal_mask = self.augmentation_artificial_vein_signal_vectorised(
            markers, veins, veins_mask
        )
        veins_mask &= labels_signal_mask
        discrete_angles = [0, 60, 120, 180, 240, 300]
        rotation_angle_deg = random.choice(discrete_angles)
        markers = self.augmentation_rotation(markers, rotation_angle_deg)
        veins = self.augmentation_rotation(veins, rotation_angle_deg)
        angle_uniform_shift_rad = random.uniform(0, 2 * math.pi)
        magnitude = random.uniform(0, 10)
        markers = self.uniform_shift(markers, angle_uniform_shift_rad, magnitude)
        veins = self.uniform_shift(veins, angle_uniform_shift_rad, magnitude)
        angle_x = math.radians(random.uniform(-5, 5))
        angle_y = math.radians(random.uniform(-5, 5))
        markers = self.rotate_xy(markers, angle_x, angle_y)
        veins = self.rotate_xy(veins, angle_x, angle_y)

        points = markers
        points_mask = markers_mask

        pyg = self.generate_pyg_vectorised(points, veins, veins_mask)
        if self.visualisation_mode:
            veins = MyDataset.generate_vein_image(
                self.h_camera_big, 
                self.w_camera_big, 
                veins,
                veins_mask
            )
            veins = torch.tensor(veins, dtype=torch.float32) / 255.0
        else:
            veins = torch.empty(0)
        return pyg, veins
    
    def eval(self):
        self.visualisation_mode = True
    
    def get_points(self, idx):
        day, traj_type, file_path, frame_ix, dilation = self.data_points[idx]
        data = np.load(file_path)
        points = data['markers'][start_ix]
        return points

    @staticmethod
    def normalise_gnn_points(points):
        principal_point = np.array([
            SYSTEM_PARAMS.fisheye_model.principal_point_x,
            SYSTEM_PARAMS.fisheye_model.principal_point_y,
        ], dtype=float)
        r = SYSTEM_PARAMS.fisheye_model.circle_radius
        points -= principal_point
        points /= r
        return points
    
    @staticmethod
    def unnormalise_gnn_points(points):
        principal_point = np.array([
            SYSTEM_PARAMS.fisheye_model.principal_point_x,
            SYSTEM_PARAMS.fisheye_model.principal_point_y,
        ], dtype=float)
        r = SYSTEM_PARAMS.fisheye_model.circle_radius
        points *= r
        points += principal_point
        return points

    def generate_pyg_vectorised(self, clip_points, clip_labels, clip_labels_mask):
        num_frames = clip_points.shape[0]
        num_nodes = clip_points.shape[1]
        num_edge_features = SYSTEM_PARAMS.gnn.num_edge_features
        num_node_features = SYSTEM_PARAMS.gnn.num_node_features
        pos = np.zeros(shape=(num_frames * num_nodes, 2), dtype=float)
        data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        base_adjacency_matrix = data['adjacency_matrix']
        num_edges_single_frame = base_adjacency_matrix.shape[0]
        ground_truth_labels_clip = np.zeros(shape=(num_frames * num_nodes,), dtype=int)
        spatial = 0
        temporal = 1
        adjacency = self.adjacency_matrix
        frame_displacements = clip_points[1:] - clip_points[:-1]
        mean_frame_displacements = np.mean(frame_displacements, axis=1)

        # Pre-compute edge distances for all frames to calculate variance
        edge_distances = np.zeros((num_frames, num_edges_single_frame))
        for t in range(num_frames):
            points = clip_points[t]
            me_points = points[adjacency[:, 0]]
            neighbor_points = points[adjacency[:, 1]]
            vec = neighbor_points - me_points
            edge_distances[t] = np.linalg.norm(vec, axis=1)

        # Calculate edge variance for spatial edges
        edge_var = np.var(edge_distances, axis=0)

        adjacency_clip_list = []
        edge_attr_clip_list = []

        for t in range(num_frames):
            points = clip_points[t]
            labels = clip_labels[t]
            labels_mask = clip_labels_mask[t]
            labels_filtered = labels[labels_mask]
            pos[t*num_nodes:(t+1)*num_nodes, 0:2] = points
            
            edge_attr = np.zeros(shape=(num_edges_single_frame, num_edge_features), dtype=float)
            me_points = points[adjacency[:, 0]]
            neighbor_points = points[adjacency[:, 1]]
            vec = neighbor_points - me_points
            dist = np.linalg.norm(vec, axis=1)
            edge_attr[:, 0] = dist
            edge_attr[:, 1] = spatial
            edge_attr[:, 2] = edge_var  # Add edge variance feature
            edge_attr_clip_list.append(edge_attr)
            adjacency_frame = adjacency + t*num_nodes
            adjacency_clip_list.append(adjacency_frame)
            
            if labels_filtered.size > 0:
                distances = cdist(points, labels_filtered)
                min_distances = np.min(distances, axis=1)
                px_threshold = SYSTEM_PARAMS.meta.vein_px_thickness
                ground_truth_labels = min_distances < px_threshold
            else:
                ground_truth_labels = np.zeros(shape=(num_nodes,), dtype=int)
            ground_truth_labels_clip[t*num_nodes:(t+1)*num_nodes] = ground_truth_labels
        
        adjacency_clip_temporal_list = []
        edge_attr_clip_temporal_list = []
        for t in range(num_frames-1):
            for direction in range(2):  # 0: forward, 1: backward
                if direction == 0:  # forward edges (t → t+1)
                    src = t*num_nodes + np.arange(num_nodes)
                    dst = (t+1)*num_nodes + np.arange(num_nodes)
                else:  # backward edges (t+1 → t)
                    src = (t+1)*num_nodes + np.arange(num_nodes)
                    dst = t*num_nodes + np.arange(num_nodes)
                
                vec = pos[dst, 0:2] - pos[src, 0:2]
                if direction == 0:
                    vec = vec - mean_frame_displacements[t]
                else:
                    vec = vec + mean_frame_displacements[t]
                
                dist = np.linalg.norm(vec, axis=1)
                
                # Add temporal edges with dummy value (0) for edge_var
                edge_attr_temporal = np.zeros((len(dist), num_edge_features))
                edge_attr_temporal[:, 0] = dist
                edge_attr_temporal[:, 1] = temporal
                edge_attr_temporal[:, 2] = 0  # Dummy value for edge_var
                
                adjacency_clip_temporal_list.append(np.column_stack([src, dst]))
                edge_attr_clip_temporal_list.append(edge_attr_temporal)
            
        if len(adjacency_clip_temporal_list) > 0:
            adjacency_clip_temporal = np.vstack(adjacency_clip_temporal_list)
            edge_attr_clip_temporal = np.vstack(edge_attr_clip_temporal_list)
            adjacency_clip_list.append(adjacency_clip_temporal)
            edge_attr_clip_list.append(edge_attr_clip_temporal)

        adjacency_clip = np.vstack(adjacency_clip_list)
        edge_attr_clip = np.vstack(edge_attr_clip_list)

        # Calculate local_var and global_var for each node
        spatial_edges_mask = edge_attr_clip[:, 1] == spatial
        spatial_edge_vars = edge_attr_clip[spatial_edges_mask, 2]
        global_var = np.mean(spatial_edge_vars)

        # Calculate individual node displacements and mean displacements
        node_displacements = clip_points[1:] - clip_points[:-1]  # Shape: (num_frames-1, num_nodes, 2)
        mean_displacements = np.mean(node_displacements, axis=1, keepdims=True)  # Shape: (num_frames-1, 1, 2)
        
        # Compute difference between individual and mean displacements
        displacement_diff = node_displacements - mean_displacements  # Shape: (num_frames-1, num_nodes, 2)
        
        # Calculate magnitude of displacement differences
        displacement_magnitudes = np.linalg.norm(displacement_diff, axis=2)  # Shape: (num_frames-1, num_nodes)
        # Store the relative displacement vectors for each node and frame
        relative_displacement_vectors = np.zeros((num_frames, num_nodes, 2))
        relative_displacement_vectors[1:] = -displacement_diff  # t-1 to t vectors
        # Use convolution to expand to full sequence
        kernel = np.array([0.5, 0.5])
        relative_displacement_magnitude = np.zeros((num_frames, num_nodes))
        # Reshape displacement_magnitudes to (num_nodes, num_frames-1) for convolution
        displacement_magnitudes_T = displacement_magnitudes.T
        # Apply convolution to all nodes at once using array operations
        relative_displacement_magnitude = np.array([np.convolve(node_magnitudes, kernel, mode='full')[:num_frames] 
                                                 for node_magnitudes in displacement_magnitudes_T]).T
        relative_displacement_magnitude[0, :] *= 2
        relative_displacement_magnitude[-1, :] *= 2

        # Initialize node features array with extra feature
        x_features = np.zeros((num_frames * num_nodes, num_node_features))

        # Initialize edge attributes for relative displacement messages
        edge_attr_rel_disp_message = np.zeros((adjacency_clip.shape[0], 1), dtype=float)
        
        # Calculate local_var for each node
        for t in range(num_frames):
            frame_offset = t * num_nodes
            frame_edges_mask = (adjacency_clip[:, 0] >= frame_offset) & (adjacency_clip[:, 0] < frame_offset + num_nodes) & spatial_edges_mask
            
            # Vectorized calculation of maximum edge_var for each node in the frame
            nodes_in_frame = np.arange(frame_offset, frame_offset + num_nodes)
            node_edge_vars = [edge_attr_clip[
                (adjacency_clip[:, 0] == node_idx) & frame_edges_mask, 2
            ] for node_idx in nodes_in_frame]
            max_vars = np.array([np.max(vars) if len(vars) > 0 else 0 for vars in node_edge_vars])
            x_features[frame_offset:frame_offset + num_nodes, 0] = max_vars
            
            # Set global_var for all nodes in the frame
            x_features[frame_offset:frame_offset + num_nodes, 1] = global_var
            
            # Set relative_displacement_magnitude for all nodes in the frame
            x_features[frame_offset:frame_offset + num_nodes, 2] = relative_displacement_magnitude[t]

            # Compute edge vectors and relative displacement messages for this frame
            frame_edges = adjacency_clip[frame_edges_mask]
            frame_edge_attr = edge_attr_clip[frame_edges_mask]
            
            # Get source nodes for edges in this frame
            src_nodes = frame_edges[:, 0] - frame_offset
            
            # Compute edge vectors using positions
            edge_vectors = pos[frame_edges[:, 1]] - pos[frame_edges[:, 0]]  # Shape: (num_frame_edges, 2)
            edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
            edge_directions = edge_vectors / (edge_lengths + 1e-10)  # Normalized edge vectors
            
            # Get displacement vectors for source nodes
            src_displacements = relative_displacement_vectors[t, src_nodes]  # Shape: (num_frame_edges, 2)
            
            # Project displacement vectors onto edge directions
            # projection = dot(displacement, edge_direction) * edge_direction
            # We only need the signed magnitude, which is just the dot product
            projection_magnitudes = np.sum(src_displacements * edge_directions, axis=1)
            
            # Store the projection magnitudes in the separate array
            edge_attr_rel_disp_message[frame_edges_mask, 0] = projection_magnitudes
        
        # Calculate total displacement magnitude for each node across all frames
        node_total_displacement = np.zeros(num_nodes)
        for t in range(1, num_frames):
            displacement_vectors = clip_points[t] - clip_points[t-1]  # shape: (num_nodes, 2)
            displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)  # shape: (num_nodes,)
            node_total_displacement += displacement_magnitudes
        node_total_displacement_sum = node_total_displacement.sum()
        node_total_displacement_mean = node_total_displacement_sum / (num_nodes * (num_frames-1))

        x_features[:, 3] = node_total_displacement_mean
        edge_attr_clip = np.concatenate([edge_attr_clip, edge_attr_rel_disp_message], axis=1)
        edge_attr_clip_mask = np.array([True, True, False, True], dtype=bool)
        edge_attr_clip = edge_attr_clip[:, edge_attr_clip_mask]

        if not self.warmup:
            edge_attr_clip[:, 0] = (edge_attr_clip[:, 0] - self.edge_attr_mean[0]) / self.edge_attr_std[0]
            edge_attr_clip[:, 2] = (edge_attr_clip[:, 2] - self.edge_attr_mean[2]) / self.edge_attr_std[2]

            # Normalize node features
            x_features[:, 0] = (x_features[:, 0] - self.x_mean[0]) / self.x_std[0]  # local_var
            x_features[:, 1] = (x_features[:, 1] - self.x_mean[1]) / self.x_std[1]  # global_var
            x_features[:, 2] = (x_features[:, 2] - self.x_mean[2]) / self.x_std[2]  # relative_displacement_magnitude
            x_features[:, 3] = (x_features[:, 3] - self.x_mean[3]) / self.x_std[3]  # node_total_displacement_mean 

        # Add the total displacement as the last feature, repeating for each frame
        for t in range(num_frames):
            frame_offset = t * num_nodes
            x_features[frame_offset:frame_offset + num_nodes, 3] = node_total_displacement

        edge_attr = torch.tensor(edge_attr_clip, dtype=torch.float)
        x = torch.tensor(x_features, dtype=torch.float)
        mask = np.ones(shape=(num_frames * num_nodes,), dtype=bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        pos = torch.tensor(pos, dtype=torch.float)
        edge_index = torch.tensor(adjacency_clip.T, dtype=torch.long)
        y = torch.tensor(ground_truth_labels_clip[mask], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask, pos=pos)

    @staticmethod
    def generate_pyg_unvectorised(clip_points, clip_labels, clip_labels_mask):
        num_frames = clip_points.shape[0]
        central_frame = num_frames // 2
        num_nodes = clip_points.shape[1]
        num_edge_features = SYSTEM_PARAMS.gnn.num_edge_features
        num_node_features = SYSTEM_PARAMS.gnn.num_node_features
        x_clip = np.zeros(shape=(num_frames * num_nodes, num_node_features), dtype=float)
        data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        base_adjacency_matrix = data['adjacency_matrix']
        num_edges_single_frame = base_adjacency_matrix.shape[0]
        ground_truth_labels_clip = np.zeros(shape=(num_frames * num_nodes,), dtype=int)
        spatial = 0
        temporal = 1

        adjacency_clip_list = []
        edge_attr_clip_list = []

        for t in range(num_frames):
            points = clip_points[t]
            labels = clip_labels[t]
            labels_mask = clip_labels_mask[t]
            labels = labels[labels_mask]
            base_points, points, adjacency = Adjacency.get_graph_connectivity(points)
            x_clip[t*num_nodes:(t+1)*num_nodes, 0:2] = points
            x_clip[t*num_nodes:(t+1)*num_nodes, 2] = t - central_frame
            
            # edge_attr = np.zeros(shape=(num_edges_single_frame, num_edge_features), dtype=float)
            edge_attr = np.zeros(shape=(num_edges_single_frame, 2), dtype=float)
            for i in range(num_edges_single_frame):
                me_ix, neighbour_ix = adjacency[i, :]
                vec = points[neighbour_ix] - points[me_ix]
                dist = np.linalg.norm(vec)
                sin_theta = vec[1] / (dist + 1e-10)
                cos_theta = vec[0] / (dist + 1e-10)
                # edge_attr[i] = np.array([*vec, dist, sin_theta, cos_theta, spatial], dtype=float)
                edge_attr[i] = np.array([dist, spatial], dtype=float)
            edge_attr_clip_list.append(edge_attr)
            adjacency += t*num_nodes
            adjacency_clip_list.append(adjacency)
            
            points_unnormalised = MyDataset.unnormalise_gnn_points(points)
            distances = cdist(points_unnormalised, labels)
            min_distances = np.min(distances, axis=1)
            px_threshold = SYSTEM_PARAMS.meta.vein_px_thickness
            ground_truth_labels = min_distances < px_threshold
            ground_truth_labels_clip[t*num_nodes:(t+1)*num_nodes] = ground_truth_labels
        
        adjacency_clip_temporal_list = []
        edge_attr_clip_temporal_list = []
        for t in range(num_frames-1):
            for i in range(num_nodes):
                src = t*num_nodes+i
                dst = (t+1)*num_nodes+i
                vec = x_clip[dst, 0:2] - x_clip[src, 0:2]
                dist = np.linalg.norm(vec)
                sin_theta = vec[1] / (dist + 1e-10)
                cos_theta = vec[0] / (dist + 1e-10)
                adjacency_clip_temporal_list.append([src, dst])
                adjacency_clip_temporal_list.append([dst, src])
                # feature_vector = [*vec, dist, sin_theta, cos_theta, temporal]
                feature_vector = [dist, temporal]
                edge_attr_clip_temporal_list.append(feature_vector)
                edge_attr_clip_temporal_list.append(feature_vector)
                    
        if len(adjacency_clip_temporal_list) > 0:
            adjacency_clip_temporal = np.array(adjacency_clip_temporal_list)
            edge_attr_clip_temporal = np.array(edge_attr_clip_temporal_list)
            adjacency_clip_list.append(adjacency_clip_temporal)
            edge_attr_clip_list.append(edge_attr_clip_temporal)
        

        adjacency_clip = np.vstack(adjacency_clip_list)
        edge_attr_clip = np.vstack(edge_attr_clip_list)

        x_clip_const = np.zeros(shape=(num_frames * num_nodes, 1), dtype=float)
        x = torch.tensor(x_clip_const, dtype=torch.float)
        mask = np.ones(shape=(num_frames * num_nodes,), dtype=bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        pos = torch.tensor(x_clip, dtype=torch.float)
        edge_index = torch.tensor(adjacency_clip.T, dtype=torch.long)
        edge_attr_clip_normalised = (edge_attr_clip - np.mean(edge_attr_clip)) / np.std(edge_attr_clip)
        edge_attr = torch.tensor(edge_attr_clip_normalised, dtype=torch.float)
        y = torch.tensor(ground_truth_labels_clip[mask], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask, pos=pos)

    def get_markers(self, idx):
        day, traj_type, file_path, frame_ix, dilation = self.data_points[idx]

        data = np.load(file_path)
        # np.load returns a dict-like object whose keys we can access directly
        markers = data['markers']
        markers_mask = data['markers_mask']
        labels = data['vein_polyline']
        labels_mask = data['vein_polyline_mask']
        
        # Extract longer sequence and apply dilation
        dilated_clip_len = self.clip_len * dilation
        images = markers[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        images_mask = markers_mask[start:start + dilated_clip_len:dilation]  # Take every dilation-th frame
        labels = labels[start:start + dilated_clip_len:dilation]
        labels_mask = labels_mask[start:start + dilated_clip_len:dilation]
        
        # images, labels_signal_mask = self.augmentation_artificial_vein_signal(
        #     images, labels, labels_mask
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
        
        images = MyDataset.downscale(self.k, images)
        labels = MyDataset.downscale(self.k, labels)

        return images[self.clip_len // 2, :, :]

    def get_clip(self, markers, markers_mask, clip_len, dilation, start_ix):
        points = markers
        points_mask = markers_mask

        dilated_clip_len = clip_len * dilation
        points = points[start_ix:start_ix + dilated_clip_len:dilation]
        points_mask = points_mask[start_ix:start_ix + dilated_clip_len:dilation]

        labels = np.zeros(shape=(
            points.shape[0],
            0,
            0,
            2
        ), dtype=int)
        labels_mask = np.zeros(shape=(
            points.shape[0],
            0,
            0
        ), dtype=bool)
        
        pyg = self.generate_pyg_vectorised(points, labels, labels_mask)
        return pyg

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
                    MyDataset.draw_point(images, t, point)
            markers_mask = SyntheticImageGenerator.compute_mask(h, w, filtered_points)
            marker_masks[t, :, :] = markers_mask
                
        return images, marker_masks

    @staticmethod
    def draw_point(images, t, point, n=8, intensity=(255, 255, 255)):
        """
        Draw a point as an nxn square with smooth interpolation at the edges.
        
        Args:
            images: Image array of shape (T, H, W) for grayscale or (T, H, W, 3) for BGR
            t: Time index
            point: (x, y) coordinates
            n: Size of the square (default=8)
            intensity: BGR intensity tuple (default=(255,255,255) for white)
                      Note: OpenCV uses BGR order, so (B,G,R) not (R,G,B)
        """
        if len(images.shape) == 3:  # Grayscale
            _, h, w = images.shape
            is_grayscale = True
        else:  # BGR
            _, h, w, _ = images.shape
            is_grayscale = False
        
        x, y = point[0], point[1]
        
        # Calculate the center of the nxn square
        center_x = x
        center_y = y
        
        # Calculate corners of the nxn square
        half_size = n / 2
        x0 = int(np.floor(center_x - half_size + 0.5))  # Left edge
        x_end = x0 + n                                   # Right edge
        y0 = int(np.floor(center_y - half_size + 0.5))  # Top edge
        y_end = y0 + n                                   # Bottom edge
        
        # Calculate weights for smooth interpolation
        for yi in range(y0, y_end):
            for xi in range(x0, x_end):
                if 0 <= xi < w and 0 <= yi < h:
                    # Calculate distance from point to center
                    dx = abs(xi + 0.5 - center_x)
                    dy = abs(yi + 0.5 - center_y)
                    
                    # Bilinear weight calculation (1 at center, 0 at edges)
                    wx = max(0, 1 - dx/(half_size))
                    wy = max(0, 1 - dy/(half_size))
                    weight = wx * wy
                    
                    if is_grayscale:
                        # For grayscale, use the average of BGR intensities
                        avg_intensity = sum(intensity) // 3
                        current_val = int(images[t, yi, xi])
                        weight_val = int(avg_intensity * weight)
                        images[t, yi, xi] = max(0, min(255, current_val + weight_val))
                    else:
                        # For BGR, apply weight to each channel in BGR order
                        for c in range(3):  # c=0 is Blue, c=1 is Green, c=2 is Red
                            current_val = int(images[t, yi, xi, c])
                            weight_val = int(intensity[c] * weight)  # intensity is already in BGR order
                            images[t, yi, xi, c] = max(0, min(255, current_val + weight_val))

    @staticmethod
    def generate_vein_image(h, w, points, points_mask):
        """
        Generate vein images by drawing polylines fitted to points.
        
        Args:
            h, w: Height and width of output image
            points: numpy array of shape (num_video_frames, num_points, 2)
            points_mask: numpy array of shape (num_video_frames, num_points)
            
        Returns:
            numpy array of shape (num_video_frames, h, w) containing binary images
        """
        if points.shape[-1] == 0:  # Check if there are any points
            return np.zeros((points.shape[0], h, w), dtype=np.uint8)

        # Initialize output array for all frames
        images = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        
        # Generate image for each frame
        for t in range(points.shape[0]):
            for v in range(points.shape[1]):
                # Get only valid points according to mask for this vein
                vein_points = points[t, v]
                vein_mask = points_mask[t, v]
                valid_points = vein_points[vein_mask]
                
                if len(valid_points) > 0:
                    # Convert points to integer coordinates for drawing
                    line_points = valid_points.astype(np.int32)
                    
                    # Draw lines connecting consecutive points
                    for i in range(len(line_points) - 1):
                        pt1 = tuple(line_points[i])
                        pt2 = tuple(line_points[i + 1])
                        cv2.line(images[t], pt1, pt2, color=255, thickness=SYSTEM_PARAMS.meta.vein_px_thickness)
                
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
        points_3d = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(points)
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
        rotated_points_2d = FisheyeModelNoTaichi.project_3d_2d_np(rotated_points_3d.reshape(-1, 3)).reshape(points.shape)
        
        return rotated_points_2d
    
    def randomly_remove(self, points):
        return
        if points.shape[-1] == 0:  # Check if there are any points
            return np.ones((points.shape[0], 0), dtype=bool)  # Return empty mask matching input shape
            
        # Get max number of points that can be removed
        max_k = self.randomly_remove_k[self.difficulty]
        
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
        
        origin = np.array([
            SYSTEM_PARAMS.fisheye_model.crop_x,
            SYSTEM_PARAMS.fisheye_model.crop_y
        ], dtype=float)
        resolution = np.array([
            SYSTEM_PARAMS.fisheye_model.crop_width,
            SYSTEM_PARAMS.fisheye_model.crop_height
        ], dtype=float)
        points, mask = SyntheticImageGenerator.crop(points, origin, resolution)
        points = points / k
        return points

    @staticmethod
    def make_3x3(points):
        origin = np.array([
            -SYSTEM_PARAMS.fisheye_model.crop_x,
            -SYSTEM_PARAMS.fisheye_model.crop_y
        ], dtype=float)
        resolution = np.array([
            3*SYSTEM_PARAMS.fisheye_model.crop_width,
            3*SYSTEM_PARAMS.fisheye_model.crop_height
        ], dtype=float)
        points, mask = SyntheticImageGenerator.crop(points, origin, resolution)
        return points, mask

    def augmentation_artificial_vein_signal_vectorised(self, clip_points, clip_vein_polyline, clip_vein_polyline_mask):
        valid_frames_mask = np.any(np.all(clip_vein_polyline_mask, axis=2), axis=1)
        num_veins = clip_vein_polyline.shape[1]
        num_vein_points = clip_vein_polyline.shape[2]
        if not np.any(valid_frames_mask):
            return clip_points, clip_vein_polyline_mask
        disp_c = random.uniform(self.min_disp_c, self.max_disp_c)

        target_num_vein_points = np.random.randint(
            int(self.min_vein_length*num_vein_points), 
            int(self.max_vein_length*num_vein_points) + 1, 
            size=num_veins
        )
        max_start_indices = num_vein_points - target_num_vein_points
        start_indices = np.random.randint(0, max_start_indices + 1)
        row_indices = np.arange(num_veins)[:, np.newaxis]
        col_indices = np.arange(num_vein_points)[np.newaxis, :]
        my_vein_mask = (col_indices >= start_indices[:, np.newaxis]) & \
                                (col_indices < (start_indices[:, np.newaxis] + target_num_vein_points[:, np.newaxis]))

        clip_vein_polyline_mask &= my_vein_mask[np.newaxis, :, :]

        for t in range(clip_points.shape[0]):
            points = clip_points[t]
            x_0 = SYSTEM_PARAMS.meta.px_dist_adjacent_markers / 2
            for v in range(clip_vein_polyline[t].shape[0]):
                tv_polyline = clip_vein_polyline[t, v, :, :][clip_vein_polyline_mask[t, v, :]]
                if tv_polyline.size == 0:
                    continue
                # Calculate vectors for all points at once
                vecs = SyntheticImageGenerator.vector_point_to_polynomial(tv_polyline, points)
                x = np.linalg.norm(vecs, axis=1)
                
                # Calculate displacements based on conditions
                displacement = np.zeros_like(x)
                mask1 = (0 < x) & (x < x_0)
                mask2 = (x_0 <= x) & (x < 2 * x_0)
                
                displacement[mask1] = x[mask1]
                displacement[mask2] = x_0 - (x[mask2] - x_0)
                
                # Apply displacement where needed
                mask = displacement > 0
                vec_normalized = vecs[mask] / x[mask, np.newaxis]
                points[mask] = points[mask] + vec_normalized * displacement[mask, np.newaxis] * disp_c

            clip_points[t] = points

        return clip_points, clip_vein_polyline_mask

    def augmentation_artificial_vein_signal_unvectorised(self, clip_points, clip_vein_polyline, clip_vein_polyline_mask):
        return
        valid_frames_mask = np.any(np.all(clip_vein_polyline_mask, axis=2), axis=1)
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
        if self.difficulty == 2:
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

        lower_disp_c = self.min_disp_c_range[self.difficulty]
        # disp_c = random.uniform(lower_disp_c, 0.5)
        disp_c = 1.0

        for j in range(start_idx, end_idx):
            points = clip_points[j]
            x_0 = SYSTEM_PARAMS.meta.px_dist_adjacent_markers / 2
            frame_vein_polyline = clip_vein_polyline[j] 
            frame_vein_polyline_mask = clip_vein_polyline_mask[j]
            valid_veins = np.all(frame_vein_polyline_mask, axis=1)
            
            for k in range(frame_vein_polyline.shape[0]):
                if not valid_veins[k]:
                    continue
                vein_polyline = frame_vein_polyline[k]

                for i in range(len(points)):
                    if self.difficulty == 2:
                        proceed = random.uniform(0, 1)
                    else:
                        proceed = 1.0
                    if proceed < 0.5:
                        continue
                    vec = SyntheticImageGenerator.vector_point_to_polynomial(vein_polyline, points[i])
                    x = np.linalg.norm(vec)
                    displacement = 0.0
                    if 0 < x < x_0:
                        displacement = x
                    elif x_0 <= x < 2 * x_0:
                        displacement = x_0 - (x - x_0)
                    if displacement > 0:
                        vec_normalized = vec / x
                        points[i] = points[i] + vec_normalized * displacement * disp_c

            clip_points[j] = points

        return clip_points, vein_visible_mask
    
    @staticmethod
    def extract_trajectory_number(file_path):
        """Extract the trajectory number from the file path.
        
        Args:
            file_path (str): Path like 'path/to/trajectory_0001.npz'
            
        Returns:
            int: The trajectory number (e.g., 1 for 'trajectory_0001.npz')
        """
        match = re.search(r'trajectory_(\d+)\.npz$', file_path)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract trajectory number from {file_path}")
    
    @staticmethod
    def extract_trajectory_number(file_path):
        """Extract the trajectory number from the file path.
        
        Args:
            file_path (str): Path like 'path/to/trajectory_0001.npz'
            
        Returns:
            int: The trajectory number (e.g., 1 for 'trajectory_0001.npz')
        """
        match = re.search(r'trajectory_(\d+)\.npz$', file_path)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract trajectory number from {file_path}")


class MyDatasetExpIterator:
    def __init__(
            self, 
            dataset: MyDataset, 
            npz_path
        ):
        self.dataset = dataset
        data = np.load(npz_path)
        markers = data['markers']
        markers_mask = data['markers_mask']
        if False and start_ix is not None and end_ix is not None:
            markers = markers[start_ix:end_ix]
            markers_mask = markers_mask[start_ix:end_ix]
        self.num_frames = markers.shape[0]
        print(f'num frames: {self.num_frames}')
        self.markers = markers
        self.markers_mask = markers_mask
        self.clip_len = SYSTEM_PARAMS.cnn.clip_len

    def __iter__(self):
        return self

    def __next__(self):
        # dilation = random.choice(self.dilations)
        # dilation = 10
        dilation = 2
        dilated_clip_len = self.clip_len * dilation
        min_start_ix = 0
        max_start_ix = self.num_frames - dilated_clip_len
        start_ix = random.randint(min_start_ix, max_start_ix)
        # start_ix = 155
        pyg = self.dataset.get_clip(
            self.markers,
            self.markers_mask,
            self.clip_len,
            dilation,
            start_ix
        )
        labels = torch.empty(0)
        return pyg, labels
