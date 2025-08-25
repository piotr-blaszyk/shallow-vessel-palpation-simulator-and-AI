from difftactile.main.constants import *

if SYSTEM_PARAMS.meta.cnn_gnn == 0:
    import cv2
import os
import torch
import numpy as np

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
        mode="root",
        data_points_today=[],
        data_points_yesterday=[],
        data_points_pos=[],
        data_points_neg=[],
        normalise_pos=True,
        exp_markers_npz=None,
        exp_ground_truth_labels_npz=None,
        exp_dilation=None,
    ):
        super().__init__()
        start_time = time.perf_counter()
        self.clip_len = SYSTEM_PARAMS.gnn.clip_len
        self.num_nodes = SYSTEM_PARAMS.vitactip.num_markers
        base_graph_connectivity_data = np.load(
            SYSTEM_PARAMS.files.base_graph_connectivity
        )
        self.adjacency = base_graph_connectivity_data["adjacency_matrix"]
        self.ring_ixs = base_graph_connectivity_data["ring_ixs"]
        self.angles = base_graph_connectivity_data["angles"]
        self.dist_from_centre = base_graph_connectivity_data["dist_from_centre"]
        self.num_edges_single_frame = self.adjacency.shape[0]
        self.num_entity_types = SYSTEM_PARAMS.gnn.num_entity_types
        self.entity_tag_one_hot = np.eye(self.num_entity_types, dtype=int)
        if mode == "exp":
            markers_data = np.load(exp_markers_npz)
            self.exp_markers = markers_data["markers"]
            ground_truth_label_data = np.load(exp_ground_truth_labels_npz)
            self.exp_ground_truth_labels = ground_truth_label_data["labels"]
            self.exp_dilation = exp_dilation
        self.scheme = scheme
        self.clip_len = SYSTEM_PARAMS.gnn.clip_len
        self.data_points_per_trajectory = 1024
        self.mode = mode
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
        self.warmup = True
        self.difficulty_fyi = None
        self.normalise_pos = normalise_pos
        self.set_difficulty_level(0.0)
        self.data_points_today = data_points_today
        self.data_points_yesterday = data_points_yesterday
        self.data_points_pos = data_points_pos
        self.data_points_neg = data_points_neg
        self.data_points = []
        if mode == "root":
            if scheme == "old":
                self.populate_clips_old_scheme()
            elif scheme == "new":
                self.populate_clips_new_scheme()
        elif mode == "exp":
            self.compute_data_points_exp()
        elif mode != "dummy":
            if scheme == "old":
                self.compute_data_points_old_scheme()
            elif scheme == "new":
                self.compute_data_points_new_scheme()
        end_time = time.perf_counter()
        if mode == "root":
            print(
                f"Time taken to initialise dataset: {end_time - start_time:.2f} seconds"
            )
            print(f"num data points: {len(self.data_points):,}")

    @staticmethod
    def get_folder_files(path):
        return sorted([os.path.join(path, f) for f in os.listdir(path)])

    def compute_data_points_exp(self):
        num_video_frames = self.exp_markers.shape[0]
        dilated_clip_len = self.clip_len * self.exp_dilation
        for i in range(num_video_frames - dilated_clip_len + 1):
            self.data_points.append(i)

    def compute_data_points_old_scheme(self):
        NP_RNG.shuffle(self.data_points_today)
        NP_RNG.shuffle(self.data_points_yesterday)
        num_triplets = min(
            len(self.data_points_today) // 2, len(self.data_points_yesterday)
        )
        today_elements = self.data_points_today[: num_triplets * 2]
        yesterday_elements = self.data_points_yesterday[:num_triplets]
        self.data_points = []
        for i in range(num_triplets):
            self.data_points.append(today_elements[i * 2])
            self.data_points.append(today_elements[i * 2 + 1])
            self.data_points.append(yesterday_elements[i])
        NP_RNG.shuffle(self.data_points)

    def compute_data_points_new_scheme(self):
        NP_RNG.shuffle(self.data_points_pos)
        NP_RNG.shuffle(self.data_points_neg)
        num_triplets = min(len(self.data_points_pos), len(self.data_points_neg))
        pos_elements = self.data_points_pos[:num_triplets]
        neg_elements = self.data_points_neg[:num_triplets]
        self.data_points = []
        for i in range(num_triplets):
            self.data_points.append(pos_elements[i])
            self.data_points.append(neg_elements[i])
        NP_RNG.shuffle(self.data_points)

    def set_stats(self, stats):
        self.warmup = False
        bad_keys = {"alpha_pos", "alpha_neg"}
        for key, value in stats.items():
            if key not in bad_keys:
                setattr(self, key, value)
        foo = 7

    def populate_clips_old_scheme(self):
        return
        today_data_dir = SYSTEM_PARAMS.files.dataset_root_today_reordered
        yesterday_data_dir = (
            SYSTEM_PARAMS.files.dataset_root_yesterday_reordered_smoothed
        )
        self.files_today = MyDataset.get_folder_files(today_data_dir)
        self.files_yesterday = MyDataset.get_folder_files(yesterday_data_dir)
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
            points = data["markers"]
            total_frames = points.shape[0]
            targets = data["target_id_array"]
            targets = targets[:, 0]
            valid_frames = (targets >= 3) & np.isin((targets - 2) % 3, [1, 2])
            displacement_vectors = np.diff(points, axis=0)
            displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=2)
            mean_displacement = np.mean(displacement_magnitudes, axis=1)
            padded_displacement = np.pad(mean_displacement, (1, 1), mode="edge")
            kernel = np.ones(2) / 2.0
            smoothed_displacement = np.convolve(
                padded_displacement, kernel, mode="valid"
            )
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
                    while (
                        clips_found < self.data_points_per_trajectory
                        and attempts < max_attempts
                    ):
                        start_idx = NP_RNG.integers(num_possible_starts)
                        if valid_frames[
                            start_idx : start_idx + dilated_clip_len : dilation
                        ].all() and self.clip_contains_vein(i, start_idx, dilation):
                            self.data_points_today.append(
                                ("today", "grid_search", file_path, start_idx, dilation)
                            )
                            clips_found += 1
                        attempts += 1
        for i in range(len(self.files_yesterday)):
            file_path = self.files_yesterday[i]
            file_num = MyDataset.extract_trajectory_number(file_path)
            if file_num % 2 == 0:
                traj_type = "grid_search"
            else:
                traj_type = "random"
            data = np.load(file_path)
            points = data["markers"]
            total_frames = points.shape[0]
            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    start_indices = sorted(
                        NP_RNG.choice(
                            range(num_possible_starts),
                            size=min(self.data_points_per_trajectory, num_possible_starts),
                            replace=False,
                        )
                    )
                    for start_idx in start_indices:
                        self.data_points_yesterday.append(
                            ("yesterday", traj_type, file_path, start_idx, dilation)
                        )
        print("clips have now been populated!")

    def populate_clips_new_scheme(self):
        new_data_dir = SYSTEM_PARAMS.files.dataset_root_2025_08_21_reordered
        self.new_files = MyDataset.get_folder_files(new_data_dir)
        self.vein_masks_new_scheme = []
        for i in range(len(self.new_files)):
            self.vein_masks_new_scheme.append(
                self.video_contains_vein(self.new_files[i])
            )
        dilations = [1, 2, 4]
        for i in range(len(self.new_files)):
            file_path = self.new_files[i]
            data = np.load(file_path)
            points = data["markers"]
            total_frames = points.shape[0]
            targets = data["target_id_array"]
            targets = targets[:, 0]
            valid_frames = (targets >= 3) & np.isin((targets - 2) % 3, [1, 2])
            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    clips_found = 0
                    max_attempts = num_possible_starts * 2
                    attempts = 0
                    while (
                        clips_found < self.data_points_per_trajectory
                        and attempts < max_attempts
                    ):
                        start_idx = NP_RNG.integers(num_possible_starts)
                        if valid_frames[
                            start_idx : start_idx + dilated_clip_len : dilation
                        ].all() and self.clip_contains_vein(i, start_idx, dilation):
                            self.data_points_pos.append(
                                ("pos", file_path, start_idx, dilation)
                            )
                            clips_found += 1
                        attempts += 1
        dilations = [4, 8, 16, 32]
        for i in range(len(self.new_files)):
            file_path = self.new_files[i]
            file_num = MyDataset.extract_trajectory_number(file_path)
            if file_num % 2 == 0:
                traj_type = "grid_search"
            else:
                traj_type = "random"
            if traj_type == "random":
                continue
            data = np.load(file_path)
            points = data["markers"]
            total_frames = points.shape[0]
            for dilation in dilations:
                dilated_clip_len = self.clip_len * dilation
                if total_frames >= dilated_clip_len:
                    num_possible_starts = total_frames - dilated_clip_len + 1
                    start_indices = sorted(
                        NP_RNG.choice(
                            range(num_possible_starts),
                            size=min(self.data_points_per_trajectory, num_possible_starts),
                            replace=False,
                        )
                    )
                    for start_idx in start_indices:
                        self.data_points_neg.append(
                            ("neg", file_path, start_idx, dilation)
                        )
        print("clips have now been populated!")

    def __len__(self):
        return len(self.data_points)

    @staticmethod
    def interpolate(a, b, b_weight):
        return a * (1 - b_weight) + b * b_weight

    def set_difficulty_level(self, difficulty):
        self.min_disp_c = MyDataset.interpolate(*self.min_disp_c_range, difficulty)
        self.min_vein_length = MyDataset.interpolate(
            *self.min_vein_length_range, difficulty
        )
        self.max_disp_c = MyDataset.interpolate(*self.max_disp_c_range, difficulty)
        self.max_vein_length = MyDataset.interpolate(
            *self.max_vein_length_range, difficulty
        )
        self.difficulty_fyi = difficulty

    def create_splits(self, *args, **kwargs):
        if self.scheme == "old":
            return self.create_splits_old_scheme(*args, **kwargs)
        elif self.scheme == "new":
            return self.create_splits_new_scheme(*args, **kwargs)

    def create_splits_old_scheme(self, train_size, val_size, test_size):
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
            train_indices = [
                i for traj in train_trajectories for i in trajectory_to_indices[traj]
            ]
            val_indices = [
                i for traj in val_trajectories for i in trajectory_to_indices[traj]
            ]
            test_indices = [
                i for traj in test_trajectories for i in trajectory_to_indices[traj]
            ]
            all_indices.append(
                {
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                }
            )
        today = all_indices[0]
        yesterday = all_indices[1]
        train_today = [self.data_points_today[i] for i in today["train_indices"]]
        train_yesterday = [
            self.data_points_yesterday[i] for i in yesterday["train_indices"]
        ]
        val_today = [self.data_points_today[i] for i in today["val_indices"]]
        val_yesterday = [
            self.data_points_yesterday[i] for i in yesterday["val_indices"]
        ]
        test_today = [self.data_points_today[i] for i in today["test_indices"]]
        test_yesterday = [
            self.data_points_yesterday[i] for i in yesterday["test_indices"]
        ]
        train_dataset = MyDataset(
            scheme=self.scheme,
            mode="train",
            data_points_today=train_today,
            data_points_yesterday=train_yesterday,
        )
        val_dataset = MyDataset(
            scheme=self.scheme,
            mode="val",
            data_points_today=val_today,
            data_points_yesterday=val_yesterday,
        )
        test_dataset = MyDataset(
            scheme=self.scheme,
            mode="test",
            data_points_today=test_today,
            data_points_yesterday=test_yesterday,
        )
        res = (train_dataset, val_dataset, test_dataset)
        print(f"split len: {[len(x) for x in res]}")
        return res

    def create_splits_new_scheme(self, train_size, val_size, test_size):
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
            train_indices = [
                i for traj in train_trajectories for i in trajectory_to_indices[traj]
            ]
            val_indices = [
                i for traj in val_trajectories for i in trajectory_to_indices[traj]
            ]
            test_indices = [
                i for traj in test_trajectories for i in trajectory_to_indices[traj]
            ]
            all_indices.append(
                {
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                }
            )
        pos = all_indices[0]
        neg = all_indices[1]
        train_pos = [self.data_points_pos[i] for i in pos["train_indices"]]
        train_neg = [self.data_points_neg[i] for i in neg["train_indices"]]
        val_pos = [self.data_points_pos[i] for i in pos["val_indices"]]
        val_neg = [self.data_points_neg[i] for i in neg["val_indices"]]
        test_pos = [self.data_points_pos[i] for i in pos["test_indices"]]
        test_neg = [self.data_points_neg[i] for i in neg["test_indices"]]
        train_dataset = MyDataset(
            scheme=self.scheme,
            mode="train",
            data_points_pos=train_pos,
            data_points_neg=train_neg,
        )
        val_dataset = MyDataset(
            scheme=self.scheme,
            mode="val",
            data_points_pos=val_pos,
            data_points_neg=val_neg,
        )
        test_dataset = MyDataset(
            scheme=self.scheme,
            mode="test",
            data_points_pos=test_pos,
            data_points_neg=test_neg,
        )
        res = (train_dataset, val_dataset, test_dataset)
        print(f"split len: {[len(x) for x in res]}")
        return res

    def compute_file_contains_vein(self, file_path):
        file_name = os.path.basename(file_path)
        file_num = int(file_name.split("_")[1][:4])
        return (file_num % 4) < 2

    def clip_contains_vein(self, file_ix, start, dilation):
        if self.scheme == "old":
            markers, veins, contains_vein = self.file_today_vein_masks[file_ix]
        elif self.scheme == "new":
            markers, veins, contains_vein = self.vein_masks_new_scheme[file_ix]
        dilated_clip_len = self.clip_len * dilation
        clip_vein_mask = contains_vein[start : start + dilated_clip_len : dilation]
        clip_veins = veins[start : start + dilated_clip_len : dilation]
        veins_present = np.all(clip_vein_mask, axis=0)
        if not np.any(veins_present):
            return False
        first_frame = clip_veins[0]
        last_frame = clip_veins[-1]
        displacement = np.linalg.norm(last_frame - first_frame, axis=2)
        mean_displacement = np.mean(displacement, axis=1)
        meets_displacement = mean_displacement >= 168
        valid_veins = veins_present & meets_displacement
        return np.any(valid_veins)

    def video_contains_vein(self, file_path):
        data = np.load(file_path)
        markers = data["markers"]
        labels = data["vein_polyline"]
        labels_mask = data["vein_polyline_mask"]
        mean_positions = np.mean(markers, axis=1)
        r = SYSTEM_PARAMS.fisheye_model.circle_radius / 3
        centres = mean_positions[:, np.newaxis, np.newaxis, :]
        cx = centres[..., 0]
        cy = centres[..., 1]
        x = labels[..., 0]
        y = labels[..., 1]
        in_circle = ((x - cx) ** 2 + (y - cy) ** 2) <= r**2
        labels_mask &= in_circle
        contains_vein = np.any(labels_mask, axis=2)
        return (markers, labels, contains_vein)

    def frame_contains_vein(self, data, ix):
        markers = data["markers"][ix]
        markers_mask = data["markers_mask"][ix]
        labels = data["vein_polyline"][ix]
        labels_mask = data["vein_polyline_mask"][ix]
        valid_markers = markers[markers_mask]
        centre = np.mean(valid_markers, axis=0)
        cx = centre[0]
        cy = centre[1]
        r = SYSTEM_PARAMS.fisheye_model.circle_radius / 3
        mask_circle = SyntheticImageGenerator.filter_points_vectorised(
            cx, cy, r, labels
        )
        labels_mask &= mask_circle
        res = labels[labels_mask].sum() > 0
        return res

    @staticmethod
    def compute_mean_marker_position(markers, markers_mask):
        frame_means = []
        for frame_idx in range(markers.shape[0]):
            valid_markers = markers[frame_idx][markers_mask[frame_idx]]
            if len(valid_markers) > 0:
                frame_mean = np.mean(valid_markers, axis=0)
                frame_means.append(frame_mean)
        if frame_means:
            return np.mean(frame_means, axis=0)
        else:
            return np.array([-1.0, -1.0])

    def __getitem__(self, idx):
        if self.mode == "exp":
            frame_ix = self.data_points[idx]
            pyg = self.get_clip(
                markers=self.exp_markers,
                clip_len=self.clip_len,
                dilation=self.exp_dilation,
                start_ix=frame_ix,
                ground_truth_labels_in=self.exp_ground_truth_labels,
            )
            veins = torch.empty(0)
            return pyg, veins
        else:
            if self.scheme == "old":
                day, traj_type, file_path, frame_ix, dilation = self.data_points[idx]
                if day == "today":
                    random_num = NP_RNG.uniform(0, 1)
                    spawn_vein = random_num < 0.75
                else:
                    spawn_vein = False
            elif self.scheme == "new":
                pos_neg_str, file_path, frame_ix, dilation = self.data_points[idx]
                spawn_vein = pos_neg_str == "pos"
        data = np.load(file_path)
        markers = data["markers"]
        markers_mask = data["markers_mask"]
        veins = data["vein_polyline"]
        veins_mask = data["vein_polyline_mask"]
        if not spawn_vein:
            veins_mask = np.zeros_like(veins_mask, dtype=bool)
        dilated_clip_len = self.clip_len * dilation
        markers = markers[frame_ix : frame_ix + dilated_clip_len : dilation]
        markers_mask = markers_mask[frame_ix : frame_ix + dilated_clip_len : dilation]
        veins = veins[frame_ix : frame_ix + dilated_clip_len : dilation]
        veins_mask = veins_mask[frame_ix : frame_ix + dilated_clip_len : dilation]
        vein_present_per_frame = np.any(veins_mask, axis=2)
        vein_present_all_frames = np.all(vein_present_per_frame, axis=0)
        if np.sum(vein_present_all_frames) > 1:
            present_vein_indices = np.where(vein_present_all_frames)[0]
            selected_vein_idx = NP_RNG.choice(present_vein_indices)
            vein_present_all_frames[:] = False
            vein_present_all_frames[selected_vein_idx] = True
        veins_mask &= vein_present_all_frames[np.newaxis, :, np.newaxis]
        if spawn_vein:
            freeze_markers = True
        else:
            freeze_markers = NP_RNG.uniform(0, 1) < 0.2
        if freeze_markers:
            first_frame_markers = markers[0]
            marker_displacements = markers - first_frame_markers[np.newaxis, :, :]
            mean_marker_displacement = np.mean(marker_displacements, axis=1)
            markers = np.tile(
                first_frame_markers[np.newaxis, :, :], (markers.shape[0], 1, 1)
            )
            veins -= mean_marker_displacement[:, np.newaxis, np.newaxis, :]
        markers, labels_signal_mask = (
            self.augmentation_artificial_vein_signal_vectorised(
                markers, veins, veins_mask
            )
        )
        veins_mask &= labels_signal_mask
        discrete_angles = [0, 60, 120, 180, 240, 300]
        rotation_angle_deg = NP_RNG.choice(discrete_angles)
        markers = self.augmentation_rotation(markers, rotation_angle_deg)
        veins = self.augmentation_rotation(veins, rotation_angle_deg)
        angle_uniform_shift_rad = NP_RNG.uniform(0, 2 * math.pi)
        magnitude = NP_RNG.uniform(0, 10)
        markers = self.uniform_shift(markers, angle_uniform_shift_rad, magnitude)
        veins = self.uniform_shift(veins, angle_uniform_shift_rad, magnitude)
        angle_x = math.radians(NP_RNG.uniform(-5, 5))
        angle_y = math.radians(NP_RNG.uniform(-5, 5))
        markers = self.rotate_xy(markers, angle_x, angle_y)
        veins = self.rotate_xy(veins, angle_x, angle_y)
        points = markers
        points_mask = markers_mask
        pyg = self.generate_pyg_vectorised(
            points, veins, veins_mask, ground_truth_labels_in=None
        )
        if self.visualisation_mode:
            veins = MyDataset.generate_vein_image(
                self.h_camera_big, self.w_camera_big, veins, veins_mask
            )
            veins = torch.tensor(veins, dtype=torch.float32) / 255.0
        else:
            veins = torch.empty(0)
        return pyg, veins

    def generate_pyg_vectorised(
        self, clip_points, clip_labels, clip_labels_mask, ground_truth_labels_in=None
    ):
        pos = self.get_pos(clip_points)
        mask = self.get_mask()
        y = self.get_y(
            clip_points,
            clip_labels,
            clip_labels_mask,
            ground_truth_labels_in,
        )
        empty_x = self.get_empty_x()
        node_xy = self.get_node_xy(clip_points)
        node_base_graph_polar_coords = self.get_node_base_graph_polar_coords()
        node_one_hot_encoding = self.get_node_one_hot_encoding()
        time_one_hot_encoding = self.get_time_one_hot_encoding()
        regular_nodes = np.concatenate(
            [
                node_xy,
                node_base_graph_polar_coords,
                node_one_hot_encoding,
                time_one_hot_encoding,
            ],
            axis=1,
        )
        edge_index_spatial = self.get_edge_index_spatial()
        spatial_edge_dist_dx_dy_cos_sin = self.get_spatial_edge_dist_dx_dy_cos_sin(
            clip_points
        )
        edge_attr_spatial = np.concatenate(
            [
                spatial_edge_dist_dx_dy_cos_sin,
            ],
            axis=1,
        )
        edge_index_temporal = self.get_edge_index_temporal()
        temporal_edge_dist_dx_dy_cos_sin = self.get_temporal_edge_dist_dx_dy_cos_sin(
            clip_points
        )
        temporal_edge_time_direction = self.get_temporal_edge_time_direction()
        edge_attr_temporal = np.concatenate(
            [
                temporal_edge_dist_dx_dy_cos_sin,
                temporal_edge_time_direction,
            ],
            axis=1,
        )
        edge_index_global_spatial = self.get_edge_index_global_spatial()
        edge_attr_global_spatial = self.get_empty_edge_attr_global_spatial()
        edge_index_global_temporal = self.get_edge_index_global_temporal()
        global_temporal_edge_time_direction = (
            self.get_global_temporal_edge_time_direction()
        )
        edge_attr_global_temporal = np.concatenate(
            [
                global_temporal_edge_time_direction,
            ],
            axis=1,
        )
        self.normalise(
            pos,
            regular_nodes,
            edge_attr_spatial,
            edge_attr_temporal,
            edge_attr_global_spatial,
            edge_attr_global_temporal,
        )
        pyg_data = self.get_pyg_data(
            pos=pos,
            mask=mask,
            y=y,
            empty_x=empty_x,
            regular_nodes=regular_nodes,
            edge_index_spatial=edge_index_spatial,
            edge_attr_spatial=edge_attr_spatial,
            edge_index_temporal=edge_index_temporal,
            edge_attr_temporal=edge_attr_temporal,
            edge_index_global_spatial=edge_index_global_spatial,
            edge_attr_global_spatial=edge_attr_global_spatial,
            edge_index_global_temporal=edge_index_global_temporal,
            edge_attr_global_temporal=edge_attr_global_temporal,
        )
        return pyg_data

    def get_pos(self, clip_points):
        pos = np.zeros(shape=(self.clip_len * self.num_nodes, 2), dtype=float)
        for t in range(self.clip_len):
            points = clip_points[t]
            pos[t * self.num_nodes : (t + 1) * self.num_nodes, 0:2] = points
        return pos

    def get_y(self, clip_points, clip_labels, clip_labels_mask, ground_truth_labels_in):
        if ground_truth_labels_in is not None:
            y = ground_truth_labels_in.reshape(
                (self.clip_len * self.num_nodes,)
            ).astype(int)
            return y
        else:
            y = np.zeros(shape=(self.clip_len * self.num_nodes,), dtype=int)
            for t in range(self.clip_len):
                points = clip_points[t]
                labels = clip_labels[t]
                labels_mask = clip_labels_mask[t]
                labels_filtered = labels[labels_mask]
                if labels_filtered.size > 0:
                    distances = cdist(points, labels_filtered)
                    min_distances = np.min(distances, axis=1)
                    px_threshold = SYSTEM_PARAMS.meta.vein_px_thickness
                    y_t = min_distances < px_threshold
                else:
                    y_t = np.zeros(shape=(self.num_nodes,), dtype=int)
                y[t * self.num_nodes : (t + 1) * self.num_nodes] = y_t
            return y

    def get_mask(self):
        mask = np.zeros(shape=(self.clip_len * (self.num_nodes + 1),), dtype=bool)
        mask[: self.clip_len * self.num_nodes] = True
        return mask
    
    def get_empty_x(self):
        empty_x = np.zeros(shape=(self.clip_len * self.num_nodes, 0), dtype=float)
        return empty_x

    def get_node_xy(self, clip_points):
        node_features_list = []
        for t in range(self.clip_len):
            points = clip_points[t]
            node_features = points.copy()
            node_features_list.append(node_features)
        node_features_clip = np.concatenate(node_features_list, axis=0)
        return node_features_clip

    def get_node_base_graph_polar_coords(self):
        node_features_list = []
        for t in range(self.clip_len):
            dist = self.dist_from_centre.reshape(-1, 1)
            angles = self.angles
            node_features = np.concatenate((dist, angles), axis=1)
            node_features_list.append(node_features)
        node_features_clip = np.concatenate(node_features_list, axis=0)
        return node_features_clip

    def get_node_one_hot_encoding(self):
        node_features_list = []
        for t in range(self.clip_len):
            node_features = np.eye(self.num_nodes, dtype=int)
            node_features_list.append(node_features)
        node_features_clip = np.concatenate(node_features_list, axis=0)
        return node_features_clip

    def get_time_one_hot_encoding(self):
        node_features_list = []
        one_hot = np.eye(self.clip_len, dtype=int)
        for t in range(self.clip_len):
            one_hot_cur = one_hot[t]
            node_features = np.tile(one_hot_cur, (self.num_nodes, 1))
            node_features_list.append(node_features)
        node_features_clip = np.concatenate(node_features_list, axis=0)
        return node_features_clip

    def get_edge_index_spatial(self):
        adjacency_clip_list = []
        for t in range(self.clip_len):
            adjacency_frame = self.adjacency + t * self.num_nodes
            adjacency_clip_list.append(adjacency_frame)
        adjacency_clip = np.concatenate(adjacency_clip_list, axis=0)
        adjacency_clip = adjacency_clip.T
        return adjacency_clip

    def get_spatial_edge_dist_dx_dy_cos_sin(self, clip_points):
        edge_attr_clip_list = []
        for t in range(self.clip_len):
            points = clip_points[t]
            edge_attr = np.zeros(shape=(self.num_edges_single_frame, 5), dtype=float)
            me_points = points[self.adjacency[:, 0]]
            neighbor_points = points[self.adjacency[:, 1]]
            vec = neighbor_points - me_points
            dist = np.linalg.norm(vec, axis=1)
            edge_attr[:, 0] = dist
            edge_attr[:, 1:3] = vec
            dist_safe = np.maximum(dist[:, np.newaxis], 1e-6)
            vec_normalized = vec / dist_safe
            edge_attr[:, 3] = vec_normalized[:, 0]
            edge_attr[:, 4] = vec_normalized[:, 1]
            edge_attr_clip_list.append(edge_attr)
        edge_attr_clip = np.concatenate(edge_attr_clip_list, axis=0)
        return edge_attr_clip

    def get_empty_edge_index_temporal(self):
        return np.zeros(shape=(2, 0), dtype=int)

    def get_edge_index_temporal(self):
        adjacency_clip_list = []
        for t in range(self.clip_len - 1):
            src_dst = np.array([[t, t + 1], [t + 1, t]], dtype=int)
            for i in range(src_dst.shape[0]):
                src, dst = src_dst[i]
                src = src * self.num_nodes + np.arange(self.num_nodes)
                dst = dst * self.num_nodes + np.arange(self.num_nodes)
                edge_index_single = np.concatenate(
                    (src[:, np.newaxis], dst[:, np.newaxis]), axis=1
                )
                adjacency_clip_list.append(edge_index_single)
        adjacency_clip = np.concatenate(adjacency_clip_list, axis=0)
        adjacency_clip = adjacency_clip.T
        return adjacency_clip

    def get_temporal_edge_dist_dx_dy_cos_sin(self, clip_points):
        edge_attr_clip_list = []
        for t in range(self.clip_len - 1):
            src_dst = np.array([[t, t + 1], [t + 1, t]], dtype=int)
            for i in range(src_dst.shape[0]):
                src, dst = src_dst[i]
                edge_attr = np.zeros(shape=(self.num_nodes, 5), dtype=float)
                me_points = clip_points[src]
                neighbor_points = clip_points[dst]
                vec = neighbor_points - me_points
                dist = np.linalg.norm(vec, axis=1)
                edge_attr[:, 0] = dist
                edge_attr[:, 1:3] = vec
                dist_safe = np.maximum(dist[:, np.newaxis], 1e-6)
                vec_normalized = vec / dist_safe
                edge_attr[:, 3] = vec_normalized[:, 0]
                edge_attr[:, 4] = vec_normalized[:, 1]
                edge_attr_clip_list.append(edge_attr)
        edge_attr_clip = np.concatenate(edge_attr_clip_list, axis=0)
        return edge_attr_clip

    def get_temporal_edge_time_direction(self):
        edge_attr_clip_list = []
        for t in range(self.clip_len - 1):
            src_dst = np.array([[t, t + 1], [t + 1, t]], dtype=int)
            for i in range(src_dst.shape[0]):
                src, dst = src_dst[i]
                edge_attr = np.zeros(shape=(self.num_nodes, 2), dtype=float)
                if i == 0:
                    edge_attr[:, 1] = 1
                else:
                    edge_attr[:, 0] = 1
                edge_attr_clip_list.append(edge_attr)
        edge_attr_clip = np.concatenate(edge_attr_clip_list, axis=0)
        return edge_attr_clip

    def get_empty_edge_attr_temporal(self):
        return np.zeros(shape=(0, 0), dtype=float)

    def get_edge_index_global_spatial(self):
        num_regular_nodes = self.num_nodes * self.clip_len
        global_node_ixs = np.arange(num_regular_nodes, num_regular_nodes+self.clip_len)

        adjacency_clip_list = []
        for t in range(self.clip_len):
            node_ixs = t * self.num_nodes + np.arange(self.num_nodes)
            global_node_ix = global_node_ixs[t]
            global_node_ix_arr = np.full(
                shape=(self.num_nodes,), 
                fill_value=global_node_ix
            )
            reg_glob = np.concatenate(
                (node_ixs[:, np.newaxis], global_node_ix_arr[:, np.newaxis]), axis=1
            )
            adjacency_clip_list.append(reg_glob)
            glob_reg = np.concatenate(
                (global_node_ix_arr[:, np.newaxis], node_ixs[:, np.newaxis]), axis=1
            )
            adjacency_clip_list.append(glob_reg)
        adjacency_clip = np.concatenate(adjacency_clip_list, axis=0)
        adjacency_clip = adjacency_clip.T
        return adjacency_clip

    def get_empty_edge_attr_global_spatial(self):
        num_regular_nodes = self.num_nodes * self.clip_len
        return np.zeros(shape=(num_regular_nodes * 2, 0), dtype=float)

    def get_edge_index_global_temporal(self):
        num_regular_nodes = self.num_nodes * self.clip_len
        adjacency_clip_list = []
        for t in range(self.clip_len - 1):
            src_dst = np.array([[t, t + 1], [t + 1, t]], dtype=int)
            for i in range(src_dst.shape[0]):
                arr = src_dst[i]+num_regular_nodes
                arr = arr[np.newaxis, :]
                adjacency_clip_list.append(
                    arr
                )
        adjacency_clip = np.concatenate(adjacency_clip_list, axis=0)
        adjacency_clip = adjacency_clip.T
        return adjacency_clip

    def get_empty_edge_attr_global_temporal(self):
        return np.zeros(shape=((self.clip_len - 1) * 2, 0), dtype=float)

    def get_global_temporal_edge_time_direction(self):
        edge_attr_clip_list = []
        for t in range(self.clip_len - 1):
            src_dst = np.array([[t, t + 1], [t + 1, t]], dtype=int)
            for i in range(src_dst.shape[0]):
                src, dst = src_dst[i]
                edge_attr = np.zeros(shape=(1, 2), dtype=float)
                if i == 0:
                    edge_attr[:, 1] = 1
                else:
                    edge_attr[:, 0] = 1
                edge_attr_clip_list.append(edge_attr)
        edge_attr_clip = np.concatenate(edge_attr_clip_list, axis=0)
        return edge_attr_clip

    def get_entity_tag_one_hot(self, ix, num_entities):
        one_hot = self.entity_tag_one_hot[ix]
        res = np.tile(one_hot, (num_entities, 1))
        return res

    def normalise(
        self,
        pos,
        regular_nodes,
        edge_attr_spatial,
        edge_attr_temporal,
        edge_attr_global_spatial,
        edge_attr_global_temporal,
    ):
        if not self.warmup:
            if self.normalise_pos:
                MyDataset.normalise_single(self.pos_mean, self.pos_std, pos, [0, 1])
            MyDataset.normalise_single(
                self.regular_nodes_mean,
                self.regular_nodes_std,
                regular_nodes,
                [0, 1, 2],
            )
            MyDataset.normalise_single(
                self.edge_attr_spatial_mean,
                self.edge_attr_spatial_std,
                edge_attr_spatial,
                [0, 1, 2],
            )
            MyDataset.normalise_single(
                self.edge_attr_temporal_mean,
                self.edge_attr_temporal_std,
                edge_attr_temporal,
                [0, 1, 2],
            )
            MyDataset.normalise_single(
                self.edge_attr_global_spatial_mean,
                self.edge_attr_global_spatial_std,
                edge_attr_global_spatial,
                [],
            )
            MyDataset.normalise_single(
                self.edge_attr_global_temporal_mean,
                self.edge_attr_global_temporal_std,
                edge_attr_global_temporal,
                [],
            )

    def get_pyg_data(
        self,
        pos,
        mask,
        y,
        empty_x,
        regular_nodes,
        edge_index_spatial,
        edge_attr_spatial,
        edge_index_temporal,
        edge_attr_temporal,
        edge_index_global_spatial,
        edge_attr_global_spatial,
        edge_index_global_temporal,
        edge_attr_global_temporal,
    ):
        pos = torch.tensor(pos, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.bool)
        y = torch.tensor(y, dtype=torch.long)
        empty_x = torch.tensor(empty_x, dtype=torch.float)
        regular_nodes = torch.tensor(regular_nodes, dtype=torch.float)
        edge_index_spatial = torch.tensor(edge_index_spatial, dtype=torch.long)
        edge_attr_spatial = torch.tensor(edge_attr_spatial, dtype=torch.float)
        edge_index_temporal = torch.tensor(edge_index_temporal, dtype=torch.long)
        edge_attr_temporal = torch.tensor(edge_attr_temporal, dtype=torch.float)
        edge_index_global_spatial = torch.tensor(
            edge_index_global_spatial, dtype=torch.long
        )
        edge_attr_global_spatial = torch.tensor(
            edge_attr_global_spatial, dtype=torch.float
        )
        edge_index_global_temporal = torch.tensor(
            edge_index_global_temporal, dtype=torch.long
        )
        edge_attr_global_temporal = torch.tensor(
            edge_attr_global_temporal, dtype=torch.float
        )
        pyg_data = Data(
            pos=pos,
            mask=mask,
            y=y,
            x=empty_x,
            regular_nodes=regular_nodes,
            edge_index_spatial=edge_index_spatial,
            edge_attr_spatial=edge_attr_spatial,
            edge_index_temporal=edge_index_temporal,
            edge_attr_temporal=edge_attr_temporal,
            edge_index_global_spatial=edge_index_global_spatial,
            edge_attr_global_spatial=edge_attr_global_spatial,
            edge_index_global_temporal=edge_index_global_temporal,
            edge_attr_global_temporal=edge_attr_global_temporal,
        )
        return pyg_data

    @staticmethod
    def normalise_single(means, stds, xs, ixs):
        ixs = np.array(ixs)
        if ixs.size > 0:
            xs[:, ixs] = (xs[:, ixs] - means[ixs]) / stds[ixs]

    def get_clip(
        self, markers, clip_len, dilation, start_ix, ground_truth_labels_in=None
    ):
        points = markers
        dilated_clip_len = clip_len * dilation
        points = points[start_ix : start_ix + dilated_clip_len : dilation]
        ground_truth_labels_in = ground_truth_labels_in[
            start_ix : start_ix + dilated_clip_len : dilation
        ]
        labels = np.zeros(shape=(points.shape[0], 0, 0, 2), dtype=int)
        labels_mask = np.zeros(shape=(points.shape[0], 0, 0), dtype=bool)
        pyg = self.generate_pyg_vectorised(
            points, labels, labels_mask, ground_truth_labels_in
        )
        return pyg

    def eval(self):
        self.visualisation_mode = True

    def get_points(self, idx):
        day, traj_type, file_path, frame_ix, dilation = self.data_points[idx]
        data = np.load(file_path)
        points = data["markers"][start_ix]
        return points

    @staticmethod
    def normalise_gnn_points(points):
        principal_point = np.array(
            [
                SYSTEM_PARAMS.fisheye_model.principal_point_x,
                SYSTEM_PARAMS.fisheye_model.principal_point_y,
            ],
            dtype=float,
        )
        r = SYSTEM_PARAMS.fisheye_model.circle_radius
        points -= principal_point
        points /= r
        return points

    @staticmethod
    def unnormalise_gnn_points(points):
        principal_point = np.array(
            [
                SYSTEM_PARAMS.fisheye_model.principal_point_x,
                SYSTEM_PARAMS.fisheye_model.principal_point_y,
            ],
            dtype=float,
        )
        r = SYSTEM_PARAMS.fisheye_model.circle_radius
        points *= r
        points += principal_point
        return points

    def get_markers(self, idx):
        day, traj_type, file_path, frame_ix, dilation = self.data_points[idx]
        data = np.load(file_path)
        markers = data["markers"]
        markers_mask = data["markers_mask"]
        labels = data["vein_polyline"]
        labels_mask = data["vein_polyline_mask"]
        dilated_clip_len = self.clip_len * dilation
        images = markers[start : start + dilated_clip_len : dilation]
        images_mask = markers_mask[start : start + dilated_clip_len : dilation]
        labels = labels[start : start + dilated_clip_len : dilation]
        labels_mask = labels_mask[start : start + dilated_clip_len : dilation]
        images = MyDataset.downscale(self.k, images)
        labels = MyDataset.downscale(self.k, labels)
        return images[self.clip_len // 2, :, :]

    @staticmethod
    def generate_markers_image(h, w, points, points_mask):
        if points.shape[-1] == 0:
            return np.zeros((points.shape[0], h, w), dtype=np.uint8)
        images = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        marker_masks = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        for t in range(points.shape[0]):
            filtered_points = []
            for i, point in enumerate(points[t]):
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
        if len(images.shape) == 3:
            _, h, w = images.shape
            is_grayscale = True
        else:
            _, h, w, _ = images.shape
            is_grayscale = False
        x, y = point[0], point[1]
        center_x = x
        center_y = y
        half_size = n / 2
        x0 = int(np.floor(center_x - half_size + 0.5))
        x_end = x0 + n
        y0 = int(np.floor(center_y - half_size + 0.5))
        y_end = y0 + n
        for yi in range(y0, y_end):
            for xi in range(x0, x_end):
                if 0 <= xi < w and 0 <= yi < h:
                    dx = abs(xi + 0.5 - center_x)
                    dy = abs(yi + 0.5 - center_y)
                    wx = max(0, 1 - dx / (half_size))
                    wy = max(0, 1 - dy / (half_size))
                    weight = wx * wy
                    if is_grayscale:
                        avg_intensity = sum(intensity) // 3
                        current_val = int(images[t, yi, xi])
                        weight_val = int(avg_intensity * weight)
                        images[t, yi, xi] = max(0, min(255, current_val + weight_val))
                    else:
                        for c in range(3):
                            current_val = int(images[t, yi, xi, c])
                            weight_val = int(intensity[c] * weight)
                            images[t, yi, xi, c] = max(
                                0, min(255, current_val + weight_val)
                            )

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
        if points.shape[-1] == 0:
            return np.zeros((points.shape[0], h, w), dtype=np.uint8)
        images = np.zeros((points.shape[0], h, w), dtype=np.uint8)
        for t in range(points.shape[0]):
            for v in range(points.shape[1]):
                vein_points = points[t, v]
                vein_mask = points_mask[t, v]
                valid_points = vein_points[vein_mask]
                if len(valid_points) > 0:
                    line_points = valid_points.astype(np.int32)
                    for i in range(len(line_points) - 1):
                        pt1 = tuple(line_points[i])
                        pt2 = tuple(line_points[i + 1])
                        cv2.line(
                            images[t],
                            pt1,
                            pt2,
                            color=255,
                            thickness=SYSTEM_PARAMS.meta.vein_px_thickness,
                        )
        return images

    def augmentation_rotation(self, points, angle_degrees):
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        if points.shape[-1] == 0:
            return points
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        centered_points = points - np.array([cx, cy])
        rotated_points = np.zeros_like(centered_points)
        rotated_points[..., 0] = (
            centered_points[..., 0] * cos_a - centered_points[..., 1] * sin_a
        )
        rotated_points[..., 1] = (
            centered_points[..., 0] * sin_a + centered_points[..., 1] * cos_a
        )
        rotated_points = rotated_points + np.array([cx, cy])
        return rotated_points

    def uniform_shift(self, points, angle_rad, magnitude):
        if points.shape[-1] == 0:
            return points
        shift_x = magnitude * math.cos(angle_rad)
        shift_y = magnitude * math.sin(angle_rad)
        shifted_points = points + np.array([shift_x, shift_y])
        return shifted_points

    def shift_radial(self, points, radial_shift):
        if points.shape[-1] == 0:
            return points
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        dx = points[..., 0] - cx
        dy = points[..., 1] - cy
        distances = np.sqrt(dx**2 + dy**2)
        nonzero_mask = distances > 0
        shifted_points = points.copy()
        unit_x = np.zeros_like(dx)
        unit_y = np.zeros_like(dy)
        unit_x[nonzero_mask] = dx[nonzero_mask] / distances[nonzero_mask]
        unit_y[nonzero_mask] = dy[nonzero_mask] / distances[nonzero_mask]
        shifted_points[..., 0] = points[..., 0] + radial_shift * unit_x
        shifted_points[..., 1] = points[..., 1] + radial_shift * unit_y
        return shifted_points

    def rotate_xy(self, points, angle_x, angle_y):
        if points.shape[-1] == 0:
            return points
        points_3d = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(points)
        z_coord = (
            SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface
            - SYSTEM_PARAMS.trajectory.press_depth_1
        )
        assert np.allclose(points_3d[..., 2], z_coord, atol=1e-6), (
            f"All z-coordinates must be equal to {z_coord}"
        )
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        translated_points = points_3d - np.array([0, 0, z_coord])
        points_reshaped = translated_points.reshape(-1, 3).T
        points_rotated_x = Rx @ points_reshaped
        points_rotated_xy = Ry @ points_rotated_x
        rotated_points_3d = (points_rotated_xy.T).reshape(points_3d.shape)
        rotated_points_3d = rotated_points_3d + np.array([0, 0, z_coord])
        rotated_points_2d = FisheyeModelNoTaichi.project_3d_2d_np(
            rotated_points_3d.reshape(-1, 3)
        ).reshape(points.shape)
        return rotated_points_2d

    def randomly_remove(self, points):
        return
        if points.shape[-1] == 0:
            return np.ones((points.shape[0], 0), dtype=bool)
        max_k = self.randomly_remove_k[self.difficulty]
        mask = np.ones(points.shape[:2], dtype=bool)
        for frame_idx in range(points.shape[0]):
            k = NP_RNG.integers(0, min(max_k, points.shape[1])+1)
            if k > 0:
                indices_to_remove = NP_RNG.choice(
                    range(points.shape[1]), 
                    size=k,
                    replace=False,
                )
                mask[frame_idx, indices_to_remove] = False
        return mask

    @staticmethod
    def downscale(k, points):
        if points.shape[-1] == 0:
            return points, np.ones((points.shape[0], 0), dtype=bool)
        origin = np.array(
            [SYSTEM_PARAMS.fisheye_model.crop_x, SYSTEM_PARAMS.fisheye_model.crop_y],
            dtype=float,
        )
        resolution = np.array(
            [
                SYSTEM_PARAMS.fisheye_model.crop_width,
                SYSTEM_PARAMS.fisheye_model.crop_height,
            ],
            dtype=float,
        )
        points, mask = SyntheticImageGenerator.crop(points, origin, resolution)
        points = points / k
        return points

    @staticmethod
    def make_3x3(points):
        origin = np.array(
            [-SYSTEM_PARAMS.fisheye_model.crop_x, -SYSTEM_PARAMS.fisheye_model.crop_y],
            dtype=float,
        )
        resolution = np.array(
            [
                3 * SYSTEM_PARAMS.fisheye_model.crop_width,
                3 * SYSTEM_PARAMS.fisheye_model.crop_height,
            ],
            dtype=float,
        )
        points, mask = SyntheticImageGenerator.crop(points, origin, resolution)
        return points, mask

    def augmentation_artificial_vein_signal_vectorised(
        self, clip_points, clip_vein_polyline, clip_vein_polyline_mask
    ):
        valid_frames_mask = np.any(np.all(clip_vein_polyline_mask, axis=2), axis=1)
        num_veins = clip_vein_polyline.shape[1]
        num_vein_points = clip_vein_polyline.shape[2]
        if not np.any(valid_frames_mask):
            return clip_points, clip_vein_polyline_mask
        disp_c = NP_RNG.uniform(self.min_disp_c, self.max_disp_c)
        target_num_vein_points = NP_RNG.integers(
            int(self.min_vein_length * num_vein_points),
            int(self.max_vein_length * num_vein_points) + 1,
            size=num_veins,
        )
        max_start_indices = num_vein_points - target_num_vein_points
        start_indices = NP_RNG.integers(0, max_start_indices + 1)
        row_indices = np.arange(num_veins)[:, np.newaxis]
        col_indices = np.arange(num_vein_points)[np.newaxis, :]
        my_vein_mask = (col_indices >= start_indices[:, np.newaxis]) & (
            col_indices
            < (start_indices[:, np.newaxis] + target_num_vein_points[:, np.newaxis])
        )
        clip_vein_polyline_mask &= my_vein_mask[np.newaxis, :, :]
        for t in range(clip_points.shape[0]):
            points = clip_points[t]
            x_0 = SYSTEM_PARAMS.meta.px_dist_adjacent_markers / 2
            for v in range(clip_vein_polyline[t].shape[0]):
                tv_polyline = clip_vein_polyline[t, v, :, :][
                    clip_vein_polyline_mask[t, v, :]
                ]
                if tv_polyline.size == 0:
                    continue
                vecs = SyntheticImageGenerator.vector_point_to_polynomial(
                    tv_polyline, points
                )
                x = np.linalg.norm(vecs, axis=1)
                displacement = np.zeros_like(x)
                mask1 = (0 < x) & (x < x_0)
                mask2 = (x_0 <= x) & (x < 2 * x_0)
                displacement[mask1] = x[mask1]
                displacement[mask2] = x_0 - (x[mask2] - x_0)
                mask = displacement > 0
                vec_normalized = vecs[mask] / x[mask, np.newaxis]
                points[mask] = (
                    points[mask]
                    + vec_normalized * displacement[mask, np.newaxis] * disp_c
                )
            clip_points[t] = points
        return clip_points, clip_vein_polyline_mask

    def augmentation_artificial_vein_signal_unvectorised(
        self, clip_points, clip_vein_polyline, clip_vein_polyline_mask
    ):
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
                length = NP_RNG.integers(3, max_length+1)
            end_idx -= length
            start_idx = NP_RNG.integers(start_idx, end_idx+2)
            end_idx = start_idx + length
            vein_visible_mask = np.zeros(clip_points.shape[0], dtype=bool)
            vein_visible_mask[start_idx:end_idx] = True
        lower_disp_c = self.min_disp_c_range[self.difficulty]
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
                        proceed = NP_RNG.uniform(0, 1)
                    else:
                        proceed = 1.0
                    if proceed < 0.5:
                        continue
                    vec = SyntheticImageGenerator.vector_point_to_polynomial(
                        vein_polyline, points[i]
                    )
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
        match = re.search(r"trajectory_(\d+)\.npz$", file_path)
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
        match = re.search(r"trajectory_(\d+)\.npz$", file_path)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract trajectory number from {file_path}")


class MyDatasetExpIterator:
    def __init__(self, dataset: MyDataset, npz_path, labels_path=None):
        self.dataset = dataset
        data = np.load(npz_path)
        markers = data["markers"]
        markers_mask = data["markers_mask"]
        if False and start_ix is not None and end_ix is not None:
            markers = markers[start_ix:end_ix]
            markers_mask = markers_mask[start_ix:end_ix]
        self.num_frames = markers.shape[0]
        print(f"num frames: {self.num_frames}")
        self.markers = markers
        self.markers_mask = markers_mask
        self.clip_len = SYSTEM_PARAMS.gnn.clip_len
        if labels_path is not None:
            data_labels = np.load(labels_path)
            labels = data_labels["labels"]
        else:
            labels = None
        self.labels = labels

    def __iter__(self):
        return self

    def __next__(self):
        dilation = 1
        dilated_clip_len = self.clip_len * dilation
        min_start_ix = 0
        max_start_ix = self.num_frames - dilated_clip_len
        start_ix = NP_RNG.integers(min_start_ix, max_start_ix+1)
        start_ix = 0
        ground_truth_labels_in = self.labels[
            start_ix : start_ix + dilated_clip_len : dilation
        ]
        pyg = self.dataset.get_clip(
            self.markers, self.clip_len, dilation, start_ix, ground_truth_labels_in
        )
        labels = torch.empty(0)
        return pyg, labels
