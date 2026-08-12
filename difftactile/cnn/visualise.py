import os
import pickle
import sys
import time

import cv2
import matplotlib
from difftactile.main.display import (
    destroy_windows, imshow, is_headless, iteration_limit, move_window, prompt,
    wait_key,
)
# Non-interactive backend before pyplot is imported, so plt.figure() does not
# try to open a Tk window on a display-less machine.
if is_headless():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from difftactile.cnn.dataset import *
from difftactile.cnn.gnn import *
from difftactile.main.paths import repo_path


def _segmentation_gnn(arch):
    """Build the arch-aware GNN from `cnn/segmentation_gnn.py`.

    Imported inside the function rather than at module scope on purpose.
    `cnn/gnn.py` and `cnn/segmentation_gnn.py` each define a class called `GNN`,
    and several modules reach this file through `from ... import *` chains
    (predict_exp.py among them), which resolve `GNN` to whichever name this
    module happens to export. A module-level import here would silently rebind
    their `GNN` to the other class and break checkpoint loading, so the
    arch-aware class is kept out of this module's namespace entirely.
    """
    from difftactile.cnn.segmentation_gnn import GNN as SegmentationGNN
    return SegmentationGNN(arch=arch)


def has_flat_stats_dict(all_stats):
    """True when a stats mapping is a single flat stats dict.

    Meat-scheme loaders store statistics flat; simulation loaders key them by
    curriculum difficulty. Detected by looking for a float difficulty key, since
    what is unpickled here is the `dataset_stats` value rather than the whole
    test-loader dict that `has_flat_stats()` inspects.
    """
    return not any(isinstance(k, float) for k in all_stats)


# The six canonical scenarios the prediction viewer can be pointed at:
# each of the three (train -> test) configurations, loaded either from the
# published checkpoint or from one retrained locally with `--train`.
#
#   arch          : architecture the checkpoint was trained with
#   ckpt_key      : SYSTEM_PARAMS.files key of the published checkpoint
#   stats_key     : test-loader pickle holding the normalisation statistics
#   test_dataset  : which dataset the predictions are shown on
VIEWER_SCENARIOS = {
    "A-to-B": {
        "description": "train on simulation (A), view predictions on silicone (B)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "silicone",
    },
    "C-to-B": {
        "description": "train on meat (C), view predictions on silicone (B)",
        "arch": "compact",
        "ckpt_key": "final_segmentation_model_gnn_meat",
        "stats_key": "test_loader_gnn_meat",
        "test_dataset": "silicone",
    },
    "A-to-C": {
        "description": "train on simulation (A), view predictions on meat (C)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "meat",
    },
}


def _retrained_variant(rel, config):
    """Path of the `*_retrained_<config>` artifact matching a published one.

    Mirrors `segmentation_gnn._retrained_path()`, which is where a `--train` run
    writes so that the published checkpoints survive.
    """
    base, ext = os.path.splitext(rel)
    return f"{base}_retrained_{config}{ext}"


class Visualisation:
    def __init__(self, scenario=None, weights="pretrained"):
        """Interactive viewer for per-frame predictions.

        With `scenario=None` the historical behaviour is kept: the checkpoint and
        test-loader come from `meta.cnn_gnn` and the dataset from the hardcoded
        `if True:` block in visualise_gnn().

        Passing one of VIEWER_SCENARIOS selects the model weights AND the test
        dataset together, so all six canonical scenarios are reachable by name
        instead of by editing source. `weights` is "pretrained" (the published
        checkpoint) or "retrained" (what a local `--train` run wrote).
        """
        self.scenario = scenario
        self.weights = weights
        self.scenario_cfg = None

        if scenario is None:
            if SYSTEM_PARAMS.meta.cnn_gnn == 0:
                self.model_path = SYSTEM_PARAMS.files.final_segmentation_model
                self.test_loader = SYSTEM_PARAMS.files.test_loader
            elif SYSTEM_PARAMS.meta.cnn_gnn == 1:
                self.model_path = SYSTEM_PARAMS.files.final_segmentation_model_gnn
                self.test_loader = SYSTEM_PARAMS.files.test_loader_gnn_sim
            return

        if scenario not in VIEWER_SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; expected one of {list(VIEWER_SCENARIOS)}"
            )
        cfg = VIEWER_SCENARIOS[scenario]
        self.scenario_cfg = cfg
        ckpt_rel = getattr(SYSTEM_PARAMS.files, cfg["ckpt_key"])
        stats_rel = getattr(SYSTEM_PARAMS.files, cfg["stats_key"])
        if weights == "retrained":
            ckpt_rel = _retrained_variant(ckpt_rel, scenario)
            stats_rel = _retrained_variant(stats_rel, scenario)
        self.model_path = repo_path(ckpt_rel)
        self.test_loader = repo_path(stats_rel)
        print(f"=== {scenario} [{weights}]: {cfg['description']} ===")
        print(f"checkpoint:  {self.model_path}")
        print(f"stats:       {self.test_loader}")

    def _build_scenario_dataset(self, all_stats):
        """Dataset for the selected scenario, normalised as the checkpoint expects."""
        cfg = self.scenario_cfg
        if has_flat_stats_dict(all_stats):
            stats, difficulty = all_stats, None
        else:
            difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
            stats = all_stats[difficulty]

        if cfg["test_dataset"] == "silicone":
            full_dataset = MyDataset(
                scheme="single_dataset",
                sim_exp="exp",
                data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
                apply_augmentations=False,
                name="silicone",
            )
            _, _, test_dataset = full_dataset.create_splits(
                train_size=0.0, val_size=0.0, test_size=1.0
            )
            if difficulty is not None:
                test_dataset.set_difficulty_level(difficulty)
        else:
            # Sequential (non-overlapping) clips so that stepping through the
            # viewer walks each trial once from start to finish. The default
            # sliding window starts a clip at every frame, which drops the viewer
            # into the middle of a vein sweep.
            full_dataset = MyDataset(
                scheme="meat",
                sim_exp="apple",
                data_dir="banana",
                apply_augmentations=False,
                meat_sequential_clips=True,
                name="meat",
            )
            _, _, test_dataset = full_dataset.create_splits(all_to_test=True)
        test_dataset.set_stats(stats)
        test_dataset.eval()
        return test_dataset

    def clip_labels(self, sequence_idx, num_clips):
        """The three metadata lines overlaid on the current clip.

        For the meat dataset (C) the clip's identity is read from the dataset
        itself: which trial it came from and which clip within that trial. The
        previous code derived all three lines from `sequence_idx` alone, using a
        "5 trajectories x 2 directions x 10 frames" layout that only ever
        described the simulated dataset. On dataset C - 10 trials cut into a
        variable number of clips each - that produced counts like "trajectory
        9/5", and the direction and frame lines were meaningless.

        The sim layout keeps its original arithmetic.
        """
        dataset = getattr(self, "viewer_dataset", None)
        meat_clips = getattr(dataset, "meat_data", None)
        if meat_clips and sequence_idx < len(meat_clips):
            trial_folder_path, _, video_frame_indices = meat_clips[sequence_idx]
            trial_id = os.path.basename(trial_folder_path)
            # Clips are ordered, so the trial's position and its clip count come
            # from grouping the clip list by trial folder.
            trial_ids = [os.path.basename(c[0]) for c in meat_clips]
            unique_trials = sorted(set(trial_ids))
            trial_no = unique_trials.index(trial_id) + 1
            clips_this_trial = [i for i, t in enumerate(trial_ids) if t == trial_id]
            clip_no = clips_this_trial.index(sequence_idx) + 1
            first, last = video_frame_indices[0], video_frame_indices[-1] + 1
            return (
                f"trial: {trial_no}/{len(unique_trials)} ({trial_id})",
                f"clip: {clip_no}/{len(clips_this_trial)}  frames [{first}, {last})",
                f"clip {sequence_idx + 1}/{num_clips} overall",
            )
        traj_ix = sequence_idx // 20
        direction = 'right' if (sequence_idx % 20) // 10 == 0 else 'left'
        small_frame_ix = sequence_idx % 10
        return (
            f"trajectory: {traj_ix+1}/5",
            f"direction: {direction}",
            f"frame number: {small_frame_ix+1}/10",
        )

    @staticmethod
    def calculate_iou(ground_truth, prediction):
        intersection = np.logical_and(ground_truth, prediction)
        union = np.logical_or(ground_truth, prediction)
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou_score

    @staticmethod
    def create_confusion_matrix_overlay(ground_truth, prediction):
        overlay = np.zeros((*ground_truth.shape, 3))
        
        # True Negative (black)
        tn_mask = (ground_truth == 0) & (prediction == 0)
        overlay[tn_mask] = [0, 0, 0]
        
        # True Positive (white)
        tp_mask = (ground_truth == 1) & (prediction == 1)
        overlay[tp_mask] = [1, 1, 1]
        
        # False Positive (red)
        fp_mask = (ground_truth == 0) & (prediction == 1)
        overlay[fp_mask] = [1, 0, 0]
        
        # False Negative (blue)
        fn_mask = (ground_truth == 1) & (prediction == 0)
        overlay[fn_mask] = [0, 0, 1]
        
        return overlay
    
    def visualize_experiment(
            self,
            mode,
            frame_num=None
        ):
        if mode == 'curved': 
            npz_path = SYSTEM_PARAMS.files.exp_video_npz
            video_path = SYSTEM_PARAMS.files.vein_slide_across_extracted_markers
        elif mode == 'straight':
            npz_path = SYSTEM_PARAMS.files.experiment_straight_markers_npz
            video_path = SYSTEM_PARAMS.files.experiment_straight_processed_video

        self.exp_data = np.load(npz_path)
        
        # Initialize model
        model = SegmentationModel()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Initialize video capture
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return

        # Get parameters needed for clip extraction
        w = SYSTEM_PARAMS.fisheye_model.crop_width
        h = SYSTEM_PARAMS.fisheye_model.crop_height
        k = SYSTEM_PARAMS.fisheye_model.down_scaling_factor
        clip_len = SYSTEM_PARAMS.gnn.clip_len
        dilation = 8
        dilated_clip_len = clip_len * dilation
        n = self.exp_data['markers'].shape[0]
        m = 0

        while m <= n - dilated_clip_len:
            if frame_num is not None:
                start_ix = frame_num
            else:
                # start_ix = NP_RNG.integers(0, n - dilated_clip_len)
                start_ix = m
            
            # Get and process clip
            clip = MyDataset.get_clip(h, w, k, self.exp_data, clip_len, dilation, start_ix=start_ix)
            
            with torch.no_grad():
                clip_input = clip.to(device)
                logits = model(clip_input)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                pred = pred.cpu()
            
            # Convert tensors to numpy arrays
            image_seq = clip.numpy().squeeze()  # Shape: (T, H, W)
            pred_seq = pred.numpy().squeeze()  # Shape: (T, H, W)

            current_frame = 0
            total_frames = image_seq.shape[0]

            # Interactively this steps through frames until 'q'. Non-interactively
            # nobody can press a key, so play a bounded number of frames and move
            # on (override with DIFFTACTILE_MAX_FRAMES).
            frame_limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", total_frames)
            shown = 0

            while frame_limit is None or shown < frame_limit:
                shown += 1
                # Prepare the current frame
                current_image = image_seq[current_frame]
                current_pred = pred_seq[current_frame]
                
                # Normalize images for display
                current_image = (current_image * 255).astype(np.uint8)
                current_pred = (current_pred * 255).astype(np.uint8)

                # Scale up images by 4x using NEAREST neighbor interpolation
                scale_factor = 2
                current_image = cv2.resize(current_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                current_pred = cv2.resize(current_pred, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

                # Add frame counter text
                frame_text = f"Frame: {current_frame + 1}/{total_frames} | Start Index: {start_ix}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # Reduced from 1.0
                font_thickness = 1  # Reduced from 2
                text_color = (255, 255, 255)  # White text
                
                # Get text size to position it at the bottom
                text_size = cv2.getTextSize(frame_text, font, font_scale, font_thickness)[0]
                text_x = 5  # Reduced from 10
                text_y = current_image.shape[0] - 10  # Reduced from 20
                
                # Add black background for text visibility
                padding = 3  # Reduced from 5
                cv2.rectangle(current_image, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                cv2.rectangle(current_pred, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                
                # Add text to both images
                cv2.putText(current_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                cv2.putText(current_pred, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Create display windows
                imshow(cv2, 'Input Image', current_image)
                imshow(cv2, 'Predicted Image', current_pred)

                # Read and display video frame
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_ix + current_frame * dilation)
                ret, video_frame = video_cap.read()
                if ret:
                    # Crop video frame using fisheye model parameters
                    start_x = SYSTEM_PARAMS.fisheye_model.crop_x
                    start_y = SYSTEM_PARAMS.fisheye_model.crop_y
                    crop_width = SYSTEM_PARAMS.fisheye_model.crop_width
                    crop_height = SYSTEM_PARAMS.fisheye_model.crop_height
                    
                    # Crop the frame using the fisheye model parameters
                    scale_factor = 1/2
                    video_frame = video_frame[start_y:start_y+crop_height, start_x:start_x+crop_width]
                    video_frame = cv2.resize(video_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                    
                    # Add the same frame counter text to video frame
                    cv2.rectangle(video_frame, 
                                (text_x - padding, text_y - text_size[1] - padding),
                                (text_x + text_size[0] + padding, text_y + padding),
                                (0, 0, 0), -1)
                    cv2.putText(video_frame, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                    
                    imshow(cv2, 'Video Frame', video_frame)

                # Position windows side by side
                window_width = current_image.shape[1]
                cv2.moveWindow('Input Image', 0, 0)
                cv2.moveWindow('Predicted Image', window_width + 25, 0)
                cv2.moveWindow('Video Frame', (window_width + 25) * 2, 0)

                # Handle keyboard input
                key = wait_key(cv2, 0) & 0xFF

                if key == ord('q'):  # Quit visualization
                    destroy_windows(cv2)
                    video_cap.release()
                    return
                elif key == ord('j'):  # Previous frame
                    current_frame = (current_frame - 1) % total_frames
                elif key == ord('c'):  # Close current sequence and load next
                    destroy_windows(cv2)
                    break
                else:
                    # 'k' advances interactively; with no key press this also
                    # drives the bounded non-interactive loop forward.
                    current_frame = (current_frame + 1) % total_frames
            m += dilated_clip_len
        
        # Clean up
        video_cap.release()

    def visualise(self, mode):
        """
        Unified visualization method that can show either dataset samples or model predictions
        Args:
            mode: Either 'dataset' or 'predictions'
        """
        BATCH_SIZE = 1
        NUM_WORKERS = 1
        if mode == 'predictions':
            with open(self.test_loader, 'rb') as f:
                test_data = pickle.load(f)
            data_loader = DataLoader(
                test_data['dataset'],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS
            )
            
            # Initialize model
            model = SegmentationModel()
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        else:  # dataset mode
            full_dataset = MyDataset(
                data_dir=SYSTEM_PARAMS.files.dataset_root
            )
            train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
                full_dataset, train_size=1.0, val_size=0.0, test_size=0.0
            )
            data_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
            )

        data_iter = iter(data_loader)
        i = 0
        
        while True:  # Main loop for continuous data loading
            try:
                image, label = next(data_iter)
            except StopIteration:
                print("End of dataset reached. Restarting...")
                data_iter = iter(data_loader)
                continue

            if label.sum() == 0:
                continue
            # Handle predictions if in prediction mode
            if mode == 'predictions':
                with torch.no_grad():
                    image_input = image.to(device)
                    logits = model(image_input)
                    probs = torch.sigmoid(logits)
                    pred = (probs > 0.5).float()
                    pred = pred.cpu()

            # Convert tensors to numpy arrays
            image_seq = image.numpy().squeeze()  # Shape: (T, H, W)
            label_seq = label.numpy().squeeze()  # Shape: (T, H, W)
            if mode == 'predictions':
                pred_seq = pred.numpy().squeeze()  # Shape: (T, H, W)

            current_frame = 0
            total_frames = image_seq.shape[0]

            # Bounded when non-interactive; see the note on the browser above.
            frame_limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", total_frames)
            shown = 0

            while frame_limit is None or shown < frame_limit:
                shown += 1
                # Prepare the current frame
                current_image = image_seq[current_frame]
                current_label = label_seq[current_frame]
                if mode == 'predictions':
                    current_pred = pred_seq[current_frame]
                    current_overlay = Visualisation.create_confusion_matrix_overlay(current_label, current_pred)
                    iou_score = Visualisation.calculate_iou(current_label, current_pred)
                
                # Normalize images for display
                current_image = (current_image * 255).astype(np.uint8)
                if mode == 'dataset':
                    current_right = (current_label * 255).astype(np.uint8)
                    
                    # Create binary versions (0 or 255)
                    binary_left = np.where(current_image > 0, 255, 0).astype(np.uint8)
                    binary_right = np.where(current_right > 0, 255, 0).astype(np.uint8)
                    
                    # Create RGB overlay
                    overlay_image = np.zeros((current_image.shape[0], current_image.shape[1], 3), dtype=np.uint8)
                    overlay_image[..., 0] = binary_left  # Red channel for markers
                    overlay_image[..., 1] = binary_right  # Green channel for ground truth
                else:  # predictions mode
                    # Convert overlay from float [0,1] to uint8 [0,255]
                    current_right = (current_overlay * 255).astype(np.uint8)

                # Scale up images by 4x using NEAREST neighbor interpolation
                scale_factor = 2
                current_image = cv2.resize(current_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                current_right = cv2.resize(current_right, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                if mode == 'dataset':
                    overlay_image = cv2.resize(overlay_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

                # Add frame counter text and other information
                frame_text = f"Frame: {current_frame + 1}/{total_frames}"
                if mode == 'predictions':
                    frame_text += f" | IoU: {iou_score:.3f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_color = (255, 255, 255)  # White text
                
                # Get text size to position it at the bottom
                text_size = cv2.getTextSize(frame_text, font, font_scale, font_thickness)[0]
                text_x = 10
                text_y = current_image.shape[0] - 20  # 20 pixels from bottom
                
                # Add black background for text visibility
                padding = 5
                cv2.rectangle(current_image, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                cv2.rectangle(current_right, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                if mode == 'dataset':
                    cv2.rectangle(overlay_image, 
                                (text_x - padding, text_y - text_size[1] - padding),
                                (text_x + text_size[0] + padding, text_y + padding),
                                (0, 0, 0), -1)
                
                # Add text to images
                cv2.putText(current_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                cv2.putText(current_right, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                if mode == 'dataset':
                    cv2.putText(overlay_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Create display windows
                imshow(cv2, f'Input Image {i}', current_image)
                right_window_title = 'Ground Truth Label' if mode == 'dataset' else 'Prediction Overlay'
                imshow(cv2, f'{right_window_title} {i}', current_right)
                if mode == 'dataset':
                    imshow(cv2, f'Overlay {i}', overlay_image)

                # Get screen dimensions using cv2
                window_width = current_image.shape[0]
                
                # Position windows - left window at (0,0), middle at (window_width + 25, 0), right at (2 * window_width + 50, 0)
                cv2.moveWindow(f'Input Image {i}', 0, 0)
                cv2.moveWindow(f'{right_window_title} {i}', window_width + 25, 0)
                if mode == 'dataset':
                    cv2.moveWindow(f'Overlay {i}', 2 * window_width + 50, 0)

                # Handle keyboard input
                key = wait_key(cv2, 0) & 0xFF

                if key == ord('q'):  # Quit visualization
                    destroy_windows(cv2)
                    return
                elif key == ord('j'):  # Previous frame
                    current_frame = (current_frame - 1) % total_frames
                elif key == ord('c'):  # Close current sequence and load next
                    i += 1
                    destroy_windows(cv2)
                    break
                else:
                    # 'k' advances interactively; with no key press (the
                    # non-interactive case) this also steps the bounded loop on.
                    current_frame = (current_frame + 1) % total_frames
    
    def visualise_gnn(self, mode, data_source):
        """
        Visualize GNN predictions and ground truth segmentation masks.
        Shows images per frame:
        For mode='predictions':
            1. Ground truth labels (red = 0, green = 1) - only shown for central frame
            2. Hard predicted labels (red = 0, green = 1) - only shown for central frame
            3. Soft predicted labels (color intensity shows confidence) - only shown for central frame
            4. Original labels image - shown for all frames
            5. Graph connectivity visualization - shown for all frames
        For mode='dataset':
            1. Ground truth labels (red = 0, green = 1) - only shown for central frame
            2. Original labels image - shown for all frames
            3. Graph connectivity visualization - shown for all frames
        Args:
            mode: Either 'dataset' or 'predictions'
        """
        BATCH_SIZE = 1
        NUM_WORKERS = 1
        LABELS_DOWNSIZE = 4
        MARKER_SIZE = 10
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        adjacency_matrix = base_graph_data['adjacency_matrix']

        if mode == 'predictions':
            # Initialize model. A named scenario dictates the architecture, since
            # a checkpoint only loads into the one it was trained with.
            if self.scenario_cfg is not None:
                model = _segmentation_gnn(self.scenario_cfg["arch"])
            else:
                model = GNN()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model.eval()
            model = model.to(device)

        # Load test data
        with open(self.test_loader, 'rb') as f:
            test_data = pickle.load(f)
        all_stats = test_data['dataset_stats']

        if data_source == 'pickled_test_dataset':
            dataset = test_data['dataset']
            dataset.eval()
            data_loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS
            )
        elif data_source == 'fresh_dataset':  # dataset mode
            # A named scenario selects the dataset (and its normalisation) to
            # match the checkpoint; otherwise fall through to the historical
            # hardcoded if True/if False toggle below.
            if self.scenario_cfg is not None:
                test_dataset = self._build_scenario_dataset(all_stats)
                # Kept so clip_labels() can name the trial each clip came from.
                self.viewer_dataset = test_dataset
                data_loader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )
            elif True:
                full_dataset = MyDataset(
                    scheme="single_dataset",
                    sim_exp="exp",
                    data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
                    apply_augmentations=False,
                )
            if False:
                full_dataset = MyDataset(
                    scheme="single_dataset",
                    sim_exp="sim",
                    data_dir=SYSTEM_PARAMS.files.sim_data,
                    apply_augmentations=True,
                )
            # Legacy path only: the scenario branch above has already built and
            # normalised its dataset, and would be overwritten by this block
            # (which also assumes a difficulty-keyed stats dict that the meat
            # loader does not have).
            if self.scenario_cfg is None:
                _, _, test_dataset = full_dataset.create_splits(
                    train_size=0.0,
                    val_size=0.0,
                    test_size=1.0
                )
                target_difficulty = 1.0
                stats = all_stats[target_difficulty]
                test_dataset.set_stats(stats)
                test_dataset.set_difficulty_level(target_difficulty)
                test_dataset.eval()
                data_loader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
                )
        elif data_source == 'exp_npz':
            exp_test_dataset_grid_search = MyDataset(
                mode='exp',
                exp_markers_npz=SYSTEM_PARAMS.files.experiment_og_markers_reordered_npz,
                exp_ground_truth_labels_npz=SYSTEM_PARAMS.files.experiment_og_ground_truth_labels_npz,
                exp_dilation=2,
                scheme='new',
            )
            exp_test_dataset_straight_line_slide = MyDataset(
                mode='exp',
                exp_markers_npz=SYSTEM_PARAMS.files.experiment_straight_markers_reordered_npz,
                exp_ground_truth_labels_npz=SYSTEM_PARAMS.files.experiment_straight_ground_truth_labels_npz,
                exp_dilation=2,
                scheme='new',
            )
            dataset = exp_test_dataset_grid_search
            target_difficulty = 1.0
            stats = all_stats[target_difficulty]
            dataset.set_stats(stats)
            dataset.set_difficulty_level(target_difficulty)
            data_loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
            )

        data_iter = iter(data_loader)
        data_list = list(data_iter)
        gt = []
        sp = []
        hp = []
        co = []
        meta = []

        for sequence_idx in range(len(data_list)):  # Main loop for continuous data loading
            try:
                # batch, labels_images, poses, metadata, frame_ix = next(data_iter)
                batch, labels_images, poses, metadata, frame_ix = data_list[sequence_idx]
                poses = poses.numpy()[0]
                metadata = metadata.numpy()[0]
                frame_ix = frame_ix.item()
                # if not (
                #     metadata[1] == 0 
                #     and metadata[0] == 0 
                #     and frame_ix == 0
                # ):
                #     continue
            except StopIteration:
                print("End of dataset reached. Restarting...")
                data_iter = iter(data_loader)
                continue
            
            ground_truth_labels_present = labels_images.numel() != 0
            
            num_frames = SYSTEM_PARAMS.gnn.clip_len
            if ground_truth_labels_present:
                labels_images = labels_images.numpy()[0, ...]
                labels_h = labels_images.shape[1] // LABELS_DOWNSIZE
                labels_w = labels_images.shape[2] // LABELS_DOWNSIZE
            else:
                labels_h = 270
                labels_w = 480
                labels_images = np.zeros((num_frames, labels_h * LABELS_DOWNSIZE, labels_w * LABELS_DOWNSIZE), dtype=np.uint8)

            # Get number of frames from the mask
            num_nodes_per_frame = SYSTEM_PARAMS.vitactip.num_markers
            central_frame = num_frames // 2

            # Pre-compute image dimensions
            h, w = 400, 400
            MARKER_SIZE = 6
            MARKER_RADIUS = MARKER_SIZE // 2

            # Initialize image stacks for each color channel with white background
            metadata_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
            ground_truth_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
            prediction_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
            soft_prediction_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)  # New stack for soft predictions
            labels_stack = np.zeros((num_frames, labels_h, labels_w, 3), dtype=np.uint8)
            graph_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)  # New stack for graph visualization
            confusion_matrix_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)  # New stack for confusion matrix visualization
            stats_stack = np.zeros((num_frames, 200, 400, 3), dtype=np.uint8) + 255  # White background for stats display

            if mode == 'predictions':
                # Get predictions
                with torch.no_grad():
                    batch = batch.to(device)
                    x, x_mask, edge_index, edge_index_regular_nodes, edge_attr = model.my_prepare_data(batch, batch.num_graphs)
                    out = model(x, edge_index, edge_attr, batch.batch)
                    out = out.squeeze(-1)  # Remove the channel dimension
                    out = out[x_mask]
                    # mask = data.mask
                    # out = out[mask]
                    probs = torch.sigmoid(out)
                    pred = (probs > 0.58).float()

                    assert batch.num_graphs == 1
                    
                    # Compute IoU scores per frame
                    num_nodes_per_frame = SYSTEM_PARAMS.vitactip.num_markers
                    clip_stats = []
                    for frame_idx in range(num_frames):
                        start_idx = frame_idx * num_nodes_per_frame
                        end_idx = (frame_idx + 1) * num_nodes_per_frame
                        frame_pred = pred[start_idx:end_idx]
                        frame_truth = batch.y[start_idx:end_idx]
                        frame_metrics = GNN.iou_score(frame_pred, frame_truth)
                        fg_iou = frame_metrics[1]
                        bg_iou = frame_metrics[0]
                        
                        # Compute confusion matrix
                        frame_pred = pred[start_idx:end_idx].cpu().numpy()
                        frame_truth = batch.y[start_idx:end_idx].cpu().numpy()
                        tp = np.sum((frame_pred == 1) & (frame_truth == 1))
                        tn = np.sum((frame_pred == 0) & (frame_truth == 0))
                        fp = np.sum((frame_pred == 1) & (frame_truth == 0))
                        fn = np.sum((frame_pred == 0) & (frame_truth == 1))
                        clip_stats.append({
                            'fg_iou': fg_iou,
                            'bg_iou': bg_iou,
                            'tp': tp,
                            'tn': tn,
                            'fp': fp,
                            'fn': fn,
                        })
                    
                    probs = probs.cpu().numpy().astype(np.float32)
                    pred = pred.cpu().numpy().astype(int)

            # if data.y.cpu().numpy().sum() == 0 or pred.sum() == 0:
            #     continue

            # Pre-compute all frames
            for frame_idx in range(num_frames):
                # Get marker positions for current frame
                start_idx = frame_idx * num_nodes_per_frame
                end_idx = (frame_idx + 1) * num_nodes_per_frame
                frame_points = batch.pos[start_idx:end_idx].cpu().numpy()[:, :2]
                
                # Transform from (-1,1) to (0,200) range
                points = (frame_points + 3) / 6 * w  # Now in range (0,200)
                points = points.astype(np.float32)  # Keep as float for draw_point
                
                # Create graph connectivity visualization
                # _, points, adjacency_matrix = Adjacency.get_graph_connectivity(points)
                graph_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Draw edges from adjacency matrix in green
                for edge in adjacency_matrix:
                    start_idx, end_idx = edge
                    start_point = tuple(map(int, points[start_idx]))
                    end_point = tuple(map(int, points[end_idx]))
                    cv2.line(graph_img, start_point, end_point, color=(0, 255, 0), thickness=1)
                
                # Draw points in red
                for point in points:
                    x, y = map(int, point)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(graph_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                
                graph_stack[frame_idx] = graph_img

                # Get predictions for current frame
                start_idx = frame_idx * num_nodes_per_frame
                end_idx = (frame_idx + 1) * num_nodes_per_frame
                ground_truth = batch.y[start_idx:end_idx].cpu().numpy()
                
                # Draw markers on ground truth image
                for point_idx, point in enumerate(points):
                    if 0 <= point[0] < w and 0 <= point[1] < h:
                        center = (int(point[0]), int(point[1]))
                        if ground_truth[point_idx] == 1:
                            # Magenta (BGR = (255, 0, 255)) for positive class
                            cv2.circle(ground_truth_stack[frame_idx], center, MARKER_RADIUS, (255, 0, 255), -1, cv2.LINE_AA)
                        else:
                            # Cyan (BGR = (255, 255, 0)) for negative class
                            cv2.circle(ground_truth_stack[frame_idx], center, MARKER_RADIUS, (255, 255, 0), -1, cv2.LINE_AA)
                
                if mode == 'predictions':
                    frame_pred = pred[start_idx:end_idx]
                    frame_probs = probs[start_idx:end_idx]
                    
                    # Draw markers on prediction image (hard predictions)
                    for point_idx, point in enumerate(points):
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            center = (int(point[0]), int(point[1]))
                            if frame_pred[point_idx] == 1:
                                # Magenta (BGR = (255, 0, 255)) for positive class
                                cv2.circle(prediction_stack[frame_idx], center, MARKER_RADIUS, (255, 0, 255), -1, cv2.LINE_AA)
                            else:
                                # Cyan (BGR = (255, 255, 0)) for negative class
                                cv2.circle(prediction_stack[frame_idx], center, MARKER_RADIUS, (255, 255, 0), -1, cv2.LINE_AA)
                    
                    # Draw markers on soft prediction image
                    for point_idx, point in enumerate(points):
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            center = (int(point[0]), int(point[1]))
                            prob = frame_probs[point_idx]
                            intensity = int(255 * prob)  # Scale to [0,255]
                            # Use white color with varying intensity for all points
                            cv2.circle(soft_prediction_stack[frame_idx], center, MARKER_RADIUS, (intensity, intensity, intensity), -1, cv2.LINE_AA)
                    
                    # Draw confusion matrix visualization
                    for point_idx, point in enumerate(points):
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            center = (int(point[0]), int(point[1]))
                            pred_val = frame_pred[point_idx]
                            true_val = ground_truth[point_idx]
                            
                            # Color coding:
                            # TP: Lime Green (50, 205, 50)
                            # TN: Yellow (255, 255, 0)
                            # FP: Red (255, 0, 0)
                            # FN: Bright Blue (0, 0, 255)
                            if pred_val == 1 and true_val == 1:  # TP
                                color = (50, 205, 50)
                            elif pred_val == 0 and true_val == 0:  # TN
                                color = (0, 255, 255)  # BGR format
                            elif pred_val == 1 and true_val == 0:  # FP
                                color = (0, 0, 255)
                            else:  # FN
                                color = (255, 0, 0)
                            
                            cv2.circle(confusion_matrix_stack[frame_idx], center, MARKER_RADIUS, color, -1, cv2.LINE_AA)
                    
                    # Draw statistics for current frame
                    stats_img = stats_stack[frame_idx]
                    frame_stats = clip_stats[frame_idx]
                    
                    # Define text positions and font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    line_spacing = 30
                    x_pos = 20
                    y_pos = 40
                    
                    # Draw statistics text
                    cv2.putText(stats_img, f"Foreground IoU: {frame_stats['fg_iou']:.3f}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"Background IoU: {frame_stats['bg_iou']:.3f}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"True Positives: {frame_stats['tp']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"True Negatives: {frame_stats['tn']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"False Positives: {frame_stats['fp']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"False Negatives: {frame_stats['fn']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)

                # Get and process labels image for current frame
                if ground_truth_labels_present:
                    labels_image = labels_images[frame_idx]
                    # Convert labels_image to BGR for visualization
                    labels_display = np.zeros((labels_image.shape[0], labels_image.shape[1], 3), dtype=np.uint8)
                    # Convert torch tensor to numpy and scale back to [0, 255]
                    labels_np = (labels_image * 255).astype(np.uint8)
                    labels_display[..., 0] = labels_np  # Set blue channel
                    labels_display[..., 1] = labels_np  # Set green channel
                    labels_display[..., 2] = labels_np  # Set red channel

                    # Downscale by factor of 4 using INTER_AREA interpolation
                    labels_stack[frame_idx] = cv2.resize(labels_display, (labels_w, labels_h), interpolation=cv2.INTER_AREA)
                else:
                    # Keep the black image for labels_stack when no ground truth is present
                    pass

            # Display loop
            current_frame = SYSTEM_PARAMS.gnn.clip_len // 2

            meta_cur = metadata_stack[current_frame]

            text1, text2, text3 = self.clip_labels(sequence_idx, len(data_list))
            org1 = (10, 50)
            org2 = (10, 100)
            org3 = (10, 150)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 3

            cv2.putText(meta_cur, text1, org1, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(meta_cur, text2, org2, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(meta_cur, text3, org3, font, font_scale, color, thickness, cv2.LINE_AA)

            meta.append(meta_cur)
            gt.append(ground_truth_stack[current_frame])
            sp.append(soft_prediction_stack[current_frame])
            hp.append(prediction_stack[current_frame])
            co.append(confusion_matrix_stack[current_frame])

        def filter_right(arr):
            return [x for i in range(0, len(arr), 20) for x in arr[i:i+10]]

        def filter_left(arr):
            return [x for i in range(0, len(arr), 20) for x in arr[i+10:i+20]]
        
        meta = filter_left(meta)
        gt = filter_left(gt)
        sp = filter_left(sp)
        hp = filter_left(hp)
        co = filter_left(co)

        need_to_wait_3 = True
        for sequence_idx_3 in range(len(meta)):
            imshow(cv2, 'Ground Truth', gt[sequence_idx_3])
            if mode == 'predictions':
                imshow(cv2, 'Hard Prediction', hp[sequence_idx_3])
                imshow(cv2, 'Confusion Matrix', co[sequence_idx_3])
                imshow(cv2, 'Soft Prediction', sp[sequence_idx_3])
                imshow(cv2, 'Metadata', meta[sequence_idx_3])
            
            sep_w = 20
            sep_h = 130
            move_window(cv2, 'Ground Truth', 0, 0)
            if mode == 'predictions':
                move_window(cv2, 'Hard Prediction', w + sep_w, 0)
                move_window(cv2, 'Confusion Matrix', 0, h + sep_h)
                move_window(cv2, 'Soft Prediction', w + sep_w, h + sep_h)
                move_window(cv2, 'Metadata', 2 * (w + sep_w), 0)
            
            if need_to_wait_3:
                # first frame delay
                wait_key(cv2, 4000)
                need_to_wait_3 = False

            # delay between frames. wait_key() caps these when non-interactive,
            # so an unattended run does not spend minutes sleeping on playback.
            if sequence_idx_3 % 10 == 9:
                foo = 2_000
            else:
                foo = 500
            if wait_key(cv2, foo) & 0xFF == ord('q'):  # wait 0.5s, press 'q' to quit
                break

        wait_key(cv2, 4000)
        destroy_windows(cv2)
        return

        sequence_idx_2 = 0
        need_to_wait = True
        # Bounded when non-interactive: there is no 'q' key press to end it.
        limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", len(meta))
        shown = 0
        while limit is None or shown < limit:
            shown += 1
            # Show images from pre-computed stacks
            # title_prefix = f'Frame {current_frame}/{num_frames-1} '
            # title_prefix += '(Central Frame) ' if current_frame == central_frame else ''
            
            imshow(cv2, f'Ground Truth', gt[sequence_idx_2])
            if mode == 'predictions':
                imshow(cv2, f'Hard Prediction', hp[sequence_idx_2])
                imshow(cv2, f'Confusion Matrix', co[sequence_idx_2])
                imshow(cv2, f'Soft Prediction', sp[sequence_idx_2])
                imshow(cv2, f'Metadata', meta[sequence_idx_2])
                # imshow(cv2, f'Frame Statistics', stats_stack[current_frame])
            # imshow(cv2, f'Labels Image', labels_stack[current_frame])
            # imshow(cv2, f'Graph Connectivity', graph_stack[current_frame])

            # Position windows side by side
            sep_w = 20
            sep_h = 130
            cv2.moveWindow(f'Ground Truth', 0, 0)
            if mode == 'predictions':
                cv2.moveWindow(f'Hard Prediction', w + sep_w, 0)
                cv2.moveWindow(f'Confusion Matrix', 0, h + sep_h)
                cv2.moveWindow(f'Soft Prediction', w + sep_w, h + sep_h)
                cv2.moveWindow(f'Metadata', 2*(w + sep_w), 0)
                # cv2.moveWindow(f'Labels Image', 2 * (w + sep_w), h + sep_h)
                # cv2.moveWindow(f'Graph Connectivity', 3 * (w + sep_w), 0)
                # cv2.moveWindow(f'Frame Statistics', 0, h + sep_h)
            else:
                # cv2.moveWindow(f'Labels Image', w + sep_w, 0)
                # cv2.moveWindow(f'Graph Connectivity', w + sep_w, labels_h + sep_h)
                pass
                
            if need_to_wait:
                # Only worth pausing for a human who is looking at the window.
                if is_interactive():
                    time.sleep(10)
                need_to_wait = False

            # Handle keyboard input
            key = wait_key(cv2, 0) & 0xFF

            if key == ord('q'):  # Quit visualization
                destroy_windows(cv2)
                return
            elif key == ord('x'):
                sequence_idx_2 -= 1
                sequence_idx_2 = max(min(sequence_idx_2, len(meta)-1), 0)
                # destroy_windows(cv2)

            elif key == ord('c'):  # Close current sequence and load next
                sequence_idx_2 += 1
                sequence_idx_2 = max(min(sequence_idx_2, len(meta)-1), 0)
                # destroy_windows(cv2)
                # break
            # elif key == ord('j'):  # Previous frame
            #     current_frame = max(0, current_frame - 1)
            # elif key == ord('k'):  # Next frame
            #     current_frame = min(num_frames - 1, current_frame + 1)
            elif key == ord('d'):
                foo = 7
            else:
                # No key press (non-interactive): step forward so the bounded
                # loop walks the sequence instead of redrawing frame 0.
                sequence_idx_2 = min(sequence_idx_2 + 1, len(meta) - 1)

        destroy_windows(cv2)

    def test_data_loader(self):
        BATCH_SIZE = 16
        NUM_WORKERS = 16
        full_dataset = MyDataset(
            data_dir=SYSTEM_PARAMS.files.dataset_root
        )
        train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
            full_dataset, train_size=1.0, val_size=0.0, test_size=0.0
        )
        data_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )

        data_iter = iter(data_loader)
        i = 0
        pbar = tqdm()
        while True:
            try:
                image, label = next(data_iter)
            except StopIteration:
                print("End of dataset reached. Restarting...")
                break
            i += 1
            pbar.update(1)
            pbar.set_description(f"Processed {i} batches")
        pbar.close()
    
    def graph(self):
        full_dataset = MyDataset(
            data_dir=SYSTEM_PARAMS.files.dataset_root
        )
        train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
            full_dataset, train_size=0.1, val_size=0.0, test_size=0.0
        )
        num_clips = len(train_dataset)
        visualise = False
        
        # Lists to store features from all frames
        all_node_features = []
        all_edge_features = []
        
        for i in range(num_clips):
            points = train_dataset.get_markers(i)
            adjacency = Visualisation.compute_knn_adjacency(points)
            
            # Compute node and edge features
            node_features, edge_features = Visualisation.compute_graph_features(points, adjacency)
            
            # Store features
            all_node_features.append(node_features)
            all_edge_features.append(edge_features)
            
            if visualise:
                should_break = Visualisation.visualise_adjacency_graph(points, adjacency)
                if should_break:
                    break
        
        plt.close('all')  # Close any remaining figures
        return all_node_features, all_edge_features

    @staticmethod
    def visualise_adjacency_graph(points, adjacency):
        # Create a new figure for each frame
        plt.figure(figsize=(10, 10))
        
        # Plot edges first (connections between points)
        for node_idx in range(len(points)):
            # Get the neighbors for this node
            neighbors = adjacency[node_idx]
            # Draw lines from this node to all its neighbors
            for neighbor_idx in neighbors:
                plt.plot([points[node_idx, 0], points[neighbor_idx, 0]],
                        [points[node_idx, 1], points[neighbor_idx, 1]],
                        'gray', alpha=0.5, linewidth=1)
        
        # Plot nodes (points)
        plt.scatter(points[:, 0], points[:, 1], c='red', s=50)
        
        plt.title(f'Frame {i+1}: K-Nearest Neighbors Graph (k=6)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Make the plot aspect ratio equal
        plt.axis('equal')
        
        # Display the plot
        plt.draw()
        plt.pause(0.1)  # Add a small pause to allow for visualization

        # Wait for key press to continue. prompt() returns "" immediately unless
        # DIFFTACTILE_INTERACTIVE=1, so an unattended run is never stuck here.
        key = prompt("Press Enter to continue to next frame, or 'q' to quit: ")
        if key.lower() == 'q':
            return True
        
        plt.close()  # Close the current figure before showing the next one
        return False


def main():
    """Open the interactive per-frame prediction viewer.

    With no arguments the historical behaviour is kept. Pass one of the three
    configurations to select the checkpoint and the test dataset together:

        python -m difftactile.scripts.script_visualise A-to-B
        python -m difftactile.scripts.script_visualise A-to-C --retrained

    The configuration may also come from DIFFTACTILE_SCENARIO and the weight
    source from DIFFTACTILE_WEIGHTS (pretrained | retrained).
    """
    argv = sys.argv[1:]
    flags = [a for a in argv if a.startswith("--")]
    positional = [a for a in argv if not a.startswith("--")]

    scenario = positional[0] if positional else os.environ.get("DIFFTACTILE_SCENARIO")
    if "--retrained" in flags:
        weights = "retrained"
    elif "--pretrained" in flags:
        weights = "pretrained"
    else:
        weights = os.environ.get("DIFFTACTILE_WEIGHTS", "pretrained")

    v = Visualisation(scenario=scenario, weights=weights)
    v.visualise_gnn(
        mode='predictions',
        data_source='fresh_dataset'
    )


if __name__ == "__main__":
    main()
