import math
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.spatial.distance import pdist

from difftactile.cnn.common import Common
from difftactile.cnn.dataset import *
from difftactile.cnn.gnn import *
from difftactile.cnn.visualise import *
from difftactile.data_analysis.experiment.hungarian_exp import *
from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.main.constants import *
from difftactile.main.synthetic_image_generator import SyntheticImageGenerator
from difftactile.sensor_model.fisheye_model_no_taichi import *


class PredictExp:
    def __init__(self):
        self.fisheye_model = FisheyeModelNoTaichi()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_min = -0.425
        self.x_max = -0.245
        self.y_min = -0.054
        self.y_max = 0.046

        self.phantom_length_x = abs(self.x_max-self.x_min)
        self.phantom_length_y = abs(self.y_max-self.y_min)

        self.clip_len = SYSTEM_PARAMS.gnn.clip_len
        self.dilation = 1
        self.dilated_clip_len = self.clip_len * self.dilation
        self.dataset = MyDataset(
            scheme="single_dataset",
            sim_exp="exp",
            data_dir=SYSTEM_PARAMS.files.exp_data_endgame,
            normalise_pos=False,
            apply_augmentations=False,
        )
        with open(SYSTEM_PARAMS.files.test_loader_gnn, 'rb') as f:
            test_data = pickle.load(f)
        self.stats = test_data['dataset_stats'][1.0]
        self.dataset.set_stats(self.stats)
        self.dataset.set_difficulty_level(1.0)

        self.init_model()
        self.init_camera_params()
        self.compute_mapping_2d_3d()

        self.bin_size = 0.0001
        self.bin_num_x = math.ceil(self.phantom_length_x / self.bin_size)
        self.bin_num_y = math.ceil(self.phantom_length_y / self.bin_size)
        self.bins = np.zeros(shape=(2, self.bin_num_x, self.bin_num_y), dtype=int)
        self.bin_pred = np.zeros(shape=(self.bin_num_x, self.bin_num_y), dtype=float)
        self.bin_trajectory = np.zeros(shape=(self.bin_num_x, self.bin_num_y), dtype=int)
        self.bins_ground_truth = np.zeros(shape=(self.bin_num_x, self.bin_num_y), dtype=int)
        self.debug_bins = np.zeros(shape=(len(self.dataset)), dtype=int)

        self.ground_truth_img_path = SYSTEM_PARAMS.files.phantom_ground_truth_segmentation_mask
        self.ground_truth_img_downsampled_path = SYSTEM_PARAMS.files.ground_truth_labels_downsampled
        self.prediction_img_path = SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask
        self.sensor_trajectory_img_path = SYSTEM_PARAMS.files.sensor_trajectory
        self.ground_truth_feasible_img_path = SYSTEM_PARAMS.files.ground_truth_feasible
        self.ground_truth_from_video_img_path = SYSTEM_PARAMS.files.experiment_ground_truth_from_video_img_path

        self.set_up_keypoints()

    def set_up_keypoints(self):

        visible_vein_frame_ixs_original = np.array([])
        
        visible_vein_frame_ixs = np.array([
            [self.markers_index_mapping[frame_ix], marker_ix] 
            for frame_ix, marker_ix in visible_vein_frame_ixs_original 
            if frame_ix in self.markers_index_mapping
        ])
        assert visible_vein_frame_ixs_original.shape == visible_vein_frame_ixs.shape
        self.visible_vein_frame_ixs = visible_vein_frame_ixs
    
    def z_unnormalise(self, points):
        x_mean = self.stats['x_mean']
        x_std = self.stats['x_std']
        y_mean = self.stats['y_mean']
        y_std = self.stats['y_std']
        points[:, 0] = (points[:, 0] * x_std) + x_mean
        points[:, 1] = (points[:, 1] * y_std) + y_mean
        return points
    
    def camera_to_world_transform(
        self,
        x_robot: float, 
        y_robot: float, 
        z_robot: float, 
        d: float
    ) -> np.ndarray:
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=float)
        t = np.array([x_robot, y_robot, z_robot + d], dtype=float)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def compute_mapping_2d_3d(self):
        pixel_coords = np.zeros((self.camera_w_big, self.camera_h_big, 2), dtype=np.float32)
        for i in range(self.camera_w_big):
            for j in range(self.camera_h_big):
                pixel_coords[i, j, :] = np.array([i, j])
        points_E = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
            ps=pixel_coords,
            dist_lens_to_plane=0.019-0.003,
        )
        self.map_2d_3d = points_E

    def init_model(self):
        model_path = SYSTEM_PARAMS.files.final_segmentation_model_gnn
        model = GNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.model = model
        self.device = device
    
    def init_camera_params(self):
        self.camera_w_big = 1920
        self.camera_h_big = 1080

    @staticmethod
    def write_video_to_npz_file(marker_tracker, path):
        n = len(marker_tracker.frame_markers)
        markers_array, markers_mask = SyntheticImageGenerator.create_padded_array_with_mask(marker_tracker.frame_markers)
        vein_data = [np.array([]) for i in range(n)]
        vein_polyline, vein_polyline_mask = SyntheticImageGenerator.create_padded_array_with_mask(vein_data)
        target_id_array = np.zeros(shape=(0, 1), dtype=int)
        np.savez(
            path,
            markers=markers_array,
            markers_mask=markers_mask,
            vein_polyline=vein_polyline,
            vein_polyline_mask=vein_polyline_mask,
            target_id_array=target_id_array
        )
    
    def predict_clip(self, i):
        pyg, _, poses, metadata, frame_ix = self.dataset[i]
        poses = poses.numpy()
        metadata = metadata.numpy()
        frame_ix = frame_ix.item()
        poses /= 1_000
        metadata[2:] = metadata[2:] / 1_000
        if not (
            metadata[1] == 0
            # and metadata[0] == 0 
            # and frame_ix == 0
        ):
            return
        with torch.no_grad():
            pyg = pyg.to(self.device)
            x_px, x_mask, edge_index, edge_index_regular_nodes, edge_attr = self.model.my_prepare_data(pyg, 1)
            out = self.model(x_px, edge_index, edge_attr)
            out = out.squeeze(-1)
            out = out[x_mask]

            mask = pyg.mask
            out = out[mask]
            y_px = pyg.y[mask]
            pos = pyg.pos[mask]

            probs = torch.sigmoid(out)
            preds = (probs > 0.9).float()
            probs = probs.cpu().numpy().astype(np.float32)
            preds = preds.cpu().numpy().astype(np.float32)
        points = pos.cpu().numpy().astype(np.float32)
        labels = y_px.cpu().numpy().astype(int)
        points = points.reshape((127, 2))
        labels = labels.reshape((127,))
        probs = probs.reshape((127,))
        preds = preds.reshape((127,))

        assert points.min() > 0
        assert points.max() > 100

        robot_pos = poses[self.clip_len//2, :3]
        x_robot, y_robot, z_robot = robot_pos
        t_EA = self.camera_to_world_transform(
            x_robot,
            y_robot,
            z_robot,
            d=0.019-0.003,
        )
        for j in range(127):
            prob = probs[j]
            pred = preds[j]
            x_px, y_px = points[j]
            label = labels[j]

            pos_E = self.map_2d_3d[int(x_px), int(y_px)]
            pos_E_homogeneous = np.append(pos_E, 1)
            pos_A_homogeneous = t_EA @ pos_E_homogeneous
            pos_A = pos_A_homogeneous[:3]
            x_A = pos_A[0]
            y_A = pos_A[1]
            x_A -= self.x_min
            y_A -= self.y_min
            succ = (
                x_A >= 0 and
                x_A < self.phantom_length_x and
                y_A >= 0 and
                y_A < self.phantom_length_y
            )
            if not succ:
                continue
            x_A = int(x_A / self.bin_size)
            y_A = int(y_A / self.bin_size)

            self.bin_trajectory[x_A, y_A] += (1 if j == 0 else 0)
            self.bin_pred[x_A, y_A] += pred
            if label == 1:
                self.bins_ground_truth[x_A, y_A] += 1

    def predict_all_clips(self):
        n = len(self.dataset)
        for i in (tqdm(range(0, n-self.dilated_clip_len), desc="clip inference")):
            self.predict_clip(i)
        
        self.write_probs_to_npz()
    
    def debug(self):
        points = self.poses_interpolated[:, :3]
        path = SYSTEM_PARAMS.files.experiment_2025_08_22_filtered_positions_npz
        np.savez(
            path,
            points=points,
        )

    @staticmethod
    def transform_image(img):
        # img = img.T
        # img = img[::-1, :]
        # img = img[:, ::-1]
        return img
    
    def write_probs_to_npz(self):
        path = SYSTEM_PARAMS.files.exp_probs_npz
        self.bin_pred = PredictExp.transform_image(self.bin_pred)
        self.bin_trajectory = PredictExp.transform_image(self.bin_trajectory)
        self.bins_ground_truth = PredictExp.transform_image(self.bins_ground_truth)
        np.savez(
            path,
            bin_prob_sum=self.bin_pred,
            bin_trajectory=self.bin_trajectory,
            bins_ground_truth=self.bins_ground_truth
        )
    
    def load_probs_from_npz(self):
        path = SYSTEM_PARAMS.files.exp_probs_npz
        data = np.load(path)
        self.bin_pred = data['bin_prob_sum']
        self.bin_trajectory = data['bin_trajectory']
        self.bins_ground_truth = data['bins_ground_truth']
    
    def generate_mask_image(self):
        img = PredictExp.process_img2(self.bin_pred)
        cv2.imwrite(self.prediction_img_path, img)

        img = PredictExp.process_img2(self.bin_trajectory)
        cv2.imwrite(self.sensor_trajectory_img_path, img)

        img = PredictExp.process_img2(self.bins_ground_truth)
        cv2.imwrite(self.ground_truth_from_video_img_path, img)
    
    @staticmethod
    def process_img2(img):
        img = (img > 0).astype(np.int32)
        img = (img * 255).astype(np.uint8)
        img = img.T
        # img = np.flip(img, axis=0)
        img = np.flip(img, axis=1)
        return img
    
    @staticmethod
    def compute_npz_grid_search_og():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_og_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_og_processed_video,
            npz_in=SYSTEM_PARAMS.files.experiment_og_markers_npz,
            npz_out=SYSTEM_PARAMS.files.experiment_og_markers_reordered_npz,
            frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_og_frame_mapping_npz,
            video_from_cache=True,
            npz_temp=SYSTEM_PARAMS.files.experiment_og_markers_reordered_npz,
            labels_out=SYSTEM_PARAMS.files.experiment_og_ground_truth_labels_npz,
            labels_in=SYSTEM_PARAMS.files.experiment_og_ground_truth_labels_npz,
        )
    
    @staticmethod
    def compute_npz_straight():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_straight_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_straight_processed_video,
            npz_in=SYSTEM_PARAMS.files.experiment_straight_markers_npz,
            npz_out=SYSTEM_PARAMS.files.experiment_straight_markers_reordered_npz,
            frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_straight_frame_mapping_npz,
            video_from_cache=False,
            npz_temp=SYSTEM_PARAMS.files.experiment_straight_markers_reordered_npz,
            labels_out=SYSTEM_PARAMS.files.experiment_straight_ground_truth_labels_npz,
            labels_in=SYSTEM_PARAMS.files.experiment_straight_ground_truth_labels_npz,
            seconds_per_frame=1e0
        )
    
    @staticmethod
    def compute_npz_grid_search_2025_08_22():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_2025_08_22_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_2025_08_22_processed_video,
            npz_in=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_npz,
            # npz_out_reordered=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz,
            # frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_2025_08_22_frame_mapping_npz,
            # video_from_cache=True,
            # npz_in=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz,
            # labels_out=SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz,
            # labels_in=SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz,
        )

    @staticmethod
    def compute_npz_helper(
        video_in,
        video_out,
        npz_out,
        npz_out_reordered=None,
        video_from_cache=False,
        npz_in=None,
        frame_mapping_npz_out=None,
        labels_out=None,
        labels_in=None,
        seconds_per_frame=None,
        foo=False,
    ):
        if not video_from_cache:
            marker_tracker = MarkerTracker()
            marker_tracker.extract_frames(
                video_in,
                frame_mapping_npz_out,
                seconds_per_frame=seconds_per_frame
            )
            PredictExp.write_video_to_npz_file(
                marker_tracker=marker_tracker,
                path=npz_out
            )
            if npz_out_reordered is not None:
                HungarianExp.reorder_exp_points(
                    input_path=npz_out,
                    output_path=npz_out_reordered
                )
            marker_tracker.create_visualization(
                out_path=video_out,
                mode="unpaired-markers",
                base_from_file=False,
                npz_in=npz_in
            )
        # player = VideoPlayer(
        #     video_in_path=video_out,
        #     markers_in_path_npz=npz_in,
        #     labels_out_path_npz=labels_out,
        #     labels_in_path_npz=labels_in
        # )
        # player.run()
    
    @staticmethod
    def compute_npz_helper2(
        video_in,
        video_out,
        npz_in,
        npz_out,
    ):
        marker_tracker = MarkerTracker()
        marker_tracker.extract_frames(video_in)
        HungarianExp.reorder_exp_points(
            input_path=npz_in,
            output_path=npz_out,
        )
        marker_tracker.create_visualization(
            out_path=video_out,
            mode="unpaired-markers",
            base_from_file=False,
            npz_in=npz_out,
        )
    
    @staticmethod
    def compute_npz_helper3(
        video_in,
        video_out,
        npz_out,
    ):
        marker_tracker = MarkerTracker()
        marker_tracker.extract_frames(video_in)
        PredictExp.write_video_to_npz_file(
            marker_tracker=marker_tracker,
            path=npz_out,
        )
        marker_tracker.create_visualization(
            out_path=video_out,
            mode="unpaired-markers",
            base_from_file=False,
        )
    
    def downsample_ground_truth_image_to_prediction_shape(self):
        ground_truth_path = self.ground_truth_img_path
        prediction_path = self.prediction_img_path
        sensor_trajectory_path = self.sensor_trajectory_img_path
        ground_truth_feasible = self.ground_truth_feasible_img_path
        
        # Load both images
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        sensor_trajectory = cv2.imread(sensor_trajectory_path, cv2.IMREAD_GRAYSCALE)
        
        if ground_truth is None or prediction is None or sensor_trajectory is None:
            raise ValueError("Failed to load ground truth, prediction, or sensor trajectory image")
            
        # Binarize images if they aren't already (assuming 0 and 255 values)
        ground_truth = (ground_truth > 127).astype(np.float32)
        
        target_height, target_width = prediction.shape
        
        # Calculate scaling factors
        scale_y = ground_truth.shape[0] / target_height
        scale_x = ground_truth.shape[1] / target_width
        
        # Initialize output array
        downsampled = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Perform block majority voting
        for y in range(target_height):
            for x in range(target_width):
                # Calculate block boundaries
                y_start = int(y * scale_y)
                y_end = int((y + 1) * scale_y)
                x_start = int(x * scale_x)
                x_end = int((x + 1) * scale_x)
                
                # Handle edge cases
                y_end = min(y_end, ground_truth.shape[0])
                x_end = min(x_end, ground_truth.shape[1])
                
                # Extract block and compute majority vote
                block = ground_truth[y_start:y_end, x_start:x_end]
                block_mean = np.mean(block) if block.size > 0 else 0.0
                downsampled[y, x] = 1.0 if block_mean > 0.5 else 0.0
        
        # Convert to uint8 image format and save
        downsampled_image = (downsampled * 255).astype(np.uint8)
        cv2.imwrite(self.ground_truth_img_downsampled_path, downsampled_image)
        
        # Compute intersection with sensor trajectory
        sensor_trajectory_binary = (sensor_trajectory > 127)
        downsampled_binary = (downsampled_image > 127)
        intersection = np.logical_and(sensor_trajectory_binary, downsampled_binary).astype(np.uint8) * 255
        cv2.imwrite(ground_truth_feasible, intersection)
        
        return downsampled

    def evaluate_downscaled(self):
        # Load prediction and feasible ground truth images
        prediction = cv2.imread(self.prediction_img_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(self.ground_truth_feasible_img_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_og = cv2.imread(self.ground_truth_img_downsampled_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_from_video = cv2.imread(self.ground_truth_from_video_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to binary format
        prediction = (prediction > 127).astype(np.uint8)
        ground_truth = (ground_truth > 127).astype(np.uint8)
        ground_truth_og = (ground_truth_og > 127).astype(np.uint8)
        ground_truth_from_video = (ground_truth_from_video > 127).astype(np.uint8)
        
        # Compute IoU scores
        prediction_tensor = torch.from_numpy(prediction).float().unsqueeze(0).unsqueeze(0)
        ground_truth_tensor = torch.from_numpy(ground_truth).float().unsqueeze(0).unsqueeze(0)
        metrics = Common.iou_score(prediction_tensor, ground_truth_tensor)
        print("IoU Metrics (downscaled):", metrics)
        
        # Create confusion matrix overlay
        confusion_overlay = Visualisation.create_confusion_matrix_overlay(ground_truth_from_video, prediction)
        
        # Load all images for visualization
        images = {
            'Original Ground Truth': cv2.imread(self.ground_truth_img_path, cv2.IMREAD_GRAYSCALE),
            'Downsampled Ground Truth': cv2.imread(self.ground_truth_img_downsampled_path, cv2.IMREAD_GRAYSCALE),
            'Ground Truth from Video': (ground_truth_from_video * 255).astype(np.uint8),
            'Sensor Trajectory': cv2.imread(self.sensor_trajectory_img_path, cv2.IMREAD_GRAYSCALE),
            'Feasible Ground Truth': cv2.imread(self.ground_truth_feasible_img_path, cv2.IMREAD_GRAYSCALE),
            'Prediction': cv2.imread(self.prediction_img_path, cv2.IMREAD_GRAYSCALE),
            'Confusion Overlay': (confusion_overlay * 255).astype(np.uint8),
        }

        # Create GBR overlay image
        trajectory = cv2.imread(self.sensor_trajectory_img_path, cv2.IMREAD_GRAYSCALE)
        prediction_img = cv2.imread(self.prediction_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a 3-channel image (BGR format)
        gbr_overlay = np.zeros((trajectory.shape[0], trajectory.shape[1], 3), dtype=np.uint8)
        
        # Set cyan (G=255, B=255) for trajectory and magenta (B=255, R=255) for prediction
        gbr_overlay[trajectory > 127] = [255, 255, 0]  # Cyan in BGR
        gbr_overlay[prediction_img > 127] = [255, 0, 255]  # Magenta in BGR
        
        # Add GBR overlay to images dictionary
        images['GBR Overlay'] = gbr_overlay

        # Check if all images were loaded successfully
        if any(img is None for img in images.values()):
            raise ValueError("Failed to load one or more images for visualization")
        
        # Define target size for each image
        k = 4
        target_width = 180 * k
        target_height = 100 * k
        
        # Resize all images to the target size
        resized_images = []
        for name, img in images.items():
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            resized_images.append(resized)
        
        # Create a 3x3 grid
        grid = np.zeros((target_height * 3, target_width * 3, 3), dtype=np.uint8)
        
        # Place images in the grid
        for idx, img in enumerate(resized_images):
            row = idx // 3
            col = idx % 3
            grid[row * target_height:(row + 1) * target_height, 
                 col * target_width:(col + 1) * target_width] = img
        
        # Display the grid
        plt.figure(figsize=(30, 10))
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()  # toggles fullscreen mode
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Add titles for each subplot
        titles = list(images.keys())
        for idx, title in enumerate(titles):
            row = idx // 3
            col = idx % 3
            plt.text(col * target_width + target_width/2, 
                    row * target_height + 20, 
                    title,
                    horizontalalignment='center',
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(SYSTEM_PARAMS.files.exp_overlay_downscaled)
        plt.show()
        plt.close()

    def evaluate_upscaled(self):
        ground_truth_path = self.ground_truth_img_path
        prediction_path = self.prediction_img_path
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = (ground_truth == 255).astype(np.uint8)
        prediction = (prediction == 255).astype(np.uint8)
        
        gt_height, gt_width = ground_truth.shape
        pred_height, pred_width = prediction.shape
        scale_x = int(gt_width / pred_width)
        scale_y = int(gt_height / pred_height)
        scale_factor = min(scale_x, scale_y)
        scale_factor = max(1, scale_factor)
        scaled_prediction = cv2.resize(prediction, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        scaled_height, scaled_width = scaled_prediction.shape
        pad_height = gt_height - scaled_height
        pad_width = gt_width - scaled_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_prediction = np.zeros((gt_height, gt_width), dtype=np.uint8)
        padded_prediction[pad_top:pad_top+scaled_height, pad_left:pad_left+scaled_width] = scaled_prediction
        prediction = padded_prediction

        # Convert numpy arrays to PyTorch tensors with correct shape [B, T, H, W]
        prediction_tensor = torch.from_numpy(prediction).float().unsqueeze(0).unsqueeze(0)
        ground_truth_tensor = torch.from_numpy(ground_truth).float().unsqueeze(0).unsqueeze(0)
        metrics = Common.iou_score(prediction_tensor, ground_truth_tensor)
        confusion_overlay = Visualisation.create_confusion_matrix_overlay(ground_truth, prediction)
        print(metrics)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(prediction, cmap='gray')
        plt.title('Prediction (Scaled & Padded)')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(confusion_overlay)
        plt.title(f'Overlay')
        plt.axis('off')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='black', label='True Negative'),
            plt.Rectangle((0, 0), 1, 1, fc='white', label='True Positive'),
            plt.Rectangle((0, 0), 1, 1, fc='red', label='False Positive'),
            plt.Rectangle((0, 0), 1, 1, fc='blue', label='False Negative')
        ]
        plt.figlegend(handles=legend_elements, loc='center right')
        plt.tight_layout()
        plt.savefig(SYSTEM_PARAMS.files.exp_overlay_upscaled)
        plt.show()
        plt.close()

    def go(self):
        self.predict_all_clips()
        self.write_probs_to_npz()
        self.load_probs_from_npz()
        self.generate_mask_image()
        self.downsample_ground_truth_image_to_prediction_shape()
        self.evaluate_downscaled()


def main():
    predict_exp = PredictExp()
    predict_exp.go()
    # PredictExp.compute_npz_grid_search_og()
