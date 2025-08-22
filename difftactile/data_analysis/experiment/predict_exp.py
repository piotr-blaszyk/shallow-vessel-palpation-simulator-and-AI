import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import math
from scipy.interpolate import interp1d

from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.data_analysis.experiment.hungarian_exp import *
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.main.constants import *
from difftactile.main.synthetic_image_generator import SyntheticImageGenerator
from difftactile.cnn.gnn import *
from difftactile.cnn.dataset import *
from difftactile.cnn.visualise import *
from difftactile.cnn.common import Common

class PredictExp:
    def __init__(self):
        self.load_npz()
        self.fisheye_model = FisheyeModelNoTaichi()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frames_poses_data = np.load(SYSTEM_PARAMS.files.experiment_2025_08_22_output_npz)
        self.frames_poses = frames_poses_data['output']
        ground_truth_labels_path = SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz
        ground_truth_labels_data = np.load(ground_truth_labels_path)
        self.ground_truth_labels = ground_truth_labels_data['labels']
        self.interpolate_poses()
        self.sensor_radius = 20
        self.phantom_length_x = 105
        self.phantom_length_y = 180
        self.min_x = np.min(self.poses_interpolated[:, 0]) - self.sensor_radius
        self.max_x = self.min_x + self.phantom_length_x
        self.min_y = np.min(self.poses_interpolated[:, 1]) - self.sensor_radius
        self.max_y = self.min_y + self.phantom_length_y
        self.bin_size = 1
        self.bin_num_x = math.ceil(self.phantom_length_x / self.bin_size)
        self.bin_num_y = math.ceil(self.phantom_length_y / self.bin_size)
        self.bins = np.zeros(shape=(2, self.bin_num_x, self.bin_num_y), dtype=int)
        self.bin_prob_sum = np.zeros(shape=(self.bin_num_x, self.bin_num_y), dtype=float)
        self.bin_count = np.zeros(shape=(self.bin_num_x, self.bin_num_y), dtype=int)
        self.bins_ground_truth = np.zeros(shape=(self.bin_num_x, self.bin_num_y), dtype=int)
        self.debug_bins = np.zeros(shape=(self.markers.shape[0]), dtype=int)
        self.clip_len = SYSTEM_PARAMS.cnn.clip_len
        self.dilation = 1
        self.dilated_clip_len = self.clip_len * self.dilation
        self.init_model()
        self.init_camera_params()
        self.compute_mapping_2d_3d()
        self.dataset = MyDataset(mode='dummy', scheme='new', normalise_pos=False)
        with open(SYSTEM_PARAMS.files.test_loader_gnn, 'rb') as f:
            test_data = pickle.load(f)
        self.stats = test_data['dataset_stats'][1.0]
        self.dataset.set_stats(self.stats)
        self.dataset.set_difficulty_level(1.0)

        self.ground_truth_img_path = SYSTEM_PARAMS.files.phantom_ground_truth_segmentation_mask
        self.ground_truth_img_downsampled_path = SYSTEM_PARAMS.files.ground_truth_labels_downsampled
        self.prediction_img_path = SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask
        self.sensor_trajectory_img_path = SYSTEM_PARAMS.files.sensor_trajectory
        self.ground_truth_feasible_img_path = SYSTEM_PARAMS.files.ground_truth_feasible
        self.ground_truth_from_video_img_path = SYSTEM_PARAMS.files.experiment_ground_truth_from_video_img_path
    
    def z_unnormalise(self, points):
        x_mean = self.stats['x_mean']
        x_std = self.stats['x_std']
        y_mean = self.stats['y_mean']
        y_std = self.stats['y_std']
        points[:, 0] = (points[:, 0] * x_std) + x_mean
        points[:, 1] = (points[:, 1] * y_std) + y_mean
        return points
    
    def get_T_EA(self, k, x, y, z):
        cos_k = np.cos(k)
        sin_k = np.sin(k)
        R = np.array([
            [ cos_k,  sin_k, 0],
            [ sin_k, -cos_k, 0],
            [     0,      0, -1]
        ])
        t = np.array([[x], [y], [z]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t
        return T
    
    def compute_mapping_2d_3d(self):
        pixel_coords = np.zeros((self.camera_w_big, self.camera_h_big, 2), dtype=np.float32)
        for i in range(self.camera_w_big):
            for j in range(self.camera_h_big):
                pixel_coords[i, j, :] = np.array([i, j])
        points_E = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
            ps=pixel_coords,
            dist_lens_to_plane=SYSTEM_PARAMS.scaling_factor_1.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.scaling_factor_1.press_depth_1
        )
        points_E *= 1_000
        self.map_2d_3d = points_E

    def init_model(self):
        model_path = SYSTEM_PARAMS.files.final_segmentation_model_gnn
        model = GNN(lr=-1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.model = model
        self.device = device
    
    def init_camera_params(self):
        self.camera_w_big = 1920
        self.camera_h_big = 1080

    def interpolate_poses(self):
        # Get the total number of video frames from markers shape
        num_video_frames = self.markers.shape[0]
        
        # Extract frame indices and poses
        frame_indices = self.frames_poses[:, 0]
        poses = self.frames_poses[:, 1:7]
        
        # Sort frame indices and poses together
        sort_idx = np.argsort(frame_indices)
        frame_indices = frame_indices[sort_idx]
        poses = poses[sort_idx]
        
        # Remove duplicates and very close indices
        eps = 1e-10  # Small threshold for considering indices as duplicates
        unique_mask = np.concatenate(([True], np.diff(frame_indices) > eps))
        frame_indices = frame_indices[unique_mask]
        poses = poses[unique_mask]
        
        # Create interpolation function for each pose dimension
        interpolators = []
        for dim in range(6):
            interpolator = interp1d(frame_indices, poses[:, dim], 
                                  kind='linear',
                                  fill_value=np.nan,  # Set values outside range to NaN
                                  bounds_error=False)  # Allow setting to NaN outside bounds
            interpolators.append(interpolator)
        
        # Initialize the interpolated poses array
        self.poses_interpolated = np.zeros((num_video_frames, 6))
        
        # Interpolate each dimension using the frame mapping
        for dim in range(6):
            # Only interpolate for frame indices within the known range
            self.poses_interpolated[:, dim] = interpolators[dim](self.frame_mapping)

        # Create a boolean mask for rows without NaN values
        valid_mask = ~np.isnan(self.poses_interpolated).any(axis=1)
        
        # Create index mapping from original to filtered indices
        original_indices = np.arange(num_video_frames)
        filtered_indices = np.cumsum(valid_mask) - 1  # -1 to make 0-based
        self.markers_index_mapping = {orig_idx: filt_idx for orig_idx, filt_idx in zip(original_indices[valid_mask], filtered_indices[valid_mask])}
        
        # Filter the arrays using the mask
        self.poses_interpolated = self.poses_interpolated[valid_mask]
        self.markers = self.markers[valid_mask]
        self.markers_mask = self.markers_mask[valid_mask]
        self.frame_mapping = self.frame_mapping[valid_mask]
        self.ground_truth_labels = self.ground_truth_labels[valid_mask]
        foo = 7

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
    
    def load_npz(self):
        path = SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz
        data = np.load(path)
        self.markers = data['markers']
        self.markers_mask = data['markers_mask']

        path_fm = SYSTEM_PARAMS.files.experiment_2025_08_22_frame_mapping_npz
        data_fm = np.load(path_fm)
        self.frame_mapping = data_fm['frame_mapping']
        
    
    def predict_clip(self, i):
        pyg = self.dataset.get_clip(
            self.markers,
            self.markers_mask,
            self.clip_len,
            self.dilation,
            i
        )
        with torch.no_grad():
            pyg = pyg.to(self.device)
            out = self.model(pyg.x, pyg.edge_index, pyg.edge_attr)
            out = out.squeeze(-1)  # Remove the channel dimension
            mask = pyg.mask
            out = out[mask]
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            probs = probs.cpu().numpy().astype(np.float32)
            preds = preds.cpu().numpy().astype(np.float32)
        points = pyg.pos.cpu().numpy().astype(np.float32)
        points = points.reshape((self.clip_len, 127, 2))
        probs = probs.reshape((self.clip_len, 127,))
        preds = preds.reshape((self.clip_len, 127,))

        assert points.min() > 0
        assert points.max() > 100

        # visible_vein_frame_ixs = np.array([
        #     32,
        #     50,
        #     84,
        #     121,
        #     129,
        #     135,
        #     167,
        #     170,
        #     179,
        #     210,
        #     220,
        #     256,
        #     # 259,
        #     297,
        #     313
        #     ], dtype=int
        # )
        # visible_vein_frame_ixs = np.array([32], dtype=int)
        # visible_vein_frame_ixs = np.array([
        #     [38, 16],
        #     [53, 16],
        #     [87, 2],
        #     [124, 32], # barely visible
        #     [130, 2],
        #     [137, 2],
        #     [168, 0],
        #     [173, 2],
        #     [181, 2],
        #     [212, 0],
        #     [222, 0],
        #     [257, 2],
        #     [301, 33], # barely visible
        #     [309, 39]
        # ], dtype=int)

        visible_vein_frame_ixs_original = np.array([
            [39, 1],
            [53, 22],
            [92, 22],
            [134, 2],
            [139, 23],
            [171, 22],
            [177, 1],
            [217, 17],
            [222, 1],
            [261, 1],
            [306, 32],
            [352, 60] # barely visible
        ])
        
        # Map frame indices using markers_index_mapping
        visible_vein_frame_ixs = np.array([
            [self.markers_index_mapping[frame_ix], marker_ix] 
            for frame_ix, marker_ix in visible_vein_frame_ixs_original 
            if frame_ix in self.markers_index_mapping
        ])
        assert visible_vein_frame_ixs_original.shape == visible_vein_frame_ixs.shape

        for t in range(self.clip_len):
            frame_ix = i + t*self.dilation
            x, y, z = self.poses_interpolated[frame_ix][:3]
            t_EA = self.get_T_EA(
                np.deg2rad(SYSTEM_PARAMS.geometry.camera_rotation_angle),
                x,
                y,
                z
            )
            if frame_ix == 34:
                foo = 7
            for j in range(127):
                prob = probs[t, j]
                pred = preds[t, j]
                x, y = points[t, j]
                pos_E = self.map_2d_3d[int(x), int(y)]
                pos_E_homogeneous = np.append(pos_E, 1)
                pos_A_homogeneous = t_EA @ pos_E_homogeneous
                pos_A = pos_A_homogeneous[:3]
                x_A = pos_A[0]
                y_A = pos_A[1]
                x_A -= self.min_x
                y_A -= self.min_y
                succ = (
                    x_A >= 0 and
                    x_A < self.phantom_length_x and
                    y_A >= 0 and
                    y_A < self.phantom_length_y
                )
                frame_ix_marker_ix = np.array([frame_ix, j], dtype=int)
                is_present = np.any(np.all(visible_vein_frame_ixs == frame_ix_marker_ix, axis=1))
                # succ &= is_present
                if not succ:
                    continue
                self.debug_bins[frame_ix] += 1
                x_A = int(x_A / self.bin_size)
                y_A = int(y_A / self.bin_size)

                # foo = 1 if frame_ix in visible_vein_frame_ixs else 0
                foo = 1 if j == 0 else 0
                self.bin_prob_sum[x_A, y_A] += foo
                self.bin_count[x_A, y_A] += 1
                if self.ground_truth_labels[frame_ix, j] == 1:
                    self.bins_ground_truth[x_A, y_A] += 1
        
    def predict_all_clips(self):
        n = self.markers.shape[0]
        # step_size = self.dilated_clip_len
        step_size = 1
        for i in (
            tqdm(
                range(0, n - self.dilated_clip_len, step_size)
                , desc="clip inference")
            ):
            self.predict_clip(i)
        
        self.write_probs_to_npz()
    
    def debug_projection(self):
        pass
    
    def write_probs_to_npz(self):
        path = SYSTEM_PARAMS.files.exp_probs_npz
        np.savez(
            path,
            bin_prob_sum=self.bin_prob_sum,
            bin_count=self.bin_count,
            bins_ground_truth=self.bins_ground_truth
        )
    
    def load_probs_from_npz(self):
        path = SYSTEM_PARAMS.files.exp_probs_npz
        data = np.load(path)
        self.bin_prob_sum = data['bin_prob_sum']
        self.bin_count = data['bin_count']
        self.bins_ground_truth = data['bins_ground_truth']
    
    def generate_mask_image(self):
        res = np.divide(self.bin_prob_sum, self.bin_count, where=self.bin_count != 0)
        # threshold = np.percentile(res, 50)
        threshold = 1e-6

        res_binary = (self.bin_prob_sum > 1e-6).astype(np.int32)
        img = (res_binary * 255).astype(np.uint8)
        img = np.flip(np.flip(img, axis=0), axis=1)
        cv2.imwrite(self.prediction_img_path, img)

        feasible_binary = (self.bin_count > 0).astype(np.int32)
        img = (feasible_binary * 255).astype(np.uint8)
        img = np.flip(np.flip(img, axis=0), axis=1)
        cv2.imwrite(self.sensor_trajectory_img_path, img)

        bins_ground_truth_binary = (self.bins_ground_truth > 0).astype(np.int32)
        img = (bins_ground_truth_binary * 255).astype(np.uint8)
        img = np.flip(np.flip(img, axis=0), axis=1)
        cv2.imwrite(self.ground_truth_from_video_img_path, img)
    
    @staticmethod
    def compute_npz():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.vein_slide_across,
            video_out=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers,
            npz_out=SYSTEM_PARAMS.files.exp_video_npz
        )
    
    @staticmethod
    def compute_npz_test():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.vein_slide_across,
            video_out=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers_test,
            npz_out=SYSTEM_PARAMS.files.exp_video_npz_test
        )
    
    @staticmethod
    def compute_npz_straight():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.video_in_straight,
            video_out=SYSTEM_PARAMS.files.video_out_straight,
            npz_out=SYSTEM_PARAMS.files.exp_simple_straight_npz
        )
    
    @staticmethod
    def compute_npz_2025_08_22():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_2025_08_22_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_2025_08_22_processed_video,
            npz_out=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_npz,
            npz_out_reordered=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz,
            frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_2025_08_22_frame_mapping_npz,
            video_from_cache=True,
            npz_in=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz,
            labels_out=SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz,
            labels_in=SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz,
        )

    @staticmethod
    def compute_npz_helper(
        video_in,
        video_out,
        npz_out,
        npz_out_reordered=None,
        video_from_cache=False,
        # npz_in=SYSTEM_PARAMS.files.exp_video_npz_reordered
        npz_in=None,
        frame_mapping_npz_out=None,
        labels_out=None,
        labels_in=None,
    ):
        if not video_from_cache:
            marker_tracker = MarkerTracker(
                # start_frame_ix=1000,
                # end_frame_ix=1050
            )
            marker_tracker.extract_frames(
                video_in,
                frame_mapping_npz_out
            )
            marker_tracker.create_visualization(
                out_path=video_out,
                mode="unpaired-markers",
                base_from_file=False,
                npz_in=npz_in
            )
        player = VideoPlayer(
            video_in_path=video_out,
            markers_in_path_npz=npz_in,
            labels_out_path_npz=labels_out,
            labels_in_path_npz=labels_in
        )
        player.run()
        if False and not video_from_cache:
            PredictExp.write_video_to_npz_file(
                marker_tracker=marker_tracker,
                path=npz_out
            )
            if npz_out_reordered is not None:
                HungarianExp.reorder_exp_points(
                    input_path=npz_out,
                    output_path=npz_out_reordered
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
                block_mean = np.mean(block)
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
        confusion_overlay = Visualisation.create_confusion_matrix_overlay(ground_truth_og, prediction)
        
        # Load all images for visualization
        images = {
            'Original Ground Truth': cv2.imread(self.ground_truth_img_path, cv2.IMREAD_GRAYSCALE),
            'Downsampled Ground Truth': cv2.imread(self.ground_truth_img_downsampled_path, cv2.IMREAD_GRAYSCALE),
            'Ground Truth from Video': ground_truth_from_video,
            'Sensor Trajectory': cv2.imread(self.sensor_trajectory_img_path, cv2.IMREAD_GRAYSCALE),
            'Feasible Ground Truth': cv2.imread(self.ground_truth_feasible_img_path, cv2.IMREAD_GRAYSCALE),
            'Prediction': cv2.imread(self.prediction_img_path, cv2.IMREAD_GRAYSCALE),
            'Confusion Overlay': (confusion_overlay * 255).astype(np.uint8),
        }
        
        # Check if all images were loaded successfully
        if any(img is None for img in images.values()):
            raise ValueError("Failed to load one or more images for visualization")
        
        # Define target size for each image
        k = 4
        target_width = 180 * k
        target_height = 105 * k
        
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
        plt.savefig(SYSTEM_PARAMS.files.exp_overlay)
        plt.show()
        plt.close()

    def go(self):
        self.predict_all_clips()
        self.write_probs_to_npz()
        self.load_probs_from_npz()
        self.generate_mask_image()
        self.downsample_ground_truth_image_to_prediction_shape()
        self.evaluate_downscaled()
        # self.evaluate_upscaled()


def main():
    predict_exp = PredictExp()
    predict_exp.go()
    # PredictExp.compute_npz_2025_08_22()
