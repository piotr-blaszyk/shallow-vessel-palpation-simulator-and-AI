import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.sensor_model.fisheye_model import *
from difftactile.main.constants import *
from difftactile.main.main import SyntheticImageGenerator
from difftactile.cnn.lit_module_unet_cnn import SegmentationModel
from difftactile.cnn.visualise import *
from difftactile.main.main import *

class PredictExp:
    def __init__(self):
        self.fisheye_model = FisheyeModel()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegmentationModel()
        self.model.load_state_dict(torch.load(SYSTEM_PARAMS.files.final_segmentation_model))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transforms = A.Compose([ToTensorV2()])
        self.poses = np.array([
            [-382.1576,   85.3686,   28.0000,   180.0,   0.0,   0.0],  # target 0
            [-382.1576,   85.3686,   24.0000,   180.0,   0.0,   0.0],  # target 1
            [-317.1576,   85.3686,   24.0000,   180.0,   0.0,   0.0],  # target 2
            [-317.1576,   65.3686,   24.0000,   180.0,   0.0,   0.0],  # target 3
            [-382.1576,   65.3686,   24.0000,   180.0,   0.0,   0.0],  # target 4
            [-382.1576,   45.3686,   24.0000,   180.0,   0.0,   0.0],  # target 5
            [-317.1576,   45.3686,   24.0000,   180.0,   0.0,   0.0],  # target 6
            [-317.1576,   25.3686,   24.0000,   180.0,   0.0,   0.0],  # target 7
            [-382.1576,   25.3686,   24.0000,   180.0,   0.0,   0.0],  # target 8
            [-382.1576,    5.3686,   24.0000,   180.0,   0.0,   0.0],  # target 9
            [-317.1576,    5.3686,   24.0000,   180.0,   0.0,   0.0],  # target 10
            [-317.1576,  -14.6314,   24.0000,   180.0,   0.0,   0.0],  # target 11
            [-382.1576,  -14.6314,   24.0000,   180.0,   0.0,   0.0],  # target 12
            [-382.1576,  -34.6314,   24.0000,   180.0,   0.0,   0.0],  # target 13
            [-317.1576,  -34.6314,   24.0000,   180.0,   0.0,   0.0],  # target 14
            [-317.1576,  -54.6314,   24.0000,   180.0,   0.0,   0.0],  # target 15
            [-382.1576,  -54.6314,   24.0000,   180.0,   0.0,   0.0],  # target 16
            [-382.1576,  -74.6314,   24.0000,   180.0,   0.0,   0.0],  # target 17
            [-317.1576,  -74.6314,   24.0000,   180.0,   0.0,   0.0],  # target 18
            [-317.1576,  -94.6314,   24.0000,   180.0,   0.0,   0.0],  # target 19
        ], dtype=float)
        self.video_frames = np.array([
            60,    # target 0
            77,    # target 1
            169,   # target 2
            205,   # target 3
            297,   # target 4
            333,   # target 5
            425,   # target 6
            461,   # target 7
            554,   # target 8
            590,   # target 9
            683,   # target 10
            719,   # target 11
            812,   # target 12
            849,   # target 13
            941,   # target 14
            978,   # target 15
            1077,  # target 16
            1114,  # target 17
            1207,  # target 18
            1243,  # target 19
        ], dtype=int)
        self.start_ix = 77
        self.end_ix = 1077
        self.sensor_radius = 20
        self.phantom_length_x = 105
        self.phantom_length_y = 180
        self.min_x = self.poses[1][0] - self.sensor_radius
        self.max_x = self.min_x + self.phantom_length_x
        self.min_y = self.poses[16][1] - self.sensor_radius
        self.max_y = self.min_y + self.phantom_length_y
        self.bins = np.zeros(shape=(2, self.phantom_length_x, self.phantom_length_y), dtype=int)
        self.bin_prob_sum = np.zeros(shape=(self.phantom_length_x, self.phantom_length_y), dtype=float)
        self.bin_count = np.zeros(shape=(self.phantom_length_x, self.phantom_length_y), dtype=int)
        self.clip_len = SYSTEM_PARAMS.cnn.clip_len
        self.dilation = 2
        self.dilated_clip_len = self.clip_len * self.dilation
        self.init_model()
        self.init_camera_params()
        self.compute_mapping_2d_3d()
    
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
        pixel_coords = np.zeros((self.camera_h_small, self.camera_w_small, 2), dtype=np.float32)
        for i in range(self.camera_h_small):
            for j in range(self.camera_w_small):
                pixel_coords[i, j, :] = np.array([j, i])
        points_E = self.fisheye_model.project_pix_to_points_3d_plane(
            ps=pixel_coords,
            dist_lens_to_plane=SYSTEM_PARAMS.scaling_factor_1.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.scaling_factor_1.press_depth_1,
            resolution_down_scaling_factor=SYSTEM_PARAMS.heatmap.down_scaling_factor
        )
        points_E *= 1_000
        self.map_2d_3d = points_E

    def init_model(self):
        model_path = SYSTEM_PARAMS.files.final_segmentation_model
        model = SegmentationModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.model = model
        self.device = device
    
    def init_camera_params(self):
        # Get resolution parameters
        self.crop_x = SYSTEM_PARAMS.fisheye_model.crop_x
        self.crop_y = SYSTEM_PARAMS.fisheye_model.crop_y
        self.crop_w_big = SYSTEM_PARAMS.fisheye_model.crop_width
        self.crop_h_big = SYSTEM_PARAMS.fisheye_model.crop_height
        self.camera_w_big = 1920
        self.camera_h_big = 1080
        self.camera_w_small = self.camera_w_big // SYSTEM_PARAMS.heatmap.down_scaling_factor
        self.camera_h_small = self.camera_h_big // SYSTEM_PARAMS.heatmap.down_scaling_factor

    def compute_all_3d_positions(self):
        n = self.data['markers'].shape[0]

        positions = self.poses[:, :3]
        all_positions = np.zeros(shape=(n, 3))
        all_positions[self.video_frames] = positions
        
        for i in range(len(self.video_frames) - 1):
            start_frame = self.video_frames[i]
            end_frame = self.video_frames[i + 1]
            start_pos = positions[i]
            end_pos = positions[i + 1]
            
            num_frames = end_frame - start_frame
            
            for j in range(1, num_frames):
                t = j / num_frames
                interpolated_pos = (1 - t) * start_pos + t * end_pos
                all_positions[start_frame + j] = interpolated_pos
        
        self.all_positions = all_positions
        print(f"interpolated positions length: {len(self.all_positions)}")

    @staticmethod
    def write_video_to_npz_file(marker_tracker, path):
        n = len(marker_tracker.frame_markers)
        markers_array, markers_mask = Contact.create_padded_array_with_mask(marker_tracker.frame_markers)
        vein_data = [np.array([]) for i in range(n)]
        veins_array, veins_mask = Contact.create_padded_array_with_mask(vein_data)
        np.savez(
            path,
            markers=markers_array,
            markers_mask=markers_mask,
            labels=veins_array,
            labels_mask=veins_mask
        )
    
    def load_npz(self):
        path = SYSTEM_PARAMS.files.exp_video_npz
        self.data = np.load(path)
    
    def predict_clip(self, i):
        w = SYSTEM_PARAMS.fisheye_model.crop_width
        h = SYSTEM_PARAMS.fisheye_model.crop_height
        k = SYSTEM_PARAMS.fisheye_model.down_scaling_factor
        clip = MyDataset.get_clip(h, w, k, self.data, self.clip_len, self.dilation, start_ix=i)
        with torch.no_grad():
            clip_input = clip.to(self.device)
            logits = self.model(clip_input)
            probs = torch.sigmoid(logits)
            pred = probs
            pred = pred.cpu()
        pred_seq = pred.numpy().squeeze()  # Shape: (T, H, W)

        # Transform each frame in the sequence
        num_frames = pred_seq.shape[0]
        
        # Create meshgrid for all pixel coordinates once
        j_coords, k_coords = np.meshgrid(np.arange(self.camera_h_small), np.arange(self.camera_w_small), indexing='ij')
        
        # Process all frames at once for resizing operations
        frames = (pred_seq * 255).astype(np.uint8)  # Shape: (T, H, W)
        
        # Resize all crops from small to big
        crops_big = np.array([cv2.resize(frame, (self.crop_w_big, self.crop_h_big), interpolation=cv2.INTER_NEAREST) 
                             for frame in frames])
        
        # Place all in full camera frames
        cameras_big = np.zeros((num_frames, self.camera_h_big, self.camera_w_big), dtype=np.uint8)
        cameras_big[:, self.crop_y:self.crop_y + self.crop_h_big, 
                      self.crop_x:self.crop_x + self.crop_w_big] = crops_big
        
        # Resize all to final small camera frames
        transformed_seq = np.array([cv2.resize(frame, (self.camera_w_small, self.camera_h_small), 
                                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 
                                  for frame in cameras_big])
        
        # Get positions for all pixels (same for all frames)
        pos_E = self.map_2d_3d[j_coords, k_coords]  # Shape: (H, W, 2)
        pos_E_homogeneous = np.pad(pos_E, ((0, 0), (0, 0), (0, 1)), constant_values=1)  # Shape: (H, W, 3)
        pos_E_flat = pos_E_homogeneous.reshape(-1, 4).T  # Shape: (4, H*W)
        
        # Process each frame's transformation matrix and accumulate results
        for t in range(num_frames):
            x, y, z = self.all_positions[i + t]
            t_EA = self.get_T_EA(
                np.deg2rad(SYSTEM_PARAMS.geometry.camera_rotation_angle),
                x, y, z
            )
            
            # Transform all points for this frame
            pos_A_homogeneous = t_EA @ pos_E_flat  # Shape: (4, H*W)
            pos_A = pos_A_homogeneous[:3].T.reshape(self.camera_h_small, self.camera_w_small, 3)
            
            # Calculate x_A and y_A for all points
            x_A = (pos_A[..., 0] - self.min_x).astype(np.int32)
            y_A = (pos_A[..., 1] - self.min_y).astype(np.int32)
            
            # Create mask for valid positions
            valid_mask = (x_A >= 0) & (x_A < 105) & (y_A >= 0) & (y_A < 180)
            
            # Update bin_prob_sum and bin_count using valid positions
            np.add.at(self.bin_prob_sum, (x_A[valid_mask], y_A[valid_mask]), transformed_seq[t][valid_mask])
            np.add.at(self.bin_count, (x_A[valid_mask], y_A[valid_mask]), 1)

    def predict_all_clips(self):
        n = self.data['markers'].shape[0]
        for i in tqdm(range(0, n - self.dilated_clip_len, self.dilated_clip_len // 4), desc="clip inference"):
            self.predict_clip(i)
        
        self.write_probs_to_npz()
    
    def write_probs_to_npz(self):
        path = SYSTEM_PARAMS.files.exp_probs_npz
        np.savez(
            path,
            bin_prob_sum=self.bin_prob_sum,
            bin_count=self.bin_count
        )
    
    def load_probs_from_npz(self):
        path = SYSTEM_PARAMS.files.exp_probs_npz
        data = np.load(path)
        self.bin_prob_sum = data['bin_prob_sum']
        self.bin_count = data['bin_count']
    
    def generate_mask_image(self):
        res = np.divide(self.bin_prob_sum, self.bin_count, where=self.bin_count != 0)
        threshold = np.percentile(res, 90)
        res_binary = (res > threshold).astype(np.int32)

        img = (res_binary * 255).astype(np.uint8)
        img = np.flip(np.flip(img, axis=0), axis=1)
        cv2.imwrite(SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask, img)
    
    def compute_npz(self):
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.vein_slide_across,
            video_out=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers,
            npz_out=SYSTEM_PARAMS.files.exp_video_npz
        )
    
    @staticmethod
    def compute_npz_straight():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.video_in_straight,
            video_out=SYSTEM_PARAMS.files.video_out_straight,
            npz_out=SYSTEM_PARAMS.files.npz_out_straight
        )

    @staticmethod
    def compute_npz_helper(
        video_in,
        video_out,
        npz_out
    ):
        marker_tracker = MarkerTracker()
        marker_tracker.extract_frames(
            video_in
        )
        marker_tracker.create_visualization(
            out_path=video_out,
            mode="unpaired-markers",
            base_from_file=False
        )
        player = VideoPlayer(
            in_path=video_out
        )
        player.run()
        PredictExp.write_video_to_npz_file(
            marker_tracker=marker_tracker,
            path=npz_out
        )

    def evaluate(self):
        ground_truth = cv2.imread(SYSTEM_PARAMS.files.phantom_ground_truth_segmentation_mask, cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask, cv2.IMREAD_GRAYSCALE)
        if ground_truth is None:
            raise ValueError(f"Failed to read ground truth mask from {SYSTEM_PARAMS.files.phantom_ground_truth_segmentation_mask}")
        if prediction is None:
            raise ValueError(f"Failed to read prediction mask from {SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask}")
        ground_truth = (ground_truth == 255).astype(np.uint8)
        prediction = (prediction == 255).astype(np.uint8)
        
        gt_height, gt_width = ground_truth.shape
        pred_height, pred_width = prediction.shape
        scale_x = gt_width // pred_width
        scale_y = gt_height // pred_height
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

        metrics = SegmentationModel.iou_score(prediction_tensor, ground_truth_tensor)
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
        # self.load_npz()
        # self.compute_all_3d_positions()
        # self.predict_all_clips()
        # self.write_probs_to_npz()
        # self.load_probs_from_npz()
        # self.generate_mask_image()
        # self.evaluate()
        # PredictExp.compute_npz_straight()
        pass


def main():
    predict_exp = PredictExp()
    predict_exp.go()
