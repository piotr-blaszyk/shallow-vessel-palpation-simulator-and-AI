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
from difftactile.cnn.lit_module import SegmentationModel
from difftactile.cnn.visualise import *
from difftactile.main.main import *

class PredictExp:
    def __init__(self):
        self.fisheye_model = FisheyeModel()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.marker_tracker = MarkerTracker()
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
        self.dilation = 1
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

    def write_video_to_npz_file(self):
        path = SYSTEM_PARAMS.files.exp_video_npz
        n = len(self.marker_tracker.frame_markers)
        markers_array, markers_mask = Contact.create_padded_array_with_mask(self.marker_tracker.frame_markers)
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
        transformed_seq = np.zeros((num_frames, self.camera_h_small, self.camera_w_small), dtype=np.float32)
        
        for t in range(num_frames):
            x, y, z = self.all_positions[i + t]
            t_EA = self.get_T_EA(
                np.deg2rad(SYSTEM_PARAMS.geometry.camera_rotation_angle),
                x,
                y,
                z
            )

            # Convert to uint8 for cv2 operations (scale from [0,1] to [0,255])
            frame = (pred_seq[t] * 255).astype(np.uint8)
            
            # Resize crop from small to big
            crop_big = cv2.resize(frame, (self.crop_w_big, self.crop_h_big), interpolation=cv2.INTER_NEAREST)
            
            # Place in full camera frame
            camera_big = np.zeros((self.camera_h_big, self.camera_w_big), dtype=np.uint8)
            camera_big[self.crop_y:self.crop_y + self.crop_h_big, self.crop_x:self.crop_x + self.crop_w_big] = crop_big
            
            # Resize to final small camera frame
            camera_small = cv2.resize(camera_big, (self.camera_w_small, self.camera_h_small), interpolation=cv2.INTER_AREA)
            
            # Convert back to float32 in range [0,1]
            transformed_seq[t] = camera_small.astype(np.float32) / 255.0

            for j in range(self.camera_h_small):
                for k in range(self.camera_w_small):
                    prob = transformed_seq[t, j, k]
                    pos_E = self.map_2d_3d[j, k]
                    pos_E_homogeneous = np.append(pos_E, 1)
                    pos_A_homogeneous = t_EA @ pos_E_homogeneous
                    pos_A = pos_A_homogeneous[:3]
                    x_A = pos_A[0]
                    y_A = pos_A[1]
                    x_A -= self.min_x
                    y_A -= self.min_y
                    x_A = int(x_A)
                    y_A = int(y_A)
                    succ = (
                        x_A >= 0 and
                        x_A < 105 and
                        y_A >= 0 and
                        y_A < 180
                    )
                    if not succ:
                        continue
                    self.bin_prob_sum[x_A, y_A] += prob
                    self.bin_count[x_A, y_A] += 1

    def predict_all_clips(self):
        n = self.data['markers'].shape[0]
        for i in tqdm(range(n - self.dilated_clip_len), desc="clip inference"):
            self.predict_clip(i)
        
        res = np.divide(self.bin_prob_sum, self.bin_count, where=self.bin_count != 0)
        res = (res > 0.4).astype(np.int32)

        img = (res * 255).astype(np.uint8)
        img = np.flip(np.flip(img, axis=0), axis=1)
        cv2.imwrite(SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask, img)
    
    def compute_npz(self):
        self.marker_tracker.extract_frames(
            SYSTEM_PARAMS.files.vein_slide_across
        )
        self.marker_tracker.create_visualization(
            out_path=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers,
            mode="unpaired-markers",
            base_from_file=False
        )
        player = VideoPlayer(
            in_path=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers
        )
        player.run()
        self.write_video_to_npz_file()

    def go(self):
        self.load_npz()
        self.compute_all_3d_positions()
        self.predict_all_clips()


def main():
    predict_exp = PredictExp()
    predict_exp.go()
