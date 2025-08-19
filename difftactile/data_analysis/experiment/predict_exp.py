import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.main.constants import *
from difftactile.main.synthetic_image_generator import SyntheticImageGenerator
from difftactile.cnn.gnn import *
from difftactile.cnn.dataset import *
from difftactile.cnn.visualise import *

class PredictExp:
    def __init__(self):
        self.fisheye_model = FisheyeModelNoTaichi()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # og video has 1305 frames - so if I now use a video /w fewer frames, I just map a different range to the range (0, 1304)
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
        self.video_frames //= 3
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
        self.dataset = MyDataset(
            data_dir=SYSTEM_PARAMS.files.dummy_data_dir
        )
        with open(SYSTEM_PARAMS.files.test_loader_gnn, 'rb') as f:
            test_data = pickle.load(f)
        self.stats = test_data['dataset_stats'][1.0]
        self.dataset.set_stats(self.stats)
    
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
        markers_array, markers_mask = SyntheticImageGenerator.create_padded_array_with_mask(marker_tracker.frame_markers)
        vein_data = [np.array([]) for i in range(n)]
        veins_array, veins_mask = SyntheticImageGenerator.create_padded_array_with_mask(vein_data)
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
        pyg = self.dataset.get_clip(
            self.data,
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
            probs = probs.cpu().numpy().astype(np.float32)
        points = pyg.pos.cpu().numpy().astype(np.float32)
        points = self.z_unnormalise(points)
        points = points.reshape((self.clip_len, 127, 2))
        probs = probs.reshape((self.clip_len, 127,))

        for t in range(self.clip_len):
            x, y, z = self.all_positions[i + t*self.dilation]
            t_EA = self.get_T_EA(
                np.deg2rad(SYSTEM_PARAMS.geometry.camera_rotation_angle),
                x,
                y,
                z
            )
            for j in range(127):
                prob = probs[t, j]
                x, y = points[t, j]
                pos_E = self.map_2d_3d[int(x), int(y)]
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
        for i in tqdm(range(0, n - self.dilated_clip_len, self.dilated_clip_len), desc="clip inference"):
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
            npz_out=SYSTEM_PARAMS.files.npz_out_straight
        )

    @staticmethod
    def compute_npz_helper(
        video_in,
        video_out,
        npz_out
    ):
        marker_tracker = MarkerTracker(
            # start_frame_ix=1000,
            # end_frame_ix=1050
        )
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

    @staticmethod
    def generate_image(points, probabilities):
        # Create a black background image
        image = np.zeros((1080, 1920), dtype=np.uint8)
        
        # Draw circles for each point with intensity based on probability
        for point, prob in zip(points, probabilities):
            # Convert point coordinates to integers
            center = (int(point[0]), int(point[1]))
            # Scale probability to intensity range (0-255)
            intensity = int(prob * 255)
            # Draw white circle with given intensity
            cv2.circle(image, center, radius=5, color=intensity, thickness=-1)
            
        return image


def main():
    # predict_exp = PredictExp()
    # predict_exp.go()
    PredictExp.compute_npz_test()
