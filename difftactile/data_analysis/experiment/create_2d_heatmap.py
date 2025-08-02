import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
from tqdm import tqdm

from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.sensor_model.fisheye_model import *
from difftactile.main.constants import *
from difftactile.main.main import SyntheticImageGenerator
from difftactile.cnn.lit_module import SegmentationModel

class HeatmapGenerator:
    def __init__(self):
        self.fisheye_model = FisheyeModel()
        self.synthetic_image_generator = SyntheticImageGenerator()
        self.marker_tracker = MarkerTracker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegmentationModel()
        self.model.load_state_dict(torch.load(SYSTEM_PARAMS.files.segmentation_model_weights))
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
        self.min_x = self.poses[1][0] - 20
        self.max_x = self.min_x + 105
        self.min_y = self.poses[16][1] - 20
        self.max_y = self.min_y + 180
        self.bins = np.zeros(shape=(2, 105, 180), dtype=int)

    def linear_interpolation(self):
        cap = cv2.VideoCapture(str(SYSTEM_PARAMS.files.vein_slide_across))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"raw video total frames: {total_frames}")

        positions = self.poses[:, :3]
        all_positions = np.zeros(shape=(total_frames, 3))
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

    def generate_synthetic_image_and_segmentation_mask(self, ix, markers):
        w = SYSTEM_PARAMS.fisheye_model.crop_width
        h = SYSTEM_PARAMS.fisheye_model.crop_height
        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x - SYSTEM_PARAMS.fisheye_model.crop_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y - SYSTEM_PARAMS.fisheye_model.crop_y
        r = SYSTEM_PARAMS.fisheye_model.circle_radius
        k = SYSTEM_PARAMS.fisheye_model.down_scaling_factor
        w_scaled = int(w / k)
        h_scaled = int(h / k)
        markers_file = SYSTEM_PARAMS.files.vein_slide_across_markers.format(ix)
        segmentation_file = SYSTEM_PARAMS.files.vein_slide_across_segmentation_mask.format(ix)

        markers = self.synthetic_image_generator.crop(markers)
        markers = self.synthetic_image_generator.filter_points(w, h, cx, cy, r, markers)
        markers /= k
        markers_img = np.zeros((w_scaled, h_scaled), dtype=np.uint8)
        for point in markers:
            x, y = int(point[0]), int(point[1])
            cv2.circle(markers_img, (x, y), radius=1, color=255, thickness=-1)
        cv2.imwrite(markers_file, markers_img)

        markers_img = markers_img.astype(np.float32) / 255.0
        with torch.no_grad():
            image_tensor = self.transforms(image=markers_img)["image"]
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            pred = self.model(image_tensor)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            pred = pred.cpu().numpy().squeeze()
            pred = (pred * 255).astype(np.uint8)
            cv2.imwrite(segmentation_file, pred)

    def upsample(self):
        crop_x = SYSTEM_PARAMS.fisheye_model.crop_x
        crop_y = SYSTEM_PARAMS.fisheye_model.crop_y
        crop_width = SYSTEM_PARAMS.fisheye_model.crop_width
        crop_height = SYSTEM_PARAMS.fisheye_model.crop_height
        down_scaling_factor = SYSTEM_PARAMS.fisheye_model.down_scaling_factor

        scaled_width = int(crop_width / down_scaling_factor)
        scaled_height = int(crop_height / down_scaling_factor)
        full_hd_width = 1920
        full_hd_height = 1080
        output_folder = SYSTEM_PARAMS.files.vein_slide_across_segmentation_mask_folder_full_hd
        os.makedirs(output_folder, exist_ok=True)
        input_folder = SYSTEM_PARAMS.files.vein_slide_across_segmentation_mask_folder
        image_files = glob.glob(os.path.join(input_folder, "*.png"))
        print(f"Found {len(image_files)} images to upsample")

        for image_file in image_files:
            filename = os.path.basename(image_file)
            scaled_img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if scaled_img is None:
                print(f"Warning: Could not read {image_file}")
                continue
            if scaled_img.shape != (scaled_height, scaled_width):
                print(f"Warning: Image {filename} has unexpected dimensions {scaled_img.shape}, expected ({scaled_height}, {scaled_width})")
                continue

            upsampled_img = cv2.resize(scaled_img, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)
            full_hd_img = np.zeros((full_hd_height, full_hd_width), dtype=np.uint8)
            full_hd_img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width] = upsampled_img
            
            downsampled_width = full_hd_width // SYSTEM_PARAMS.heatmap.down_scaling_factor
            downsampled_height = full_hd_height // SYSTEM_PARAMS.heatmap.down_scaling_factor
            downsampled_img = cv2.resize(full_hd_img, (downsampled_width, downsampled_height), interpolation=cv2.INTER_NEAREST)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, downsampled_img)
        print(f"Upsampled {len(image_files)} images to {output_folder}")
    
    def aggregate_segmentation_mask(self):
        image_files = sorted(glob.glob(os.path.join(SYSTEM_PARAMS.files.vein_slide_across_segmentation_mask_folder_full_hd, '*')))
        if not image_files:
            raise ValueError(f"No images found in {SYSTEM_PARAMS.files.vein_slide_across_segmentation_mask_folder_full_hd}")
        first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if first_img is None:
            raise ValueError(f"Failed to read first image: {image_files[0]}")
        height, width = first_img.shape
        num_images = len(image_files)
        print(f"num images: {num_images}")

        for i, img_path in tqdm(enumerate(image_files)):
            for label in [0, 1]:
                if i >= self.start_ix and i < self.end_ix:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Failed to read image {img_path}, skipping...")
                        continue
                    threshold = img.max() * 0.5
                    if label == 1:
                        y_coords, x_coords = np.where(img > threshold)
                    else:
                        y_coords, x_coords = np.where(img <= threshold)
                    if len(x_coords) > 0:
                        pixel_coords = np.column_stack((x_coords, y_coords))
                        points_E = self.fisheye_model.project_pix_to_points_3d_plane(
                            ps=pixel_coords, 
                            dist_lens_to_plane=SYSTEM_PARAMS.scaling_factor_1.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.scaling_factor_1.press_depth_1,
                            resolution_down_scaling_factor=SYSTEM_PARAMS.heatmap.down_scaling_factor
                        )
                        points_E *= 1_000

                        n = points_E.shape[0]
                        points_A = np.hstack((points_E, np.ones((n, 1))))
                        x, y, z = self.all_positions[i]
                        t_EA = self.get_T_EA(
                            np.deg2rad(SYSTEM_PARAMS.geometry.camera_rotation_angle),
                            x,
                            y,
                            z
                        )
                        points_A_h = points_A @ t_EA.T
                        points_A = points_A_h[:, :3]
                        
                        if i == 82 and label == 1:
                            foo = 7
                        mask = (
                            (points_A[:, 0] >= self.min_x) & 
                            (points_A[:, 0] <= self.max_x) & 
                            (points_A[:, 1] >= self.min_y) & 
                            (points_A[:, 1] <= self.max_y)
                        )
                        points_A = points_A[mask]
                        points_A[:, 0] -= self.min_x
                        points_A[:, 1] -= self.min_y
                        points_A = points_A[:, :2]
                        
                        x_indices = points_A[:, 0].astype(int)
                        y_indices = points_A[:, 1].astype(int)
                        x_indices = np.clip(x_indices, 0, 104)
                        y_indices = np.clip(y_indices, 0, 179)
                        for x_idx, y_idx in zip(x_indices, y_indices):
                            self.bins[label, x_idx, y_idx] += 1
        
        bins_0 = self.bins[0, :, :]
        bins_1 = self.bins[1, :, :]
        bins_0 = bins_0.astype(float)
        bins_1 = bins_1.astype(float)
        total_bins = bins_0 + bins_1
        ratio = np.zeros_like(total_bins, dtype=float)
        nonzero_mask = total_bins > 0
        ratio[nonzero_mask] = bins_1[nonzero_mask] / total_bins[nonzero_mask]
        binary = (ratio > 0.5).astype(int)
        binary_2 = (bins_1 > 0).astype(int)
        
        img = (binary_2 * 255).astype(np.uint8)
        cv2.imwrite(SYSTEM_PARAMS.files.vein_slide_across_predicted_aggregated_segmentation_mask, img)
        foo = 7

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
        
    def go(self):
        self.linear_interpolation()
        print(f"interpolated positions length: {len(self.all_positions)}")
        # self.marker_tracker.extract_frames(
        #     SYSTEM_PARAMS.files.vein_slide_across
        # )
        # print(f"marker tracker num of extracted frames: {len(self.marker_tracker.frame_markers)}")
        # self.marker_tracker.create_visualization(
        #     out_path=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers,
        #     mode="unpaired-markers",
        #     base_from_file=False
        # )
        # player = VideoPlayer(
        #     in_path=SYSTEM_PARAMS.files.vein_slide_across_extracted_markers
        # )
        # player.run()
        # for i in range(len(self.marker_tracker.frame_markers)):
        #     markers = self.marker_tracker.frame_markers[i]
        #     self.generate_synthetic_image_and_segmentation_mask(i, markers)
        # self.upsample()
        self.aggregate_segmentation_mask()


def main():
    heatmap_generator = HeatmapGenerator()
    heatmap_generator.go()
