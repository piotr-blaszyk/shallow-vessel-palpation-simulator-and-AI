import numpy as np
import cv2
from os import path as osp
import os
import math
import taichi as ti
import glob
import pickle
import json
from difftactile.main.constants import *


class FisheyeModel:
    def __init__(self):
        pass

    @ti.func
    def project_3d_2d(self, a):
        a_norm = a.norm(1e-12)
        cos = a[2] / a_norm
        cos = ti.min(1.0, cos)
        cos = ti.max(-1.0, cos)
        theta = ti.acos(cos)
        x_normalized = ti.select(
            abs(a[0]) < 1e-10, ti.select(a[0] >= 0, 1e-10, -1e-10), a[0]
        )
        omega = ti.atan2(a[1], x_normalized) + ti.math.pi
        r_x = SYSTEM_PARAMS.fisheye_model.focal_length_x * theta
        r_y = SYSTEM_PARAMS.fisheye_model.focal_length_y * theta
        p = ti.Vector([0.0, 0.0])
        p[0] = -r_x * ti.cos(omega) + SYSTEM_PARAMS.fisheye_model.principal_point_x
        p[1] = -r_y * ti.sin(omega) + SYSTEM_PARAMS.fisheye_model.principal_point_y
        return p

    def project_3d_2d_np(self, a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a.reshape(1, 3)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        a_norm = np.maximum(a_norm, 1e-12)
        cos = a[:, 2:3] / a_norm.flatten()
        cos = np.clip(cos, -1.0, 1.0)
        theta = np.arccos(cos)
        x_normalized = np.where(
            np.abs(a[:, 0]) < 1e-10,
            np.where(a[:, 0] >= 0, 1e-10, -1e-10),
            a[:, 0]
        )
        omega = np.arctan2(a[:, 1], x_normalized) + np.pi
        r_x = SYSTEM_PARAMS.fisheye_model.focal_length_x * theta
        r_y = SYSTEM_PARAMS.fisheye_model.focal_length_y * theta
        p = np.zeros((len(a), 2))
        p[:, 0] = -r_x * np.cos(omega) + SYSTEM_PARAMS.fisheye_model.principal_point_x
        p[:, 1] = -r_y * np.sin(omega) + SYSTEM_PARAMS.fisheye_model.principal_point_y
        return p

    def project_pix_to_points(self, p, hemisphere_radius):
        x_norm = (
            p[:, 0] - SYSTEM_PARAMS.fisheye_model.principal_point_x
        ) / SYSTEM_PARAMS.fisheye_model.focal_length_x
        y_norm = (
            p[:, 1] - SYSTEM_PARAMS.fisheye_model.principal_point_y
        ) / SYSTEM_PARAMS.fisheye_model.focal_length_y
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = r
        phi = np.arctan2(y_norm, x_norm)
        points = np.zeros((len(p), 3))
        points[:, 0] = hemisphere_radius * np.sin(theta) * np.cos(phi)
        points[:, 1] = hemisphere_radius * np.sin(theta) * np.sin(phi)
        points[:, 2] = hemisphere_radius * np.cos(theta)
        xy_projections = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        if np.any(xy_projections > hemisphere_radius):
            raise ValueError(
                "Marker projected outside the inner surface of the spherical cap"
            )
        return points

    def project_pix_to_points_3d_plane(
            self,
            ps,
            dist_lens_to_plane=SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface - SYSTEM_PARAMS.trajectory.press_depth_1,
            resolution_down_scaling_factor=None
        ):
        cx = SYSTEM_PARAMS.fisheye_model.principal_point_x
        cy = SYSTEM_PARAMS.fisheye_model.principal_point_y
        fx = SYSTEM_PARAMS.fisheye_model.focal_length_x
        fy = SYSTEM_PARAMS.fisheye_model.focal_length_y
        if resolution_down_scaling_factor is not None:
            cx /= resolution_down_scaling_factor
            cy /= resolution_down_scaling_factor
            fx /= resolution_down_scaling_factor
            fy /= resolution_down_scaling_factor
        x_norm = (
            ps[:, 0] - cx
        ) / fx
        y_norm = (
            ps[:, 1] - cy
        ) / fy
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = r
        phi = np.arctan2(y_norm, x_norm)
        z = dist_lens_to_plane
        r_plane = z * np.tan(theta)
        ps = np.zeros((len(ps), 3))
        ps[:, 0] = r_plane * np.cos(phi)
        ps[:, 1] = r_plane * np.sin(phi)
        ps[:, 2] = z
        return ps

    def get_marker_image(self, img):
        if len(img.shape) == 3:
            source_height, source_width = img.shape[:2]
        else:
            source_height, source_width = img.shape
        scale_x = source_width / SYSTEM_PARAMS.fisheye_model.target_image_width
        scale_y = source_height / SYSTEM_PARAMS.fisheye_model.target_image_height
        params = cv2.SimpleBlobDetector_Params()
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        circle_center = np.array(
            [
                SYSTEM_PARAMS.fisheye_model.circle_centre_x * scale_x,
                SYSTEM_PARAMS.fisheye_model.circle_centre_y * scale_y,
            ]
        )
        circle_radius = SYSTEM_PARAMS.fisheye_model.circle_radius * min(
            scale_x, scale_y
        )
        MarkerCenter = []
        for pt in keypoints:
            point = np.array([pt.pt[0], pt.pt[1]])
            distance = np.linalg.norm(point - circle_center)
            if distance < circle_radius:
                MarkerCenter.append([pt.pt[0], pt.pt[1]])
        MarkerCenter = np.array(MarkerCenter)
        return MarkerCenter, circle_center, circle_radius

    def project_points_to_pix_cv2(self, points3d):
        points3d = np.asarray(points3d, dtype=np.float64)
        if points3d.ndim == 1:
            points3d = points3d.reshape(1, 3)
        points3d = points3d.reshape(-1, 1, 3)
        K = np.array(
            [
                [
                    SYSTEM_PARAMS.fisheye_model.focal_length_x,
                    0,
                    SYSTEM_PARAMS.fisheye_model.principal_point_x,
                ],
                [
                    0,
                    SYSTEM_PARAMS.fisheye_model.focal_length_y,
                    SYSTEM_PARAMS.fisheye_model.principal_point_y,
                ],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        D = np.zeros((4, 1), dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        imgpts, _ = cv2.fisheye.projectPoints(points3d, rvec, tvec, K, D)
        res = imgpts.reshape(-1, 2)
        return res

    def interactive_exploration(self):
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(
                glob.glob(os.path.join(SYSTEM_PARAMS.files.fisheye_model_image_dir, ext))
            )
            image_files.extend(
                glob.glob(
                    os.path.join(
                        SYSTEM_PARAMS.files.fisheye_model_image_dir, ext.upper()
                    )
                )
            )
        image_files.sort()
        if not image_files:
            print(
                f"No image files found in {SYSTEM_PARAMS.files.fisheye_model_image_dir}"
            )
            exit()
        print(f"Found {len(image_files)} images")
        print("Controls: 'j' - previous image, 'l' - next image, 'q' - quit")
        current_index = 0
        while True:
            img_path = image_files[current_index]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                current_index = (current_index + 1) % len(image_files)
                continue
            marker_positions, circle_center, circle_radius = self.get_marker_image(img)
            vis_img = img.copy()
            circle_center_int = (int(circle_center[0]), int(circle_center[1]))
            circle_radius_int = int(circle_radius)
            cv2.circle(
                vis_img,
                circle_center_int,
                circle_radius_int,
                color=(0, 255, 0),
                thickness=2,
            )
            for pos in marker_positions:
                center = (int(pos[0]), int(pos[1]))
                cv2.circle(vis_img, center, radius=5, color=(0, 0, 255), thickness=2)
            filename = os.path.basename(img_path)
            info_text = f"Image {current_index + 1}/{len(image_files)}: {filename}"
            cv2.putText(
                vis_img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis_img,
                f"Markers detected: {len(marker_positions)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Interactive Marker Detection", vis_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("j"):
                current_index = (current_index - 1) % len(image_files)
            elif key == ord("l"):
                current_index = (current_index + 1) % len(image_files)
        cv2.destroyAllWindows()

    def extract_experimental_markers_and_save_to_file(self):
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(
                glob.glob(os.path.join(SYSTEM_PARAMS.files.fisheye_model_image_dir, ext))
            )
            image_files.extend(
                glob.glob(
                    os.path.join(
                        SYSTEM_PARAMS.files.fisheye_model_image_dir, ext.upper()
                    )
                )
            )
        image_files.sort()
        if not image_files:
            print(
                f"No image files found in {SYSTEM_PARAMS.files.fisheye_model_image_dir}"
            )
            exit()
        print(f"Found {len(image_files)} images")
        all_marker_positions = []
        class_labels = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            marker_positions, _, _ = self.get_marker_image(img)
            filename = os.path.basename(img_path)
            try:
                file_num = int(filename.split("press-")[1].split(".")[0])
            except (IndexError, ValueError):
                print(
                    f"Warning: Filename {filename} doesn't match expected format 'press-{{number}}.jpg'"
                )
                continue
            if 0 <= file_num <= 10:
                label = 0
            elif 11 <= file_num <= 20:
                label = 1
            elif 21 <= file_num <= 30:
                label = 2
            elif 31 <= file_num <= 40:
                label = 3
            elif file_num == 41:
                label = 4
            elif file_num == 42:
                label = 5
            elif file_num == 43:
                label = 6
            else:
                print(
                    f"Warning: File number {file_num} doesn't match any label criteria"
                )
                continue
            all_marker_positions.append(marker_positions)
            class_labels.append(label)
            print(f"Processed image {filename} - Found {len(marker_positions)} markers")
        all_marker_positions = np.array(all_marker_positions)
        class_labels = np.array(class_labels)
        results = {
            "marker_positions": all_marker_positions,
            "class_labels": class_labels,
        }
        output_file = SYSTEM_PARAMS.files.vascular_tumour_press_results
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {output_file}")
        print(f"Marker positions shape: {all_marker_positions.shape}")
        print(f"Class labels shape: {class_labels.shape}")

    def save_init_marker_positions(self):
        img = cv2.imread(
            SYSTEM_PARAMS.files.vitactip_photo_default_state, cv2.IMREAD_GRAYSCALE
        )
        if img is None:
            raise FileNotFoundError(
                "Could not find or open "
                + SYSTEM_PARAMS.files.vitactip_photo_default_state
            )
        marker_positions, circle_center, circle_radius = self.get_marker_image(img)
        with open(SYSTEM_PARAMS.files.init_marker_positions, "wb") as f:
            pickle.dump(marker_positions, f)
        print(f"Found {len(marker_positions)} markers")
        print(f"Marker positions saved")
        return marker_positions

    def generate_marker_3d_projection(self):
        with open(SYSTEM_PARAMS.files.init_marker_positions, "rb") as f:
            marker_positions_2d = pickle.load(f)
        A_points = self.project_pix_to_points(
            marker_positions_2d,
            hemisphere_radius=SYSTEM_PARAMS.fisheye_model.shell_outer_r,
        )
        B_points = self.project_pix_to_points(
            marker_positions_2d,
            hemisphere_radius=SYSTEM_PARAMS.fisheye_model.shell_outer_r + 2,
        )
        obj = {
            "A_points": A_points,
            "B_points": B_points,
        }
        with open(SYSTEM_PARAMS.files.biomimetic_tip_points, "wb") as f:
            pickle.dump(obj, f)


def main():
    fisheye_model = FisheyeModel()
    fisheye_model.generate_marker_3d_projection()
