import os
import glob
import numpy as np
import cv2
from scipy.interpolate import interp1d
import tqdm
import time
import shutil
from scipy.spatial.distance import pdist, squareform

from difftactile.data_analysis.experiment.predict_exp import *
from difftactile.main.constants import *


class Iros:
    def __init__(self):
        self.root = "difftactile/manual_or_experimental_data/endgame"
        self.dir = "20250901-131547"
        self.input_dir = os.path.join(self.root, self.dir)
        self.output_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        os.makedirs(self.output_dir, exist_ok=True)
        self.og_theta = SYSTEM_PARAMS.geometry.camera_rotation_angle
        self.cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        self.cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        self.r = SYSTEM_PARAMS.fisheye_model.circle_radius
        th = np.zeros(shape=(10,), dtype=float)
        th[0] = 0
        th[1] = 0
        th[2] = 0
        th[3] = 0
        th[4] = -5
        th[5] = -5
        th[6] = -5
        th[7] = -5
        th[8] = 0
        th[9] = 0
        self.thetas = th

    def _get_file_pairs(self, dir):
        avi_files = sorted(glob.glob(os.path.join(dir, "*.avi")))
        npz_files = sorted(glob.glob(os.path.join(dir, "*.npz")))
        return avi_files, npz_files

    def _get_file_pairs_2(self, dir):
        avi_files = sorted(glob.glob(os.path.join(dir, "*.avi")))
        npz_files_poses = sorted(glob.glob(os.path.join(dir, "*_poses.npz")))
        npz_files_markers = sorted(glob.glob(os.path.join(dir, "*_markers.npz")))
        return avi_files, npz_files_poses, npz_files_markers

    def interpolate_metadata_and_trim_videos(self):
        avi_files, npz_files = self._get_file_pairs(self.input_dir)
        for avi_path, npz_path in zip(avi_files, npz_files):
            npz_data = np.load(npz_path)
            output = npz_data["output"]
            metadata = npz_data["metadata"]
            vy = metadata[3]
            frame_numbers = output[:, 0].astype(int)
            poses = output[:, 1:]
            unique_frames, unique_idx = np.unique(frame_numbers, return_index=True)
            poses_unique = poses[unique_idx]
            interp_funcs = [
                interp1d(
                    unique_frames, poses_unique[:, d], kind="linear", bounds_error=False
                )
                for d in range(poses_unique.shape[1])
            ]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_frames = np.arange(n_frames)
            interp_poses = np.stack([f(all_frames) for f in interp_funcs], axis=1)
            valid_mask = (all_frames >= unique_frames.min()) & (
                all_frames <= unique_frames.max()
            )
            interp_poses = interp_poses[valid_mask]
            valid_frames = all_frames[valid_mask]
            y_range_low, y_range_high = vy - 100 + 20, vy - 20
            y = interp_poses[:, 1]
            y_mask = (y >= y_range_low) & (y <= y_range_high)
            interp_poses = interp_poses[y_mask]
            valid_frames = valid_frames[y_mask]
            output_npz = os.path.join(self.output_dir, os.path.basename(npz_path))
            np.savez(
                output_npz,
                output=interp_poses,
                metadata=metadata,
            )
            output_avi = os.path.join(self.output_dir, os.path.basename(avi_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_avi, fourcc, fps, (w, h))
            frame_idx = 0
            kept_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in valid_frames:
                    writer.write(frame)
                    kept_idx += 1
                frame_idx += 1
            cap.release()
            writer.release()

    def apply_dilation(self, k):
        input_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        output_dir = os.path.join(self.root, f"{self.dir}_dilated")
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files = self._get_file_pairs(input_dir)
        for avi_path, npz_path in zip(avi_files, npz_files):
            npz_data = np.load(npz_path)
            output = npz_data["output"]
            metadata = npz_data["metadata"]
            dilated_output = output[::k]
            output_npz = os.path.join(output_dir, os.path.basename(npz_path))
            np.savez(
                output_npz,
                output=dilated_output,
                metadata=metadata,
            )
            cap = cv2.VideoCapture(avi_path)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS) / k
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_avi = os.path.join(output_dir, os.path.basename(avi_path))
            writer = cv2.VideoWriter(output_avi, fourcc, fps, (w, h))
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % k == 0:
                    writer.write(frame)
                frame_idx += 1
            cap.release()
            writer.release()

    def visualise_videos(self):
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        if not avi_files:
            print("No videos found.")
            return
        video_idx = 0
        frame_idx = 0
        while True:
            avi_path = avi_files[video_idx]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {avi_path}")
                break
            display = frame.copy()
            text = f"Video {video_idx + 1}/{len(avi_files)} | Frame {frame_idx + 1}/{n_frames}"
            cv2.putText(
                display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            cv2.imshow("Video Viewer", display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                video_idx = (video_idx + 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("n"):
                video_idx = (video_idx - 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("k"):
                frame_idx += 1
            elif key == ord("j"):
                frame_idx -= 1
        cv2.destroyAllWindows()
        for i in range(10):
            cv2.waitKey(1)
            time.sleep(0.1)

    def extract_markers(self):
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_markers")
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files = self._get_file_pairs(input_dir)
        for avi_path in avi_files:
            base_name = os.path.splitext(os.path.basename(avi_path))[0]
            video_in = avi_path
            video_out = os.path.join(output_dir, f"{base_name}_markers.avi")
            npz_out = os.path.join(output_dir, f"{base_name}_markers.npz")
            PredictExp.compute_npz_helper3(
                video_in=video_in,
                video_out=video_out,
                npz_out=npz_out,
            )
            for npz_path in npz_files:
                base_name = os.path.splitext(os.path.basename(npz_path))[0]
                dst_path = os.path.join(output_dir, f"{base_name}_poses.npz")
                shutil.copy(npz_path, dst_path)

    def reorder_interpolate_markers(self):
        input_dir = os.path.join(self.root, f"{self.dir}_markers")
        output_dir = os.path.join(
            self.root, f"{self.dir}_reordered_interpolated_markers"
        )
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files_poses, npz_files_markers = self._get_file_pairs_2(
            input_dir
        )
        for avi_path, npz_in in zip(avi_files, npz_files_markers):
            base_name = os.path.splitext(os.path.basename(avi_path))[0]
            video_in = avi_path
            video_out = os.path.join(output_dir, f"{base_name}.avi")
            npz_out = os.path.join(output_dir, f"{base_name}.npz")
            PredictExp.compute_npz_helper2(
                video_in=video_in,
                video_out=video_out,
                npz_in=npz_in,
                npz_out=npz_out,
            )
            for npz_path in npz_files_poses:
                base_name = os.path.splitext(os.path.basename(npz_path))[0]
                dst_path = os.path.join(output_dir, f"{base_name}.npz")
                shutil.copy(npz_path, dst_path)

    def visualise_line_points(self):
        """
        this method needs to be edited such that references to manual annotations are removed
        """
        cx = self.cx
        cy = self.cy
        r = self.r
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        npz_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not avi_files or not npz_files:
            print("No videos or line points found.")
            return
        avi_map = {os.path.splitext(os.path.basename(f))[0]: f for f in avi_files}
        npz_map = {os.path.splitext(os.path.basename(f))[0]: f for f in npz_files}
        common_keys = sorted(set(avi_map.keys()) & set(npz_map.keys()))
        if not common_keys:
            print("No matching video/npz pairs found.")
            return
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        video_idx = 0
        frame_idx = 0
        while True:
            key_base = common_keys[video_idx]
            avi_path = avi_map[key_base]
            npz_path = npz_map[key_base]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {avi_path}")
                break
            data = np.load(npz_path)
            line_points = data["line_points"]
            mask = data["mask"]
            display = frame.copy()
            if frame_idx < line_points.shape[0]:
                for line_idx in range(4):
                    if mask[frame_idx, line_idx]:
                        pts = line_points[frame_idx, line_idx]
                        for x, y in pts.astype(int):
                            cv2.circle(display, (x, y), 2, colors[line_idx], -1)
            cv2.circle(display, (cx, cy), radius=r, color=(0, 255, 255), thickness=3)
            text = f"Video {video_idx + 1}/{len(common_keys)} | Frame {frame_idx + 1}/{n_frames}"
            cv2.putText(
                display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            cv2.imshow("Line Points Viewer", display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                video_idx = (video_idx + 1) % len(common_keys)
                frame_idx = 0
            elif key == ord("n"):
                video_idx = (video_idx - 1) % len(common_keys)
                frame_idx = 0
            elif key == ord("k"):
                frame_idx += 1
            elif key == ord("j"):
                frame_idx -= 1
        cv2.destroyAllWindows()
        for i in range(10):
            cv2.waitKey(1)
            time.sleep(0.1)

    def merge_npz_to_sim_format(self):
        annotations_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        markers_dir = os.path.join(
            self.root, f"{self.dir}_reordered_interpolated_markers"
        )
        output_dir = os.path.join(self.root, f"{self.dir}_sim_format")
        os.makedirs(output_dir, exist_ok=True)
        ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.npz")))
        if not ann_files:
            print("No annotation npz files found.")
            return
        for ann_path in ann_files:
            base = os.path.splitext(os.path.basename(ann_path))[0]
            marker_path = os.path.join(markers_dir, f"{base}_markers.npz")
            if not os.path.exists(marker_path):
                print(f"Skipping {base}: marker file not found at {marker_path}")
                continue
            line_point_data = np.load(ann_path)
            marker_data = np.load(marker_path)
            markers = marker_data["markers"]
            markers_mask = marker_data["markers_mask"]
            vein_polyline = line_point_data["line_points"]
            vein_polyline_mask = line_point_data["mask"]
            a, b, c, d = vein_polyline.shape
            vein_polyline_mask = np.broadcast_to(
                vein_polyline_mask[..., None], (a, b, c)
            )
            out_path = os.path.join(output_dir, f"{base}.npz")
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
            )


    def merge_npz_sim_format_poses(self):
        sim_dir = os.path.join(self.root, f"{self.dir}_sim_format")
        dilated_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_sim_format_poses")
        os.makedirs(output_dir, exist_ok=True)

        sim_files = sorted(glob.glob(os.path.join(sim_dir, "*.npz")))
        if not sim_files:
            print("No sim_format npz files found.")
            return

        for sim_path in sim_files:
            base = os.path.basename(sim_path)
            dilated_path = os.path.join(dilated_dir, base)
            if not os.path.exists(dilated_path):
                print(f"Skipping {base}: matching dilated npz not found.")
                continue

            sim_format_data = np.load(sim_path)
            dilated_data = np.load(dilated_path)

            markers = sim_format_data["markers"]
            markers_mask = sim_format_data["markers_mask"]
            vein_polyline = sim_format_data["vein_polyline"]
            vein_polyline_mask = sim_format_data["vein_polyline_mask"]
            poses = dilated_data["output"]
            metadata = dilated_data["metadata"]

            out_path = os.path.join(output_dir, base)
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
                poses=poses,
                metadata=metadata,
            )


def main():
    e = Iros()
    e.interpolate_metadata_and_trim_videos()
    e.apply_dilation()
    e.visualise_videos()
    e.extract_markers()
    e.reorder_interpolate_markers()
    e.visualise_line_points()
    e.merge_npz_to_sim_format()
    e.merge_npz_sim_format_poses()


if __name__ == "__main__":
    main()
