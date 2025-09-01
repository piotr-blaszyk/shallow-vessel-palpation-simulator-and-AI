import os
import glob
import numpy as np
import cv2
from scipy.interpolate import interp1d
import tqdm
import time
import shutil
from difftactile.data_analysis.experiment.predict_exp import *


class Endgame:
    def __init__(self):
        self.root = "difftactile/manual_or_experimental_data/endgame"
        self.dir = "20250901-131547"
        self.input_dir = os.path.join(self.root, self.dir)
        self.output_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        os.makedirs(self.output_dir, exist_ok=True)

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
            PredictExp.compute_npz_helper(
                video_in=video_in,
                video_out=video_out,
                npz_in=npz_out,
            )
            for npz_path in npz_files:
                base_name = os.path.splitext(os.path.basename(npz_path))[0]
                dst_path = os.path.join(output_dir, f"{base_name}_poses.npz")
                shutil.copy(npz_path, dst_path)

    def reorder_interpolate_markers(self):
        input_dir = os.path.join(self.root, f"{self.dir}_markers")
        output_dir = os.path.join(self.root, f"{self.dir}_reordered_interpolated_markers")
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files_poses, npz_files_markers = self._get_file_pairs_2(input_dir)
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

def main():
    e = Endgame()
    # e.apply_dilation(16)
    e.reorder_interpolate_markers()


if __name__ == "__main__":
    main()
