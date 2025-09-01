import os
import glob
import numpy as np
import cv2
from scipy.interpolate import interp1d
import tqdm


class Endgame:
    def __init__(self):
        self.root = "difftactile/manual_or_experimental_data/endgame"
        self.dir = "20250901-131547"
        self.input_dir = os.path.join(self.root, self.dir)
        self.output_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_file_pairs(self):
        avi_files = sorted(glob.glob(os.path.join(self.input_dir, "*.avi")))
        npz_files = sorted(glob.glob(os.path.join(self.input_dir, "*.npz")))
        return avi_files, npz_files

    def interpolate_metadata_and_trim_videos(self):
        avi_files, npz_files = self._get_file_pairs()
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


def main():
    e = Endgame()
    e.interpolate_metadata_and_trim_videos()


if __name__ == '__main__':
    main()
