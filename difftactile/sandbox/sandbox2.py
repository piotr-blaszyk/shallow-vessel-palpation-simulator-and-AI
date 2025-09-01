import os
import glob
import numpy as np
import cv2
from scipy.interpolate import interp1d
from tqdm import tqdm

class Endgame:
    def __init__(self):
        self.root = 'difftactile/manual_or_experimental_data/endgame'
        self.dir = '20250901-131547'
        self.input_dir = os.path.join(self.root, self.dir)
        self.output_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_file_pairs(self, directory):
        avi_files = sorted(glob.glob(os.path.join(directory, "*.avi")))
        npz_files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        return avi_files, npz_files

    def interpolate_metadata_and_trim_videos(self):
        avi_files, npz_files = self._get_file_pairs(self.input_dir)

        for avi_path, npz_path in tqdm(zip(avi_files, npz_files),
                                       total=min(len(avi_files), len(npz_files)),
                                       desc="Interpolating & trimming"):
            # --- Load npz ---
            npz_data = np.load(npz_path, allow_pickle=True)
            output = npz_data["output"]  # shape (k, 1+6)
            metadata = npz_data["metadata"]  # shape (5,)
            vy = metadata[3]

            frame_numbers = output[:, 0].astype(int)
            poses = output[:, 1:]

            # --- Interpolation setup ---
            unique_frames, unique_idx = np.unique(frame_numbers, return_index=True)
            poses_unique = poses[unique_idx]

            interp_funcs = [
                interp1d(unique_frames, poses_unique[:, d], kind="linear", bounds_error=False)
                for d in range(poses_unique.shape[1])
            ]

            # --- Video handling ---
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_frames = np.arange(n_frames)

            # Interpolate poses for all video frames
            interp_poses = np.stack([f(all_frames) for f in interp_funcs], axis=1)

            # Valid mask (no extrapolation)
            valid_mask = (all_frames >= unique_frames.min()) & (all_frames <= unique_frames.max())
            interp_poses = interp_poses[valid_mask]
            valid_frames = all_frames[valid_mask]

            # --- Y filtering ---
            y_range_low, y_range_high = vy - 100 + 20, vy - 20  # (vy-80, vy-20)
            y = interp_poses[:, 1]  # assuming pose format [x,y,z,...]
            y_mask = (y >= y_range_low) & (y <= y_range_high)

            interp_poses = interp_poses[y_mask]
            valid_frames = valid_frames[y_mask]

            # --- Save npz ---
            output_npz = os.path.join(self.output_dir, os.path.basename(npz_path))
            np.savez(
                output_npz,
                output=interp_poses,  # shape (n,6)
                metadata=metadata,
            )

            # --- Save trimmed video ---
            output_avi = os.path.join(self.output_dir, os.path.basename(avi_path))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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

            print(f"Processed {os.path.basename(avi_path)} -> {kept_idx} frames kept")

    def apply_dilation(self, k):
        input_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        output_dir = os.path.join(self.root, f"{self.dir}_dilated")
        os.makedirs(output_dir, exist_ok=True)

        avi_files, npz_files = self._get_file_pairs(input_dir)

        for avi_path, npz_path in tqdm(zip(avi_files, npz_files),
                                       total=min(len(avi_files), len(npz_files)),
                                       desc=f"Dilating (k={k})"):
            # --- Load npz ---
            npz_data = np.load(npz_path, allow_pickle=True)
            output = npz_data["output"]  # shape (n, 6)
            metadata = npz_data["metadata"]

            # Apply dilation to poses
            dilated_output = output[::k]

            # Save new npz
            output_npz = os.path.join(output_dir, os.path.basename(npz_path))
            np.savez(
                output_npz,
                output=dilated_output,
                metadata=metadata,
            )

            # --- Handle video ---
            cap = cv2.VideoCapture(avi_path)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS) / k  # adjust FPS to reflect dilation
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

            print(f"Dilated {os.path.basename(avi_path)} -> stride {k}")

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

            # Clamp frame index to valid range
            frame_idx = max(0, min(frame_idx, n_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {avi_path}")
                break

            # Overlay text with frame info
            display = frame.copy()
            text = f"Video {video_idx+1}/{len(avi_files)} | Frame {frame_idx+1}/{n_frames}"
            cv2.putText(display, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Video Viewer", display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('m'):  # next video
                video_idx = (video_idx + 1) % len(avi_files)
                frame_idx = 0
            elif key == ord('n'):  # previous video
                video_idx = (video_idx - 1) % len(avi_files)
                frame_idx = 0
            elif key == ord('k'):  # next frame
                frame_idx += 1
            elif key == ord('j'):  # previous frame
                frame_idx -= 1

        cv2.destroyAllWindows()
