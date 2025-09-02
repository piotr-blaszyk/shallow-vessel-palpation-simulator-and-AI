import os
import glob
import numpy as np
import cv2
from scipy.interpolate import interp1d
import tqdm
import time
import shutil
from difftactile.data_analysis.experiment.predict_exp import *
from difftactile.main.constants import *


class Endgame:
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

    def annotate(self):
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_annotations")
        os.makedirs(output_dir, exist_ok=True)
        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        if not avi_files:
            print("No videos found.")
            return
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        annotations = {}
        for avi_path in avi_files:
            base = os.path.splitext(os.path.basename(avi_path))[0]
            pkl_path = os.path.join(output_dir, f"{base}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    annotations[avi_path] = pickle.load(f)
            else:
                cap = cv2.VideoCapture(avi_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                annotations[avi_path] = [[] for _ in range(n_frames)]
        video_idx = 0
        frame_idx = 0

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                avi_path = avi_files[video_idx]
                frame_annots = annotations[avi_path][frame_idx]
                if len(frame_annots) < 4:
                    annotations[avi_path][frame_idx].append([x, y])
                    print(f"Added point {len(frame_annots)}/{4}")
                else:
                    print("Maximum 4 points allowed.")

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", mouse_callback)
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
            frame_annots = annotations[avi_path][frame_idx]
            for idx, pt in enumerate(frame_annots):
                color = colors[idx % len(colors)]
                cv2.circle(display, (int(pt[0]), int(pt[1])), 6, color, -1)
            text = f"Video {video_idx + 1}/{len(avi_files)} | Frame {frame_idx + 1}/{n_frames} | Annotations {len(frame_annots)}/4"
            cv2.putText(
                display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            cv2.imshow("Annotator", display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                for avi_path in avi_files:
                    base = os.path.splitext(os.path.basename(avi_path))[0]
                    pkl_path = os.path.join(output_dir, f"{base}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(annotations[avi_path], f)
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
            elif key == ord("d"):
                annotations[avi_path][frame_idx] = []
                print("Deleted annotations for this frame.")
            elif key == ord("p"):
                for avi_path in avi_files:
                    base = os.path.splitext(os.path.basename(avi_path))[0]
                    pkl_path = os.path.join(output_dir, f"{base}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(annotations[avi_path], f)
                print("Annotations saved.")
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
            time.sleep(0.1)

    def annotations_to_line_points(self):
        cx = self.cx
        cy = self.cy
        r = self.r
        ann_dir = os.path.join(self.root, f"{self.dir}_annotations")
        out_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        os.makedirs(out_dir, exist_ok=True)
        pkl_files = sorted(glob.glob(os.path.join(ann_dir, "*.pkl")))
        if not pkl_files:
            print("No annotation files found.")
            return
        C = np.array([float(cx), float(cy)], dtype=np.float64)
        R2 = float(r) ** 2
        max_lines = 4
        num_pts = 50
        for i in range(len(pkl_files)):
            th = np.deg2rad(self.thetas[i])
            u = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
            pkl_path = pkl_files[i]
            base = os.path.splitext(os.path.basename(pkl_path))[0]
            out_path = os.path.join(out_dir, f"{base}.npz")
            with open(pkl_path, "rb") as f:
                ann = pickle.load(f)
            num_frames = len(ann)
            line_points = np.zeros(
                (num_frames, max_lines, num_pts, 2), dtype=np.float32
            )
            mask = np.zeros((num_frames, max_lines), dtype=bool)
            for fi, frame_pts in enumerate(ann):
                for li, pt in enumerate(frame_pts[:max_lines]):
                    P0 = np.array(pt, dtype=np.float64)
                    t0 = -np.dot(P0 - C, u)
                    F = P0 + t0 * u
                    d2 = np.sum((F - C) ** 2)
                    if d2 > R2:
                        continue
                    L = float(np.sqrt(max(R2 - d2, 0.0)))
                    A = F - L * u
                    B = F + L * u
                    xs = np.linspace(A[0], B[0], num_pts, dtype=np.float32)
                    ys = np.linspace(A[1], B[1], num_pts, dtype=np.float32)
                    line_points[fi, li, :, 0] = xs
                    line_points[fi, li, :, 1] = ys
                    mask[fi, li] = True
            np.savez(
                out_path,
                line_points=line_points,
                mask=mask,
            )

    def visualise_line_points(self):
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
    e = Endgame()
    e.merge_npz_sim_format_poses()


if __name__ == "__main__":
    main()
