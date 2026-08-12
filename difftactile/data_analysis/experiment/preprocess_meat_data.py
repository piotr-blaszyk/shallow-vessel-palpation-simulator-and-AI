import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from difftactile.data_analysis.experiment.predict_exp import PredictExp
from difftactile.main.display import (
    destroy_windows, imshow, is_interactive, run_frame_browser, wait_key,
)
from difftactile.sensor_model.fisheye_model_no_taichi import FisheyeModelNoTaichi


@dataclass
class TrialConfig:
    trial_id: str
    steaks_above: int
    straw_centers_y_mm: List[float]
    straw_diameters_mm: List[float]


class MeatPreprocessData:
    def __init__(self):
        repo_root = Path(__file__).resolve().parents[3]
        base_dir = (
            repo_root / "difftactile/manual_or_experimental_data/meat_training_data"
        )
        self.input_dir = base_dir / "raw"
        self.spec_path = (
            repo_root
            / "difftactile/manual_or_experimental_data/meat_experiment_spec.md"
        )
        self.output_dir = base_dir / "clean"
        self.tmp_dir = base_dir / "intermediate"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.decimation_step = 15
        self.expected_w = 1920
        self.expected_h = 1080
        self.coarse_w = 240
        self.coarse_h = 135
        self.points_per_side = 25
        self.world_sample_points = 801
        self.camera_to_straw_z_offset_mm = 24.0

    @staticmethod
    def _load_pose_rows(npz_path: Path) -> np.ndarray:
        with np.load(npz_path) as data:
            if "output" in data:
                arr = data["output"]
            elif "arr_0" in data:
                arr = data["arr_0"]
            else:
                keys = list(data.keys())
                if not keys:
                    raise ValueError(f"No arrays found in {npz_path}")
                arr = data[keys[0]]
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Expected shape (n, >=4) in {npz_path}, got {arr.shape}")
        return arr

    @staticmethod
    def _interpolate_poses_to_video_frames(
        pose_rows: np.ndarray,
        n_video_frames: int,
    ) -> np.ndarray:
        frame_numbers = pose_rows[:, 0].astype(np.int32)
        poses = pose_rows[:, 1:]
        unique_frames, unique_idx = np.unique(frame_numbers, return_index=True)
        unique_poses = poses[unique_idx]
        if len(unique_frames) < 2:
            raise ValueError("Need at least 2 pose rows with distinct frame numbers")

        all_video_frames = np.arange(n_video_frames, dtype=np.int32)
        interp_pose_dims = []
        for dim in range(unique_poses.shape[1]):
            func = interp1d(
                unique_frames,
                unique_poses[:, dim],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_pose_dims.append(func(all_video_frames))
        all_interp = np.stack(interp_pose_dims, axis=1)
        valid = ~np.isnan(all_interp).any(axis=1)
        return all_interp, valid

    def _decimate_video_and_poses(
        self,
        avi_path: Path,
        pose_npz_path: Path,
        trial_id: str,
    ) -> Tuple[Path, np.ndarray]:
        pose_rows = self._load_pose_rows(pose_npz_path)

        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {avi_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w != self.expected_w or h != self.expected_h:
            print(
                f"Warning [{trial_id}]: expected 1920x1080, got {w}x{h}. "
                "Proceeding anyway."
            )

        poses_interp, valid_interp = self._interpolate_poses_to_video_frames(
            pose_rows,
            n_frames,
        )
        target_indices = np.arange(0, n_frames, self.decimation_step, dtype=np.int32)
        keep_indices = target_indices[valid_interp[target_indices]]

        decimated_poses = poses_interp[keep_indices].astype(np.float32)

        out_video = self.tmp_dir / f"{trial_id}_decimated.avi"
        out_fps = fps / self.decimation_step if fps > 0 else 1.0
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"XVID"),
            out_fps,
            (w, h),
        )

        keep_set = set(int(i) for i in keep_indices.tolist())
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in keep_set:
                writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        return out_video, decimated_poses

    @staticmethod
    def _sample_polyline_equally(points_xy: np.ndarray, n_out: int) -> np.ndarray:
        if points_xy.shape[0] < 2:
            return np.repeat(points_xy[:1], n_out, axis=0)
        deltas = np.diff(points_xy, axis=0)
        seg_len = np.linalg.norm(deltas, axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg_len)))
        total = arc[-1]
        if total < 1e-9:
            return np.repeat(points_xy[:1], n_out, axis=0)
        targets = np.linspace(0.0, total, n_out)
        x = np.interp(targets, arc, points_xy[:, 0])
        y = np.interp(targets, arc, points_xy[:, 1])
        return np.stack([x, y], axis=1)

    @staticmethod
    def _world_to_camera(points_world: np.ndarray, cam_xyz_world: np.ndarray) -> np.ndarray:
        rel = points_world - cam_xyz_world[None, :]
        points_cam = np.empty_like(rel)
        points_cam[:, 0] = rel[:, 1]
        points_cam[:, 1] = rel[:, 0]
        points_cam[:, 2] = -rel[:, 2]
        return points_cam

    def _visible_x_world_range(self, x_cam_mm: float) -> Tuple[float, float]:
        xs = np.linspace(0, self.expected_w - 1, 120)
        ys = np.linspace(0, self.expected_h - 1, 80)

        boundary = []
        for x in xs:
            boundary.append([x, 0])
            boundary.append([x, self.expected_h - 1])
        for y in ys:
            boundary.append([0, y])
            boundary.append([self.expected_w - 1, y])
        boundary = np.asarray(boundary, dtype=np.float64)

        plane_pts = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
            boundary,
            dist_lens_to_plane=self.camera_to_straw_z_offset_mm,
        )
        y_cam = plane_pts[:, 1]
        x_wf = x_cam_mm + y_cam
        x_margin = 20.0
        return float(np.min(x_wf) - x_margin), float(np.max(x_wf) + x_margin)

    def _build_frame_mask(
        self,
        cfg: TrialConfig,
        pose_xyz: np.ndarray,
    ) -> np.ndarray:
        mask_full = np.zeros((self.expected_h, self.expected_w), dtype=np.uint8)
        if len(cfg.straw_centers_y_mm) == 0:
            return np.zeros((self.coarse_h, self.coarse_w), dtype=np.uint8)

        x_cam, y_cam, z_tip = pose_xyz[:3].astype(np.float64)
        x_wf_min, x_wf_max = self._visible_x_world_range(x_cam)
        x_samples = np.linspace(x_wf_min, x_wf_max, self.world_sample_points)

        z_straw = z_tip - (cfg.steaks_above * 5.0)
        z_cam = z_straw + self.camera_to_straw_z_offset_mm
        cam_xyz_world = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

        for y_center, diameter in zip(cfg.straw_centers_y_mm, cfg.straw_diameters_mm):
            y1 = y_center - (diameter / 2.0)
            y2 = y_center + (diameter / 2.0)

            side1_world = np.stack(
                [x_samples, np.full_like(x_samples, y1), np.full_like(x_samples, z_straw)],
                axis=1,
            )
            side2_world = np.stack(
                [x_samples, np.full_like(x_samples, y2), np.full_like(x_samples, z_straw)],
                axis=1,
            )

            side1_cam = self._world_to_camera(side1_world, cam_xyz_world)
            side2_cam = self._world_to_camera(side2_world, cam_xyz_world)

            side1_px_dense = FisheyeModelNoTaichi.project_3d_2d_np(side1_cam)
            side2_px_dense = FisheyeModelNoTaichi.project_3d_2d_np(side2_cam)

            side1_px = self._sample_polyline_equally(side1_px_dense, self.points_per_side)
            side2_px = self._sample_polyline_equally(side2_px_dense, self.points_per_side)

            polygon = np.vstack([side1_px, side2_px[::-1]])
            polygon_int = np.round(polygon).astype(np.int32)
            cv2.fillPoly(mask_full, [polygon_int], color=1)

        coarse = cv2.resize(
            mask_full,
            (self.coarse_w, self.coarse_h),
            interpolation=cv2.INTER_AREA,
        )
        return (coarse > 0).astype(np.uint8)

    def _labels_from_markers_and_masks(
        self,
        markers_xy: np.ndarray,
        coarse_masks: np.ndarray,
    ) -> np.ndarray:
        n_frames, n_markers, _ = markers_xy.shape
        labels = np.zeros((n_frames, n_markers), dtype=np.int32)
        for t in range(n_frames):
            mask = coarse_masks[t]
            x = np.clip((markers_xy[t, :, 0] // 8).astype(np.int32), 0, self.coarse_w - 1)
            y = np.clip((markers_xy[t, :, 1] // 8).astype(np.int32), 0, self.coarse_h - 1)
            labels[t] = mask[y, x].astype(np.int32)
        return labels

    @staticmethod
    def _parse_trial_config(trial_id: str, description: str) -> TrialConfig:
        text = description.lower()
        if "no straw" in text:
            return TrialConfig(trial_id, steaks_above=0, straw_centers_y_mm=[], straw_diameters_mm=[])

        if "on top" in text:
            steaks_above = 0
        else:
            m = re.search(r"beneath\s+(\d+)\s+steak", text)
            if not m:
                raise ValueError(f"Cannot parse depth from description: {description}")
            steaks_above = int(m.group(1))

        if "silicone" in text:
            return TrialConfig(
                trial_id,
                steaks_above=steaks_above,
                straw_centers_y_mm=[-285.0],
                straw_diameters_mm=[8.0],
            )

        if "3 straws" in text:
            return TrialConfig(
                trial_id,
                steaks_above=steaks_above,
                straw_centers_y_mm=[-285.0, -265.0, -305.0],
                straw_diameters_mm=[6.0, 6.0, 6.0],
            )

        if "2 straws" in text:
            return TrialConfig(
                trial_id,
                steaks_above=steaks_above,
                straw_centers_y_mm=[-275.0, -295.0],
                straw_diameters_mm=[6.0, 6.0],
            )

        return TrialConfig(
            trial_id,
            steaks_above=steaks_above,
            straw_centers_y_mm=[-285.0],
            straw_diameters_mm=[6.0],
        )

    def _load_spec(self) -> Dict[str, TrialConfig]:
        if not self.spec_path.exists():
            raise FileNotFoundError(f"Spec file not found: {self.spec_path}")
        mapping: Dict[str, TrialConfig] = {}
        for line in self.spec_path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            m = re.match(r"-\s*(\d{8}-\d{6}):\s*(.+)$", line)
            if not m:
                continue
            trial_id = m.group(1)
            description = m.group(2)
            description = description.split("(")[0].strip()
            description = description.split("-")[0].strip()
            mapping[trial_id] = self._parse_trial_config(trial_id, description)
        return mapping

    @staticmethod
    def _get_trial_file_pairs(input_dir: Path) -> Dict[str, Tuple[Path, Path]]:
        avi_map = {p.stem: p for p in sorted(input_dir.glob("*.avi"))}
        npz_map = {p.stem: p for p in sorted(input_dir.glob("*.npz"))}
        keys = sorted(set(avi_map.keys()) & set(npz_map.keys()))
        return {k: (avi_map[k], npz_map[k]) for k in keys}

    def _extract_reordered_markers(self, decimated_video: Path, trial_id: str) -> np.ndarray:
        markers_raw_npz = self.tmp_dir / f"{trial_id}_markers_raw.npz"
        markers_vis_avi = self.tmp_dir / f"{trial_id}_markers_vis.avi"
        markers_reordered_npz = self.tmp_dir / f"{trial_id}_markers_reordered.npz"
        markers_reordered_vis_avi = self.tmp_dir / f"{trial_id}_markers_reordered_vis.avi"

        PredictExp.compute_npz_helper3(
            video_in=str(decimated_video),
            video_out=str(markers_vis_avi),
            npz_out=str(markers_raw_npz),
            mode='meat',
        )
        PredictExp.compute_npz_helper2(
            video_in=str(decimated_video),
            video_out=str(markers_reordered_vis_avi),
            npz_in=str(markers_raw_npz),
            npz_out=str(markers_reordered_npz),
            mode='meat',
        )
        with np.load(markers_reordered_npz) as data:
            markers = data["markers"].astype(np.float32)
        return markers

    @staticmethod
    def _write_marker_labels_video(
        video_in_path: Path,
        markers_xy: np.ndarray,
        marker_labels: np.ndarray,
        video_out_path: Path,
    ) -> None:
        cap = cv2.VideoCapture(str(video_in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for label visualization: {video_in_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(video_out_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps if fps > 0 else 1.0,
            (w, h),
        )

        n_frames = markers_xy.shape[0]
        for t in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            for marker_idx in range(markers_xy.shape[1]):
                x, y = markers_xy[t, marker_idx]
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                xi = int(round(float(x)))
                yi = int(round(float(y)))
                if xi < 0 or yi < 0 or xi >= w or yi >= h:
                    continue

                color = (0, 0, 255) if int(marker_labels[t, marker_idx]) == 1 else (0, 255, 0)
                cv2.circle(frame, (xi, yi), 4, color, -1)

            writer.write(frame)

        cap.release()
        writer.release()

    def _bundled_trial_inputs(self, trial_id: str):
        """The shipped decimated video and poses for a trial, if present.

        The data bundle carries `clean/<trial>/frames.mp4` (the 26 frames that
        survived decimation) and `frames_poses.npz` (their robot poses), which
        together are everything the rest of the pipeline needs. Starting from
        them makes preprocessing reproducible from the published bundle alone,
        instead of requiring the 1.6 GB raw archive just to redo a decimation
        whose output is already known.

        Returns (video_path, poses) or None when the trial is not shipped.
        """
        trial_dir = self.output_dir / trial_id
        video = trial_dir / "frames.mp4"
        poses_npz = trial_dir / "frames_poses.npz"
        if not video.exists() or not poses_npz.exists():
            return None
        with np.load(poses_npz) as d:
            poses = d["poses"]
        return video, poses

    def process_all_trials(self, prefer_bundled=True):
        """Derive marker positions and vessel labels for every meat trial.

        With `prefer_bundled` (the default) a trial is processed from the
        decimated video shipped in the data bundle when one is available, and
        only falls back to the raw recording otherwise. Pass False, or set
        DIFFTACTILE_MEAT_FROM_RAW=1, to always re-decimate from `raw/`.

        NOTE ON EXACTNESS. `frames.mp4` is H.264 (CRF 26), so marker detection
        sees very slightly different pixels than it did on the lossless raw
        recording. Measured over all 23 trials, the recomputed marker positions
        move by a median of 0.03 px and a 99th percentile of 0.47 px against an
        inter-marker spacing of ~55 px, and 20 of 23 trials come out with
        bit-identical labels; the other three differ in 1, 3 and 12 labels out of
        3302 (16 of ~76000 overall, 0.02%). That is well inside the labelling's
        own accuracy - the paper already notes a visible offset between the
        kinematics-derived labels and the actual straw - but it does mean this
        path *regenerates* the dataset rather than reproducing it bit-for-bit.
        The published `marker_positions.npz` / `marker_labels.npz` in the bundle
        remain the authoritative artifacts; use DIFFTACTILE_MEAT_FROM_RAW=1 with
        the raw archive if you need an exact rebuild.
        """
        if os.environ.get("DIFFTACTILE_MEAT_FROM_RAW") == "1":
            prefer_bundled = False

        spec_cfg = self._load_spec()

        # Trials reachable without the raw archive.
        bundled_ids = set()
        if prefer_bundled and self.output_dir.is_dir():
            bundled_ids = {
                t.name for t in self.output_dir.iterdir()
                if t.is_dir() and self._bundled_trial_inputs(t.name) is not None
            }

        pairs = {}
        if self.input_dir.exists():
            pairs = self._get_trial_file_pairs(self.input_dir)

        available = (set(pairs) | bundled_ids) & set(spec_cfg.keys())
        if not available:
            raise FileNotFoundError(
                "No meat trials to process.\n"
                f"Looked for raw .avi/.npz pairs in {self.input_dir} and for "
                f"bundled frames.mp4 + frames_poses.npz under {self.output_dir}.\n"
                "Restore the data bundle, or place the raw recordings in raw/."
            )
        common_trials = sorted(available)
        if bundled_ids:
            print(
                f"{len(bundled_ids & available)} trial(s) from the bundled decimated "
                f"videos, {len(available - bundled_ids)} from raw recordings."
            )

        i = 0
        for trial_id in tqdm(common_trials, desc="Processing trials"):
            i += 1
            cfg = spec_cfg[trial_id]

            bundled = self._bundled_trial_inputs(trial_id) if prefer_bundled else None
            if bundled is not None:
                # Already decimated when the bundle was built; frames and poses
                # are aligned by construction.
                decimated_video, decimated_poses = bundled
            else:
                avi_path, pose_npz_path = pairs[trial_id]
                decimated_video, decimated_poses = self._decimate_video_and_poses(
                    avi_path=avi_path,
                    pose_npz_path=pose_npz_path,
                    trial_id=trial_id,
                )

            markers_xy = self._extract_reordered_markers(
                decimated_video=decimated_video,
                trial_id=trial_id,
            )

            n_frames = min(len(decimated_poses), markers_xy.shape[0])
            if n_frames == 0:
                raise Exception(f"Trial {trial_id} has no valid decimated frames")
            markers_xy = markers_xy[:n_frames]
            decimated_poses = decimated_poses[:n_frames]

            coarse_masks = np.zeros((n_frames, self.coarse_h, self.coarse_w), dtype=np.uint8)
            for t in range(n_frames):
                coarse_masks[t] = self._build_frame_mask(cfg, decimated_poses[t, :3])

            labels = self._labels_from_markers_and_masks(markers_xy, coarse_masks)

            trial_out_dir = self.output_dir / trial_id
            trial_out_dir.mkdir(parents=True, exist_ok=True)

            np.savez(
                trial_out_dir / "marker_positions.npz",
                marker_positions=markers_xy,
            )
            np.savez(
                trial_out_dir / "marker_labels.npz",
                marker_labels=labels,
            )

            # The pre-rendered overlay video is optional: `browse_annotations()`
            # composites the same thing live from frames.mp4 + the labels, at a
            # fraction of the size (the .avi overlays were 81 MB across trials).
            # Set DIFFTACTILE_MEAT_WRITE_OVERLAY=1 to write them anyway.
            if os.environ.get("DIFFTACTILE_MEAT_WRITE_OVERLAY") == "1":
                self._write_marker_labels_video(
                    video_in_path=decimated_video,
                    markers_xy=markers_xy,
                    marker_labels=labels,
                    video_out_path=trial_out_dir / "marker_labels.avi",
                )


    def browse_annotations(self):
        """Step through the meat trials' marker annotations frame by frame.

        The counterpart to the silicone `annotate()` viewer, and the tool that
        produces Fig. "annotation-line"(d) of the paper: ground-truth vessel
        labels (red = vessel present, green = absent) over the marker grid.

        Meat labels are derived analytically from robot kinematics and the straw
        geometry in `meat_experiment_spec.md` rather than by clicking, so this is
        a *review* tool - there is nothing to hand-label.

        Labels are drawn over the real camera frames in
        `clean/<trial>/frames.mp4`, which the bundle ships: 26 frames per trial,
        exactly the ones preprocessing kept, so frame i of the video is frame i
        of `marker_labels.npz`. If that video is missing (an older bundle) the
        markers are drawn on a plain canvas instead, which still shows the label
        geometry but not the meat underneath.

        This is the view behind Fig. "annotation-line"(d) of the paper.

        Keys: m / n next / previous trial, k / j next / previous frame, q quit.
        """
        if not is_interactive():
            print(
                "Skipping browse_annotations(): it is a viewer with nothing to "
                "show unattended. Set DIFFTACTILE_INTERACTIVE=1 to open it."
            )
            return

        trials = sorted(p for p in self.output_dir.iterdir() if p.is_dir())
        trials = [
            t for t in trials
            if (t / "marker_positions.npz").exists() and (t / "marker_labels.npz").exists()
        ]
        if not trials:
            print(f"No processed meat trials found in {self.output_dir}.")
            return
        print(f"{len(trials)} meat trials. Keys: m/n trial, k/j frame, q quit.")

        # The markers are stored in full camera pixel coordinates; everything is
        # scaled down so a 1920x1080 frame fits on screen.
        scale = 0.5
        canvas_w = int(self.expected_w * scale)
        canvas_h = int(self.expected_h * scale)

        def build_trial(trial_dir):
            """Render every frame of a trial up front, ready to display.

            A trial is only ~26 frames, so the whole overlaid sequence is
            composited once and kept in memory (~26 x 960x540x3 = 40 MB). Doing
            the work per keypress instead would mean an H.264 seek and GOP
            re-decode, an .npz reload and a resize on every step, which is what
            made navigation lag by about a second per frame.
            """
            with np.load(trial_dir / "marker_positions.npz") as d:
                markers = d["marker_positions"]
            with np.load(trial_dir / "marker_labels.npz") as d:
                labels = d["marker_labels"]
            n_frames = markers.shape[0]

            # Decode sequentially - no seeking - which is both correct and fast.
            frames = []
            video_path = trial_dir / "frames.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(cv2.resize(frame, (canvas_w, canvas_h)))
                    cap.release()

            rendered = []
            for t in range(n_frames):
                if t < len(frames):
                    canvas = frames[t].copy()
                else:
                    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                frame_markers = markers[t]
                frame_labels = labels[t]
                for marker_idx in range(frame_markers.shape[0]):
                    x, y = frame_markers[marker_idx]
                    # BGR: red where a vessel is present, green where it is absent.
                    colour = (0, 0, 255) if frame_labels[marker_idx] == 1 else (0, 255, 0)
                    centre = (int(x * scale), int(y * scale))
                    # Dark outline first, so the dots stay readable over the
                    # bright specular highlights on the meat.
                    cv2.circle(canvas, centre, 6, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, centre, 4, colour, -1, cv2.LINE_AA)

                text = (
                    f"[{trial_dir.name}] Frame {t + 1}/{n_frames} | "
                    f"vessel markers {int(frame_labels.sum())}   "
                    f"(m/n trial, k/j frame, q quit)"
                )
                cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                rendered.append(canvas)
            return rendered, len(frames) > 0

        # Trials are rendered on first visit and kept, so revisiting one is
        # instant. Rendering all 23 up front would cost ~1 GB of RAM.
        trial_cache = {}

        def get_trial(trial_dir):
            if trial_dir not in trial_cache:
                print(f"rendering {trial_dir.name} ...", flush=True)
                trial_cache[trial_dir] = build_trial(trial_dir)
            return trial_cache[trial_dir]

        # Mutable browse state, shared with the callbacks below.
        state = {"trial": 0, "frame": 0, "warned": False}

        def render():
            """Current frame, or None to end the loop."""
            trial_dir = trials[state["trial"]]
            frames, had_video = get_trial(trial_dir)
            if not had_video and not state["warned"]:
                print(
                    "No clean/<trial>/frames.mp4 found - drawing markers on a "
                    "plain canvas. Re-restore the data bundle to get the "
                    "camera frames."
                )
                state["warned"] = True
            if not frames:
                return None
            state["frame"] = max(0, min(state["frame"], len(frames) - 1))
            return frames[state["frame"]]

        def on_key(key):
            """Map a keypress to a state change; see the docstring for bindings."""
            n_frames = len(get_trial(trials[state["trial"]])[0])
            if key == ord("q"):
                return "quit"
            if key == ord("m"):
                state["trial"] = (state["trial"] + 1) % len(trials)
                state["frame"] = 0
                return "redraw"
            if key == ord("n"):
                state["trial"] = (state["trial"] - 1) % len(trials)
                state["frame"] = 0
                return "redraw"
            if key == ord("k"):
                new_frame = min(state["frame"] + 1, n_frames - 1)
            elif key == ord("j"):
                new_frame = max(state["frame"] - 1, 0)
            else:
                return None
            if new_frame == state["frame"]:
                return None  # already at the end; nothing to repaint
            state["frame"] = new_frame
            return "redraw"

        run_frame_browser(cv2, "Meat annotations", render, on_key)


def main():
    processor = MeatPreprocessData()
    processor.process_all_trials()


def browse():
    """Entrypoint for reviewing the meat annotations (see browse_annotations)."""
    MeatPreprocessData().browse_annotations()


if __name__ == "__main__":
    main()
