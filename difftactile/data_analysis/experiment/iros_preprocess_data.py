import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from difftactile.data_analysis.experiment.predict_exp import PredictExp
from difftactile.sensor_model.fisheye_model_no_taichi import FisheyeModelNoTaichi


@dataclass
class TrialConfig:
    trial_id: str
    steaks_above: int
    straw_centers_y_mm: List[float]
    straw_diameters_mm: List[float]


class IrosPreprocessData:
    def __init__(self):
        repo_root = Path(__file__).resolve().parents[3]
        base_dir = (
            repo_root / "difftactile/manual_or_experimental_data/iros_training_data"
        )
        self.input_dir = base_dir / "raw"
        self.spec_path = (
            repo_root
            / "difftactile/manual_or_experimental_data/iros_experiment_spec.md"
        )
        self.output_dir = base_dir / "clean"
        self.tmp_dir = base_dir / "intermediate"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.decimation_step = 30
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
            mode='iros',
        )
        PredictExp.compute_npz_helper2(
            video_in=str(decimated_video),
            video_out=str(markers_reordered_vis_avi),
            npz_in=str(markers_raw_npz),
            npz_out=str(markers_reordered_npz),
            mode='iros',
        )
        with np.load(markers_reordered_npz) as data:
            markers = data["markers"].astype(np.float32)
        return markers

    def process_all_trials(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.input_dir}. "
                "Place trial .avi/.npz pairs there."
            )

        spec_cfg = self._load_spec()
        pairs = self._get_trial_file_pairs(self.input_dir)
        if not pairs:
            raise RuntimeError(f"No .avi/.npz trial pairs found in {self.input_dir}")

        common_trials = sorted(set(spec_cfg.keys()) & set(pairs.keys()))
        if not common_trials:
            raise RuntimeError("No overlapping trial IDs between input folder and spec")

        i = 0
        for trial_id in tqdm(common_trials, desc="Processing trials"):
            i += 1
            cfg = spec_cfg[trial_id]
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


def main():
    processor = IrosPreprocessData()
    processor.process_all_trials()


if __name__ == "__main__":
    main()
