"""Bird's-eye vessel maps: 2D -> 3D -> 2D reprojection of per-marker predictions.

WHAT IT DOES. A trained GNN predicts, for every marker of every clip, the
probability that the marker sits over a vessel. Each marker's image position is
lifted onto the plane the sensor tip rests on (through the fisheye model), moved
into the world frame with the sensor pose from robot kinematics (or the
simulator's own pose for the Sim dataset), and dropped onto a top-view grid of
the phantom at ONE MILLIMETRE PER PIXEL. The map is then thresholded, compared
against a ground-truth map on the same grid, and scored pixel by pixel.

FOUR CONFIGURATIONS (train dataset -> test dataset):

    A-to-A  Sim -> Sim          one freshly simulated slide, poses from the
                                simulator (main.py::vessel_map_trajectory_main)
    A-to-B  Sim -> Silicone     the ten silicone sweeps, one shared 180 x 100 mm map
    C-to-B  Meat -> Silicone    same map, meat-trained model
    A-to-C  Sim -> Meat         one map per meat trial (ten), constant grid size

GROUND-TRUTH SOURCES:

    video      the per-marker labels of the test data reprojected exactly like
               the predictions - manual video annotation for Silicone,
               kinematics-derived labels for Meat, the simulator's own vein
               projection for Sim ("simulator" in folder names). Available for
               every configuration.
    photo      the silicone phantom's top-view photograph, segmented once and
               block-downsampled onto the same grid. Silicone only.

THE DECISION THRESHOLD IS CHOSEN, NOT ASSUMED. No fixed cut (0.5, 0.58, ...)
is used anywhere here. For each run the per-pixel map (the MAX probability of
any marker that landed on a pixel) is swept over every candidate threshold and
the one that keeps pixel-level PRECISION >= PRECISION_TARGET while maximising
RECALL is used, pooled over all the run's maps (so a configuration gets one
threshold). A predicted pixel counts as correct when it lies within
PRECISION_TOLERANCE_MM (2 mm) of a true pixel - see that constant for why the
tolerance is needed. This is a deliberate, conservative operating point: for
venipuncture assistance a false vessel (blue) sends the needle towards a
vessel that is not there, while a missed vessel (red) merely costs a re-scan.
If no threshold reaches the target (with a non-trivial number of pixels), the
run falls back to the F1-optimal threshold and says so loudly in its report
and run.json - it never silently substitutes a fixed cut. Override with
DIFFTACTILE_VESSEL_MAP_THRESHOLD (`--threshold` on the shell entrypoints).

WHAT IS WRITTEN. Everything goes to a versioned run directory,

    difftactile/output/vessel_maps/<train>-to-<test>_gt-<source>/<timestamp>/

so runs never overwrite each other. Per map: the prediction mask alone, the
ground-truth mask alone, and the confusion overlay - the latter repeated with
the ground-truth positives GROWN by an L2 (Euclidean disc) radius of 0, 1, 2 mm,
each with its own confusion counts, MCC, F1, precision, recall, accuracy and
the distribution of distances from every predicted pixel to its nearest true
pixel (median, mean, decile histogram). Text tables carry all of it
(`metrics_by_radius.md`); `run.json` is the machine-readable twin.

THE COLOUR SCHEME is Visualisation.CONFUSION_COLOURS_RGB, unchanged: green =
both say vessel, RED = truth says vessel and the map does not (a miss), BLUE =
the map says vessel and the truth does not (a false alarm), black = neither.
"""

import json
import os
import pickle
import time

import cv2
import numpy as np
from scipy import ndimage

import matplotlib
from difftactile.main.display import finish_plot, is_headless
if is_headless():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from difftactile.cnn.common import has_flat_stats
from difftactile.cnn.dataset import MyDataset
from difftactile.cnn.model_selection import resolve_model
from difftactile.cnn.visualise import Visualisation
from difftactile.main.constants import SYSTEM_PARAMS
from difftactile.main.paths import repo_path
from difftactile.sensor_model.fisheye_model_no_taichi import FisheyeModelNoTaichi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid resolution: one pixel is one millimetre, in every map. The manuscript's
# "1 px = 1 mm" convention rests on this constant.
MM_PER_PIXEL = 1.0

# The threshold rule: precision at least this, then maximise recall ...
PRECISION_TARGET = 0.9
# ... where a predicted pixel counts as correct if it lies within this L2
# distance (mm == px) of a true pixel. WHY A TOLERANCE, AND WHY 2 mm. The
# reprojected ground truth is a set of marker POINTS (2 mm apart), not a
# filled vessel region, so exact-pixel precision measures coincidence with a
# marker rather than "is this pixel over the vessel": at radius 0 the rule is
# met by 3-7 pixels on the silicone maps and by a single pixel on meat, i.e.
# by an empty map. 2 mm is the inter-marker spacing and the vessel-radius scale
# (2 mm radius phantom vessels, 3 mm metal straws), it is the largest growth
# radius the tables report (GROWTH_RADII_MM), and it is applied identically to
# every run. The 0 mm metrics are still computed and reported for every map.
PRECISION_TOLERANCE_MM = 2.0

# Ground-truth growth radii (mm == px) for the morphological-tolerance sweep.
# 0 is the plain map; 1 and 2 show how the per-pixel statistics move when the
# sparse marker-point truth is thickened. NOTE that recall/accuracy/IoU can
# FALL with radius: growing adds true pixels that nothing predicted (FN).
GROWTH_RADII_MM = [0, 1, 2]

# Distance from the camera lens to the plane the marker pixels are lifted onto,
# in REAL millimetres, per test dataset. This plane is where the sensor tip
# meets the phantom surface, so it is where the reprojected markers should lie.
#
#   silicone  19 mm lens-to-shell minus the 3 mm the sensor was pressed into the
#             phantom - the value the published silicone map has always used.
#   sim       identical: get_slide_trajectory() presses 3 mm (real) into the
#             phantom, so the contact plane sits 16 mm from the lens.
#   meat      the undeformed distance, 19 mm: the sensor is assumed to rest on
#             the meat surface with the dome in its default state (the trials
#             give no press depth to correct for), which is the assumption the
#             meat map is built on and is documented in the manuscript.
LENS_TO_SHELL_MM = 19.0
PLANE_DIST_MM = {"silicone": LENS_TO_SHELL_MM - 3.0, "sim": LENS_TO_SHELL_MM - 3.0,
                 "meat": LENS_TO_SHELL_MM}

# Silicone rig workspace, in metres, exactly as the published map defined it.
# The map is 180 x 100 px = 180 x 100 mm.
SILICONE_X_RANGE_M = (-0.425, -0.245)
SILICONE_Y_RANGE_M = (-0.054, 0.046)

# Margin (px) around the swept region for the meat and sim grids, whose extent
# is data-driven rather than a fixed workspace.
GRID_MARGIN_PX = 3

CONFIGS = {
    "A-to-A": {"train": "sim", "test": "sim", "gt_sources": ("simulator",)},
    "A-to-B": {"train": "sim", "test": "silicone", "gt_sources": ("video", "photo")},
    "C-to-B": {"train": "meat", "test": "silicone", "gt_sources": ("video", "photo")},
    "A-to-C": {"train": "sim", "test": "meat", "gt_sources": ("video",)},
}
# Where the Sim->Sim trajectory lives (written by vessel_map_trajectory_main).
SIM_MAP_TRAJECTORY_DIR = "difftactile/output/vessel_map_sim/raw_reordered_dense"
OUTPUT_ROOT = "difftactile/output/vessel_maps"

# Cheap upscale factor for the `_big` PNG twins - nearest-neighbour, so a
# 180 x 100 map is readable in a document without any interpolation blur.
BIG_SCALE = 5


# ---------------------------------------------------------------------------
# Small helpers: metrics, growth, distances
# ---------------------------------------------------------------------------

def confusion_counts(gt, pred):
    """TP, FP, FN, TN over every pixel of two boolean maps."""
    gt = np.asarray(gt, bool)
    pred = np.asarray(pred, bool)
    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))
    return tp, fp, fn, tn


def confusion_metrics(tp, fp, fn, tn):
    """MCC, F1, precision, recall, accuracy and IoU from the four counts.

    IoU is the vessel-class (foreground) intersection over union,
    TP / (TP + FP + FN) - the same quantity the marker-level tables call
    "foreground IoU", here over map pixels.

    Undefined ratios (0/0) are reported as NaN rather than 0, so a map with no
    true pixels (the meat control trial) shows "n/a" recall instead of a fake 0.
    MCC's denominator is 0 whenever a row or column of the matrix is empty; the
    conventional value there is 0.
    """
    def ratio(a, b):
        return float(a) / b if b > 0 else float("nan")
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = ratio(tp + tn, tp + fp + fn + tn)
    iou = ratio(tp, tp + fp + fn)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (float(tp) * tn - float(fp) * fn) / denom if denom > 0 else 0.0
    return {"mcc": mcc, "f1": f1, "precision": precision, "recall": recall,
            "accuracy": accuracy, "iou": iou}


def grow_l2(mask, radius):
    """Every pixel within Euclidean distance `radius` of a true pixel.

    Distance-transform based, so the structuring element is a true L2 disc
    (not the square/diamond of a binary dilation with a box/cross), and it is
    exact for non-integer radii too. Radius 0 returns the mask itself.
    """
    mask = np.asarray(mask, bool)
    if radius <= 0 or not mask.any():
        return mask.copy()
    dist = ndimage.distance_transform_edt(~mask)
    return dist <= radius


def distances_to_truth(pred, gt):
    """L2 distance (px == mm) from each predicted pixel to the nearest true pixel.

    Returns a 1-D array with one entry per predicted pixel; empty if nothing is
    predicted, all-inf if there is no true pixel at all.
    """
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    if not pred.any():
        return np.zeros(0)
    if not gt.any():
        return np.full(int(pred.sum()), np.inf)
    dist = ndimage.distance_transform_edt(~gt)
    return dist[pred]


def distance_summary(dists):
    """median / mean / deciles of a distance sample (NaN when empty)."""
    finite = dists[np.isfinite(dists)] if dists.size else dists
    if finite.size == 0:
        return {"count": int(dists.size), "median": float("nan"), "mean": float("nan"),
                "deciles": [float("nan")] * 11}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "deciles": [float(v) for v in np.percentile(finite, np.arange(0, 101, 10))],
    }


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

# A "satisfying" operating point must flag at least this many pixels; below it
# the precision estimate is a coin toss (see choose_threshold).
MIN_PREDICTED_PIXELS = 20


def choose_threshold(maxprob_list, gt_list, visited_list, precision_target=PRECISION_TARGET,
                     tolerance_mm=PRECISION_TOLERANCE_MM,
                     min_predicted_pixels=MIN_PREDICTED_PIXELS):
    """The threshold with the highest recall among those with precision >= target.

    Precision and recall are pixel-level over the pooled maps of the run (all
    visited pixels of all maps), with a predicted pixel counted as correct when
    it lies within `tolerance_mm` of a true pixel (the ground truth grown by an
    L2 disc of that radius) - see PRECISION_TOLERANCE_MM for why. Recall is
    likewise against the grown truth. Every distinct probability value is a
    candidate threshold.

    Returns a dict describing the choice. `satisfied=False` means the target
    was out of reach (or reachable only by flagging fewer than
    `min_predicted_pixels`) and the F1-optimal threshold was used instead;
    the caller must make that visible.
    """
    # Whole grid, so precision and recall here are the same quantities the
    # per-radius tables report (unvisited pixels carry maxprob -1 and are never
    # predicted, but grown truth can extend into them and does count as FN).
    gt_list = [grow_l2(gt, tolerance_mm) for gt in gt_list]
    n_pos = int(sum(int(gt.sum()) for gt in gt_list))
    p = np.concatenate([m[v] for m, v in zip(maxprob_list, visited_list)])
    g = np.concatenate([gt[v] for gt, v in zip(gt_list, visited_list)]).astype(bool)
    n_visited = int(len(p))
    order = np.argsort(-p, kind="stable")
    p, g = p[order], g[order]
    cum_tp = np.cumsum(g)
    k = np.arange(1, len(p) + 1)
    # A threshold admits every pixel with probability >= it, so pixels sharing
    # a value enter together: evaluate at the LAST index of each value group.
    last = np.r_[p[1:] != p[:-1], True]
    thr = p[last]
    prec = cum_tp[last] / k[last]
    rec = cum_tp[last] / n_pos if n_pos > 0 else np.zeros_like(prec)
    curve = {"threshold": thr.tolist(), "precision": prec.tolist(), "recall": rec.tolist(),
             "n_predicted": k[last].tolist()}

    # "Satisfied" means the target is reached with a non-trivial map: a run
    # where 0.9 precision is only reachable by flagging a handful of pixels
    # (which happens - a single very confident marker) is not a usable
    # operating point, so it counts as unreachable and takes the fallback.
    ok = (prec >= precision_target) & (k[last] >= min_predicted_pixels)
    if n_pos > 0 and ok.any():
        idx_ok = np.flatnonzero(ok)
        best_rec = rec[idx_ok].max()
        cands = idx_ok[rec[idx_ok] == best_rec]
        i = cands[np.argmax(prec[cands])]
        satisfied = True
        note = (f"precision >= {precision_target} (within {tolerance_mm:g} mm) reached; "
                f"recall maximised at threshold {thr[i]:.4f} (precision {prec[i]:.3f}, "
                f"recall {rec[i]:.3f}, {k[last][i]} predicted pixels)")
    else:
        # FALLBACK: the F1-optimal threshold at the same tolerance - the
        # balanced operating point, chosen because the highest-precision
        # threshold is degenerate (one confident pixel). Flagged everywhere.
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
        i = int(np.argmax(f1))
        satisfied = False
        note = (f"PRECISION TARGET NOT REACHED: no threshold gives precision >= "
                f"{precision_target} (within {tolerance_mm:g} mm) with at least "
                f"{min_predicted_pixels} predicted pixels on this run (max reachable "
                f"precision {prec[k[last] >= min_predicted_pixels].max() if (k[last] >= min_predicted_pixels).any() else float('nan'):.3f}); "
                f"FALLING BACK to the F1-optimal threshold {thr[i]:.4f} (precision "
                f"{prec[i]:.3f}, recall {rec[i]:.3f}, F1 {f1[i]:.3f}, {k[last][i]} predicted pixels)")
    return {
        "threshold": float(thr[i]),
        "precision_target": precision_target,
        "tolerance_mm": tolerance_mm,
        "min_predicted_pixels": min_predicted_pixels,
        "satisfied": bool(satisfied),
        "precision_at_threshold": float(prec[i]),
        "recall_at_threshold": float(rec[i]),
        "n_predicted_pixels": int(k[last][i]),
        "n_true_pixels": n_pos,
        "n_visited_pixels": n_visited,
        "note": note,
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# One map (grid + samples)
# ---------------------------------------------------------------------------

class MapData:
    """A top-view grid and the (probability, label) samples dropped onto it.

    `shape` is (rows, cols); `to_pixel(x_mm, y_mm)` maps world millimetres to
    (row, col) - the dataset-specific orientation lives in that callable.
    """

    def __init__(self, name, shape, to_pixel, description=""):
        self.name = name
        self.shape = tuple(shape)
        self.to_pixel = to_pixel
        self.description = description
        # -1 marks "no marker ever landed here"; probabilities are in [0, 1].
        self.maxprob = -np.ones(self.shape, dtype=np.float32)
        self.gt = np.zeros(self.shape, dtype=bool)
        self.visited = np.zeros(self.shape, dtype=bool)
        self.n_samples = 0
        self.n_dropped = 0

    def add(self, x_mm, y_mm, probs, labels):
        """Drop one clip's marker samples onto the grid (out-of-grid ones skipped)."""
        rows, cols = self.to_pixel(np.asarray(x_mm, float), np.asarray(y_mm, float))
        rows = np.floor(rows).astype(int)
        cols = np.floor(cols).astype(int)
        inside = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
        self.n_samples += int(inside.sum())
        self.n_dropped += int((~inside).sum())
        for r, c, p, l in zip(rows[inside], cols[inside], probs[inside], labels[inside]):
            self.visited[r, c] = True
            if p > self.maxprob[r, c]:
                self.maxprob[r, c] = p
            if l:
                self.gt[r, c] = True

    def prediction(self, threshold):
        return self.visited & (self.maxprob >= threshold)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_stats(stats_path):
    """Normalisation statistics out of a test-loader pickle (flat or by difficulty)."""
    with open(stats_path, "rb") as f:
        test_data = pickle.load(f)
    if has_flat_stats(test_data):
        return test_data["dataset_stats"], None
    all_stats = test_data["dataset_stats"]
    difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
    return all_stats[difficulty], difficulty


def load_model(spec, device):
    """Instantiate the GNN of `spec['arch']` and load `spec['checkpoint']`."""
    from difftactile.cnn.segmentation_gnn import GNN
    model = GNN(arch=spec["arch"])
    model.load_state_dict(torch.load(spec["checkpoint"], map_location=device))
    model.eval()
    return model.to(device)


def predict_clip(model, dataset, idx, device):
    """Central-frame probabilities, labels and pixel positions of one clip.

    Returns (probs (127,), labels (127,), points (127, 2) in image pixels, item)
    where `item` is the dataset's raw tuple, so callers needing the poses it
    carries do not have to fetch (and re-normalise) the clip twice.
    """
    item = dataset[idx]
    batch = item[0]
    with torch.no_grad():
        batch = batch.to(device)
        x_px, x_mask, edge_index, _, edge_attr = model.my_prepare_data(batch, 1)
        out = model(x_px, edge_index, edge_attr, batch.batch).squeeze(-1)
        out = out[x_mask]
        # Only the CENTRAL frame of the window is reported - the same mask the
        # scored metrics use (dataset.get_mask marks clip_len // 2).
        mask = batch.mask
        probs = torch.sigmoid(out[mask]).cpu().numpy().astype(np.float32)
        labels = batch.y[mask].cpu().numpy().astype(int)
        points = batch.pos[mask].cpu().numpy().astype(np.float64)
    n = SYSTEM_PARAMS.vitactip.num_markers
    return probs.reshape(n), labels.reshape(n), points.reshape(n, 2), item


def pixels_to_plane_mm(points_px, dist_mm):
    """Lift image pixels onto the plane `dist_mm` from the lens (camera frame, mm)."""
    return FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
        np.asarray(points_px, float), dist_lens_to_plane=dist_mm
    )


# ---------------------------------------------------------------------------
# Dataset-specific sample collection
# ---------------------------------------------------------------------------

def _dataset_norm(dataset, stats, difficulty):
    if difficulty is not None:
        dataset.set_difficulty_level(difficulty)
    dataset.set_stats(stats)


def collect_silicone(model, stats, difficulty, device):
    """The published silicone map: ten sweeps on one 180 x 100 mm workspace grid.

    Geometry is byte-for-byte the published one: camera x = world x, camera y =
    -world y (R = diag(1, -1, -1)), robot pose in mm converted to metres, the
    plane 16 mm from the lens, and the image laid out with column = flipped x
    bin and row = y bin (`process_img2` in the old code did the .T and flip).
    """
    dataset = MyDataset(scheme="single_dataset", sim_exp="exp",
                        data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
                        normalise_pos=False, apply_augmentations=False, name="silicone")
    _dataset_norm(dataset, stats, difficulty)
    clip_len = SYSTEM_PARAMS.gnn.clip_len
    x0, x1 = SILICONE_X_RANGE_M
    y0, y1 = SILICONE_Y_RANGE_M
    n_x = int(np.ceil((x1 - x0) * 1000 / MM_PER_PIXEL))
    n_y = int(np.ceil((y1 - y0) * 1000 / MM_PER_PIXEL))

    def to_pixel(x_mm, y_mm):
        # x_mm, y_mm are offsets from the workspace corner; column is flipped so
        # the image matches the published orientation.
        col = (n_x - 1) - np.floor(x_mm / MM_PER_PIXEL)
        row = np.floor(y_mm / MM_PER_PIXEL)
        return row, col

    m = MapData("silicone", (n_y, n_x), to_pixel,
                "silicone phantom, ten sweeps, 180 x 100 mm workspace")
    dist_mm = PLANE_DIST_MM["silicone"]
    for idx in range(len(dataset)):
        probs, labels, points, item = predict_clip(model, dataset, idx, device)
        poses = item[2].numpy() / 1000.0                    # mm -> m
        x_r, y_r, z_r = poses[clip_len // 2, :3]
        pts_cam = pixels_to_plane_mm(points, dist_mm) / 1000.0  # m, camera frame
        # camera -> world: R = diag(1, -1, -1), t = robot position (+ plane offset in z)
        x_w = pts_cam[:, 0] + x_r
        y_w = -pts_cam[:, 1] + y_r
        m.add((x_w - x0) * 1000.0, (y_w - y0) * 1000.0, probs, labels)
    return [m]


def _meat_trial_dirs():
    from difftactile.cnn.dataset import MEAT_TRAIN_TRIALS, MEAT_VALIDATION_TRIALS
    from difftactile.cnn.dataset import MEAT_CLEAN_DATA_DIR
    allowed = set(MEAT_TRAIN_TRIALS) | set(MEAT_VALIDATION_TRIALS)
    return sorted(d for d in os.listdir(MEAT_CLEAN_DATA_DIR) if d in allowed)


def collect_meat(model, stats, difficulty, device):
    """One map per meat trial, all on the same fixed grid.

    Every trial is a single straight slide of the same length along -y at x = 0
    (robot frame, mm), so one grid fits all ten: columns run along the slide
    (-y, so the sweep reads left to right), rows across it (+x). The camera
    convention is the one the meat labels were built with
    (preprocess_meat_data._world_to_camera): camera x = world dy, camera y =
    world dx. The plane is the undeformed lens-to-tip distance (19 mm).
    """
    from difftactile.cnn.dataset import meat_trial_description
    from difftactile.cnn.dataset import MEAT_CLEAN_DATA_DIR
    dataset = MyDataset(scheme="meat", sim_exp="unused", data_dir="unused",
                        normalise_pos=False, apply_augmentations=False, name="meat")
    _dataset_norm(dataset, stats, difficulty)
    clip_len = SYSTEM_PARAMS.gnn.clip_len
    dist_mm = PLANE_DIST_MM["meat"]

    # First pass: run the model, keep world-frame samples per trial.
    per_trial = {}
    poses_cache = {}
    for idx in range(len(dataset)):
        trial_dir, _, frame_ixs = dataset.meat_data[idx]
        if trial_dir not in poses_cache:
            with np.load(os.path.join(trial_dir, "frames_poses.npz")) as d:
                poses_cache[trial_dir] = d["poses"]
        pose = poses_cache[trial_dir][frame_ixs[clip_len // 2]]
        probs, labels, points, _ = predict_clip(model, dataset, idx, device)
        pts_cam = pixels_to_plane_mm(points, dist_mm)          # mm, camera frame
        x_w = pose[0] + pts_cam[:, 1]
        y_w = pose[1] + pts_cam[:, 0]
        per_trial.setdefault(trial_dir, []).append((x_w, y_w, probs, labels))

    # A grid that fits every trial, so all ten maps share one resolution/extent.
    all_x = np.concatenate([s[0] for v in per_trial.values() for s in v])
    all_y = np.concatenate([s[1] for v in per_trial.values() for s in v])
    x_lo = np.floor(all_x.min()) - GRID_MARGIN_PX
    x_hi = np.ceil(all_x.max()) + GRID_MARGIN_PX
    y_lo = np.floor(all_y.min()) - GRID_MARGIN_PX
    y_hi = np.ceil(all_y.max()) + GRID_MARGIN_PX
    n_rows = int((x_hi - x_lo) / MM_PER_PIXEL)
    n_cols = int((y_hi - y_lo) / MM_PER_PIXEL)

    def to_pixel(x_mm, y_mm):
        row = np.floor((x_mm - x_lo) / MM_PER_PIXEL)
        col = np.floor((y_hi - y_mm) / MM_PER_PIXEL)   # slide (-y) runs left -> right
        return row, col

    maps = []
    for k, trial_dir in enumerate(sorted(per_trial)):
        trial_id = os.path.basename(trial_dir)
        m = MapData(f"trial_{k + 1:02d}_{trial_id}", (n_rows, n_cols), to_pixel,
                    meat_trial_description(trial_id))
        for x_w, y_w, probs, labels in per_trial[trial_dir]:
            m.add(x_w, y_w, probs, labels)
        maps.append(m)
    return maps


def collect_sim(model, stats, difficulty, device):
    """The Sim -> Sim map from the dedicated simulated slide (with poses).

    Reprojection mirrors the real datasets, with the simulator's exact frames:
    pixel -> point on the tip plane in the camera frame E (16 mm from the lens,
    the slide's press depth) -> sensor body frame B (E is 19 mm real above the
    tip along z) -> world A through the recorded per-frame T_BA -> millimetres
    (sim length = real length x meta.distance_scaling_factor).
    Columns run along +y (the slide direction), rows along +x.
    """
    traj_dir = repo_path(SIM_MAP_TRAJECTORY_DIR)
    files = sorted(f for f in os.listdir(traj_dir) if f.endswith(".npz")) if os.path.isdir(traj_dir) else []
    if not files:
        raise FileNotFoundError(
            f"No simulated map trajectory in {traj_dir}. Run "
            "./docker/vessel_map_sim_trajectory.sh (inside the container) first."
        )
    path = os.path.join(traj_dir, files[0])
    with np.load(path) as d:
        T_BA = d["T_BA"].astype(np.float64)                     # (frames, 4, 4)
        centreline_A = d["vein_centreline_A"].astype(np.float64)
        vein_radius = float(d["vein_radius"][0])
        dist_sf = float(d["distance_scaling_factor"][0])
        n_frames = d["markers"].shape[0]
    mm_per_sim_unit = 1000.0 / dist_sf

    dataset = MyDataset(scheme="single_dataset", sim_exp="sim", data_dir=traj_dir,
                        normalise_pos=False, apply_augmentations=False, name="sim-map")
    _dataset_norm(dataset, stats, difficulty)
    dataset.eval()
    clip_len = SYSTEM_PARAMS.gnn.clip_len
    dilation = 24   # the sim clip dilation the model was trained with
    span = clip_len * dilation
    # Every start frame, not the shuffled coarse subset the training loader
    # uses: denser windows give a denser map, and the model sees the same input
    # structure (a clip_len window at dilation 24) either way.
    dataset.data_points = [(path, s, dilation) for s in range(0, n_frames - span + 1)]

    lens_to_shell_sim = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface
    plane_sim = PLANE_DIST_MM["sim"] / mm_per_sim_unit           # sim units
    T_EB = np.eye(4)
    T_EB[2, 3] = -lens_to_shell_sim                              # inverse of T_BE

    samples = []
    for idx in range(len(dataset)):
        probs, labels, points, _ = predict_clip(model, dataset, idx, device)
        start = dataset.data_points[idx][1]
        centre = start + (clip_len // 2) * dilation
        # The recorded pixel convention is already the one the fisheye inverse
        # expects: verified by reprojecting the recorded vein_polyline pixels
        # at the vein's depth through this exact chain, which reproduces
        # vein_centreline_A to numerical precision (y std 0.0).
        p_E = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(points, plane_sim)
        p_E_h = np.c_[p_E, np.ones(len(p_E))]
        p_A = (T_BA[centre] @ T_EB @ p_E_h.T).T[:, :3] * mm_per_sim_unit
        samples.append((p_A[:, 0], p_A[:, 1], probs, labels))

    all_x = np.concatenate([s[0] for s in samples])
    all_y = np.concatenate([s[1] for s in samples])
    x_lo = np.floor(all_x.min()) - GRID_MARGIN_PX
    x_hi = np.ceil(all_x.max()) + GRID_MARGIN_PX
    y_lo = np.floor(all_y.min()) - GRID_MARGIN_PX
    y_hi = np.ceil(all_y.max()) + GRID_MARGIN_PX
    n_rows = int((x_hi - x_lo) / MM_PER_PIXEL)
    n_cols = int((y_hi - y_lo) / MM_PER_PIXEL)

    def to_pixel(x_mm, y_mm):
        return np.floor((x_mm - x_lo) / MM_PER_PIXEL), np.floor((y_mm - y_lo) / MM_PER_PIXEL)

    m = MapData("sim", (n_rows, n_cols), to_pixel,
                f"simulated slide over the vein ({os.path.basename(path)})")
    for s in samples:
        m.add(*s)
    # The vein's true centreline, for reference (world geometry, no projection).
    cl_mm = centreline_A * mm_per_sim_unit
    m.reference_centreline = to_pixel(cl_mm[:, 0], cl_mm[:, 1])
    m.reference_radius_mm = vein_radius * mm_per_sim_unit
    return [m]


# ---------------------------------------------------------------------------
# Silicone photo ground truth
# ---------------------------------------------------------------------------

def photo_ground_truth(shape):
    """The silicone phantom's top-view photo segmentation on the map grid.

    Block-majority downsample of the segmented photograph
    (files.phantom_ground_truth_segmentation_mask) onto `shape`, exactly as
    the published pipeline did. Its orientation matches the silicone map's.
    """
    path = SYSTEM_PARAMS.files.phantom_ground_truth_segmentation_mask
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"photo ground truth not found: {path}")
    gt = (img > 127).astype(np.float32)
    h, w = shape
    sy = gt.shape[0] / h
    sx = gt.shape[1] / w
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            block = gt[int(y * sy):min(int((y + 1) * sy), gt.shape[0]),
                       int(x * sx):min(int((x + 1) * sx), gt.shape[1])]
            out[y, x] = block.size > 0 and block.mean() > 0.5
    return out


def swept_region(visited, radius=2):
    """The region the sensor covered: visited pixels with the gaps between
    neighbouring markers (2 mm apart) closed by an L2 disc of that radius."""
    return ndimage.binary_closing(visited, structure=_disc(radius))


def _disc(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= radius ** 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _write_mask_png(mask, path):
    """White = true, black = false; plus a `_big` nearest-neighbour twin."""
    img = (np.asarray(mask, bool) * 255).astype(np.uint8)
    cv2.imwrite(path, img)
    base, ext = os.path.splitext(path)
    cv2.imwrite(f"{base}_big{ext}", cv2.resize(img, None, fx=BIG_SCALE, fy=BIG_SCALE,
                                              interpolation=cv2.INTER_NEAREST))


def _write_confusion(gt, pred, path_png, path_pdf, title):
    """Raw confusion PNG (+ big twin) and a legended PDF of the same map."""
    bgr = Visualisation.confusion_overlay_bgr(gt, pred)
    cv2.imwrite(path_png, bgr)
    base, ext = os.path.splitext(path_png)
    cv2.imwrite(f"{base}_big{ext}", cv2.resize(bgr, None, fx=BIG_SCALE, fy=BIG_SCALE,
                                              interpolation=cv2.INTER_NEAREST))
    overlay = Visualisation.create_confusion_matrix_overlay(gt, pred)
    fig_w = 10
    fig_h = fig_w * overlay.shape[0] / overlay.shape[1] + 1.6
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(overlay, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.figlegend(handles=Visualisation.confusion_legend_handles(
        plt, positive="vessel", reference="ground truth", candidate="prediction"),
        loc="lower center", ncol=2, frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    finish_plot(plt, path_pdf, dpi=300, bbox_inches="tight")


def _write_distance_histogram(dists, path, title):
    """Decile-based histogram: ten bars, each holding 10 % of the predicted
    pixels, spanning that decile's distance range; height = density."""
    finite = dists[np.isfinite(dists)]
    plt.figure(figsize=(7, 4))
    if finite.size >= 10:
        edges = np.percentile(finite, np.arange(0, 101, 10))
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            width = max(hi - lo, 0.05)
            plt.bar(lo, 0.1 / width, width=width, align="edge", alpha=0.7,
                    edgecolor="k", linewidth=0.5)
        plt.axvline(np.median(finite), color="r", linestyle="--",
                    label=f"median {np.median(finite):.2f} mm")
        plt.axvline(np.mean(finite), color="k", linestyle=":",
                    label=f"mean {np.mean(finite):.2f} mm")
        plt.legend()
    else:
        plt.text(0.5, 0.5, f"{finite.size} predicted pixel(s) - too few for deciles",
                 ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("distance from predicted pixel to nearest true pixel (mm)")
    plt.ylabel("density (fraction per mm)")
    plt.title(title)
    plt.tight_layout()
    finish_plot(plt, path, dpi=150)


def _write_threshold_curve(choice, path):
    """Precision and recall of the pooled map versus threshold, chosen point marked."""
    c = choice["curve"]
    plt.figure(figsize=(7, 4))
    plt.plot(c["threshold"], c["precision"], label="precision")
    plt.plot(c["threshold"], c["recall"], label="recall")
    plt.axhline(choice["precision_target"], color="0.5", linestyle="--",
                label=f"precision target {choice['precision_target']}")
    plt.axvline(choice["threshold"], color="r", linestyle=":",
                label=f"chosen {choice['threshold']:.3f}")
    plt.xlabel("decision threshold")
    plt.ylabel("pixel-level value")
    plt.title(f"Map-level precision / recall vs threshold (pooled, {choice['tolerance_mm']:g} mm tolerance)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    finish_plot(plt, path, dpi=150)


# ---------------------------------------------------------------------------
# Scoring and reporting one map
# ---------------------------------------------------------------------------

def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def score_map(m, gt, pred, out_dir, title):
    """All artifacts and metrics for one map: masks, overlays per radius,
    metrics table, distance statistics. Returns the per-radius records."""
    os.makedirs(out_dir, exist_ok=True)
    _write_mask_png(pred, os.path.join(out_dir, "prediction.png"))
    _write_mask_png(gt, os.path.join(out_dir, "ground_truth.png"))
    _write_mask_png(m.visited, os.path.join(out_dir, "swept_pixels.png"))
    if getattr(m, "reference_centreline", None) is not None:
        ref = np.zeros(m.shape, bool)
        r, c = m.reference_centreline
        r = np.floor(r).astype(int)
        c = np.floor(c).astype(int)
        ok = (r >= 0) & (r < m.shape[0]) & (c >= 0) & (c < m.shape[1])
        ref[r[ok], c[ok]] = True
        _write_mask_png(ref, os.path.join(out_dir, "ground_truth_centreline_reference.png"))

    records = []
    lines = [
        f"# {title}", "",
        f"{m.description}", "",
        f"Grid {m.shape[0]} x {m.shape[1]} px at {MM_PER_PIXEL:g} mm/px; "
        f"{int(m.visited.sum())} swept pixels; {m.n_samples} marker samples "
        f"({m.n_dropped} outside the grid).", "",
        "Ground-truth positives grown by an L2 disc of radius r (mm == px); the",
        "prediction is fixed. Distances are from each predicted pixel to the nearest",
        "(grown) true pixel.", "",
        "| r (mm) | TP | FP | FN | TN | MCC | F1 | precision | recall | accuracy | IoU | "
        "L2 median | L2 mean | n pred |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in GROWTH_RADII_MM:
        gt_r = grow_l2(gt, r)
        tp, fp, fn, tn = confusion_counts(gt_r, pred)
        met = confusion_metrics(tp, fp, fn, tn)
        dists = distances_to_truth(pred, gt_r)
        dsum = distance_summary(dists)
        _write_confusion(gt_r, pred, os.path.join(out_dir, f"confusion_r{r:02d}.png"),
                         os.path.join(out_dir, f"confusion_r{r:02d}.pdf"),
                         f"{title} - ground truth grown by {r} mm")
        _write_distance_histogram(dists, os.path.join(out_dir, f"l2_distances_r{r:02d}.png"),
                                  f"{title} - r = {r} mm")
        rec = {"radius_mm": r, "tp": tp, "fp": fp, "fn": fn, "tn": tn, **met,
               "l2": dsum}
        records.append(rec)
        lines.append(
            f"| {r} | {tp} | {fp} | {fn} | {tn} | {_fmt(met['mcc'])} | {_fmt(met['f1'])} | "
            f"{_fmt(met['precision'])} | {_fmt(met['recall'])} | {_fmt(met['accuracy'], 4)} | "
            f"{_fmt(met['iou'])} | {_fmt(dsum['median'], 2)} | {_fmt(dsum['mean'], 2)} | {dsum['count']} |"
        )
    lines += ["", "L2 deciles (0 %, 10 %, ..., 100 %) at r = 0 mm: " +
              ", ".join(_fmt(v, 2) for v in records[0]["l2"]["deciles"]), ""]
    with open(os.path.join(out_dir, "metrics_by_radius.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "metrics_by_radius.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def run_dir_for(config, gt_source, legacy=False, root=None):
    """`<root>/<train>-to-<test>_gt-<source>/<timestamp>[-legacy]/`, created."""
    cfg = CONFIGS[config]
    root = root or repo_path(OUTPUT_ROOT)
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-legacy" if legacy else "")
    path = os.path.join(root, f"{cfg['train']}-to-{cfg['test']}_gt-{gt_source}", stamp)
    os.makedirs(path, exist_ok=True)
    return path


def run(config, gt_source=None, model_choice="best", threshold=None, seed=None,
        out_root=None):
    """Build, threshold and score the bird's-eye maps of one configuration.

    Returns the run directory. See the module docstring for what is written.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown configuration {config!r}; expected {sorted(CONFIGS)}")
    cfg = CONFIGS[config]
    gt_source = gt_source or cfg["gt_sources"][0]
    if gt_source not in cfg["gt_sources"]:
        raise ValueError(
            f"{config} supports ground truth from {cfg['gt_sources']}, not {gt_source!r}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = resolve_model(config, model_choice, seed=seed)
    print(f"model: {spec['description']}\n  checkpoint {spec['checkpoint']}\n  stats {spec['stats']}")
    stats, difficulty = load_stats(spec["stats"])
    model = load_model(spec, device)

    collect = {"silicone": collect_silicone, "meat": collect_meat, "sim": collect_sim}[cfg["test"]]
    t0 = time.perf_counter()
    maps = collect(model, stats, difficulty, device)
    print(f"{len(maps)} map(s) built in {time.perf_counter() - t0:.1f} s")

    # Ground truth per map. "photo" replaces the reprojected labels with the
    # segmented top-view photograph, restricted to the region the sensor swept
    # (a vessel the sensor never passed over cannot be found or missed).
    gts = []
    extras = {}
    if gt_source == "photo":
        photo = photo_ground_truth(maps[0].shape)
        for m in maps:
            gts.append(photo & swept_region(m.visited))
    else:
        gts = [m.gt for m in maps]

    # Threshold: the rule, or an explicit override.
    choice = choose_threshold([m.maxprob for m in maps], gts, [m.visited for m in maps])
    if threshold is not None:
        choice["overridden_by_user"] = float(threshold)
        choice["threshold"] = float(threshold)
        choice["note"] = (f"threshold {threshold} given by the user (rule would have picked "
                          f"{choice['note']})")
    print(f"threshold: {choice['note']}")

    out_dir = run_dir_for(config, gt_source, legacy=(model_choice == "legacy"), root=out_root)
    print(f"run directory: {out_dir}")
    _write_threshold_curve(choice, os.path.join(out_dir, "threshold_selection.png"))

    # Silicone bonus: how well the two independent ground truths agree, which
    # bounds what any model can score against either.
    if cfg["test"] == "silicone":
        video_gt = maps[0].gt
        photo = photo_ground_truth(maps[0].shape)
        cv2.imwrite(os.path.join(out_dir, "ground_truth_sources_overlay.png"),
                    Visualisation.confusion_overlay_bgr(video_gt, photo))
        inter = int(np.logical_and(video_gt, photo).sum())
        union = int(np.logical_or(video_gt, photo).sum())
        extras["ground_truth_video_vs_photo_iou"] = inter / union if union else float("nan")
        print(f"ground-truth agreement (video vs photo) IoU: {extras['ground_truth_video_vs_photo_iou']:.4f}")

    per_map = {}
    summary_lines = [
        f"# Bird's-eye vessel map run: {config} ({cfg['train']} -> {cfg['test']}), "
        f"ground truth from {gt_source}", "",
        f"Model: {spec['description']}", f"Threshold: {choice['note']}", "",
        "Per-map metrics at every growth radius (full tables in each map's "
        "metrics_by_radius.md):", "",
        "| map | r | TP | FP | FN | TN | MCC | F1 | precision | recall | accuracy | IoU | L2 mean |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, gt in zip(maps, gts):
        pred = m.prediction(choice["threshold"])
        map_dir = os.path.join(out_dir, m.name) if len(maps) > 1 else out_dir
        title = f"{config} {m.name}"
        records = score_map(m, gt, pred, map_dir, title)
        per_map[m.name] = {"description": m.description, "shape": m.shape,
                           "swept_pixels": int(m.visited.sum()), "records": records}
        for rec in records:
            if True:
                summary_lines.append(
                    f"| {m.name} | {rec['radius_mm']} | {rec['tp']} | {rec['fp']} | {rec['fn']} | "
                    f"{rec['tn']} | {_fmt(rec['mcc'])} | {_fmt(rec['f1'])} | "
                    f"{_fmt(rec['precision'])} | {_fmt(rec['recall'])} | "
                    f"{_fmt(rec['accuracy'], 4)} | {_fmt(rec['iou'])} | {_fmt(rec['l2']['mean'], 2)} |")

    # Pooled over maps (meat: the ten trials together), per radius.
    pooled = []
    for r in GROWTH_RADII_MM:
        tp = fp = fn = tn = 0
        d_all = []
        for m, gt in zip(maps, gts):
            pred = m.prediction(choice["threshold"])
            gt_r = grow_l2(gt, r)
            a, b, c, d = confusion_counts(gt_r, pred)
            tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
            d_all.append(distances_to_truth(pred, gt_r))
        d_all = np.concatenate(d_all) if d_all else np.zeros(0)
        pooled.append({"radius_mm": r, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       **confusion_metrics(tp, fp, fn, tn), "l2": distance_summary(d_all)})
    if len(maps) > 1:
        summary_lines += ["", "Pooled over all maps:", "",
                          "| r | TP | FP | FN | TN | MCC | F1 | precision | recall | accuracy | IoU | L2 median | L2 mean |",
                          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rec in pooled:
            summary_lines.append(
                f"| {rec['radius_mm']} | {rec['tp']} | {rec['fp']} | {rec['fn']} | {rec['tn']} | "
                f"{_fmt(rec['mcc'])} | {_fmt(rec['f1'])} | {_fmt(rec['precision'])} | "
                f"{_fmt(rec['recall'])} | {_fmt(rec['accuracy'], 4)} | {_fmt(rec['iou'])} | "
                f"{_fmt(rec['l2']['median'], 2)} | {_fmt(rec['l2']['mean'], 2)} |")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    run_json = {
        "config": config, "train": cfg["train"], "test": cfg["test"],
        "ground_truth_source": gt_source,
        "model": {k: v for k, v in spec.items()},
        "clip_len": SYSTEM_PARAMS.gnn.clip_len,
        "mm_per_pixel": MM_PER_PIXEL,
        "plane_distance_mm": PLANE_DIST_MM[cfg["test"]],
        "threshold": {k: v for k, v in choice.items() if k != "curve"},
        "growth_radii_mm": GROWTH_RADII_MM,
        "maps": per_map, "pooled": pooled, **extras,
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_json, f, indent=2, default=float)
    print(f"report: {os.path.join(out_dir, 'report.md')}")
    return out_dir


def main():
    """Entrypoint (scripts/script_vessel_map.py). Everything comes from env vars,
    set by docker/vessel_map.sh - there is no argparse anywhere in the project."""
    config = os.environ.get("DIFFTACTILE_MAP_CONFIG", "A-to-B")
    gt_source = os.environ.get("DIFFTACTILE_MAP_GT") or None
    model_choice = os.environ.get("DIFFTACTILE_MAP_MODEL", "best")
    seed_env = os.environ.get("DIFFTACTILE_MAP_SEED")
    seed = int(seed_env) if seed_env else None
    thr_env = os.environ.get("DIFFTACTILE_VESSEL_MAP_THRESHOLD")
    threshold = float(thr_env) if thr_env else None
    run(config, gt_source=gt_source, model_choice=model_choice, threshold=threshold, seed=seed)
