"""Per-central-frame GNN predictions of the four best-of-five models, with marker pixels.

WHY THIS EXISTS. The published sweep stores, per model, only the pooled
(probability, label) pairs of every central-frame marker (`scores_<config>.npz`).
Two poster tasks need more than that:

  1. the SYMMETRIC marker-distance metric in video-frame space needs the pixel
     position of every scored marker;
  2. the "representative video frame" selection for the workflow diagram needs
     the predictions grouped by trial and by video frame.

So this script re-runs inference of each configuration's best-of-five instance
on that configuration's test set, built exactly as the prediction viewer builds
it (`cnn/visualise.py::Visualisation._build_scenario_dataset`, sliding windows,
central frame only), and caches per central frame:

  probs  (n_frames, 127)  sigmoid output of the central frame's markers
  labels (n_frames, 127)  ground-truth vessel label (1 = vessel present)
  pos    (n_frames, 127, 2) marker pixel positions at 1920x1080 (x, y)
  trial  (n_frames,)      trial id (meat trial directory / npz stem)
  frame  (n_frames,)      video frame index of the central frame in its trial

into  analysis/results/frame_space_predictions_<config>.npz.

CHECK. After inference the pooled TP/FP/FN/TN at threshold 0.5 is compared with
the manuscript's Table 4 (FRAME_SPACE_METRICS.md of the code repository); the
script raises if they differ, so the cached predictions are guaranteed to be
the ones the manuscript reports.

Run inside the container (torch + torch_geometric; no Taichi needed):
  ./docker/reproduce_analysis.sh            # this and every downstream script, in order
  python analysis/scripts/frame_space_predictions.py    # just this step
"""

import os
import pickle
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO as _REPO, RESULTS  # noqa: E402

REPO = str(_REPO)
OUT_DIR = str(RESULTS)

sys.path.insert(0, REPO)
os.chdir(REPO)   # the simulator's config resolves a few paths against the working directory

from difftactile.cnn.visualise import Visualisation, _segmentation_gnn, VIEWER_SCENARIOS  # noqa: E402
from difftactile.main.constants import SYSTEM_PARAMS  # noqa: E402

CONFIGS = ("A-to-A", "A-to-B", "A-to-C", "C-to-B")
# Table 4 of the manuscript (upper half) = FRAME_SPACE_METRICS.md, threshold 0.5.
EXPECTED = {
    "A-to-A": (3569, 15169, 38, 66949),
    "A-to-B": (909, 2070, 818, 11443),
    "A-to-C": (1493, 7134, 317, 18869),
    "C-to-B": (1600, 8448, 127, 5065),
}


def clip_identity(dataset, clip_ix):
    """(trial id, central video-frame index, raw marker pixels of that frame)."""
    clip_len = dataset.clip_len
    centre = clip_len // 2
    meat = getattr(dataset, "scheme", None) == "meat"
    if meat:
        trial_dir, _dilation, frames = dataset.meat_data[clip_ix]
        frame = int(frames[centre])
        markers = np.load(os.path.join(trial_dir, "marker_positions.npz"))["marker_positions"]
        return os.path.basename(trial_dir), frame, markers[frame].astype(float)
    file_path, start_ix, dilation = dataset.data_points[clip_ix]
    frame = int(start_ix) + centre * int(dilation)
    markers = np.load(file_path)["markers"]
    return os.path.splitext(os.path.basename(file_path))[0], frame, markers[frame].astype(float)


def run(config):
    vis = Visualisation(scenario=config, weights="best", frames="central")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _segmentation_gnn(VIEWER_SCENARIOS[config]["arch"])
    model.load_state_dict(torch.load(vis.model_path, map_location=device))
    model.eval().to(device)
    with open(vis.test_loader, "rb") as f:
        all_stats = pickle.load(f)["dataset_stats"]
    dataset = vis._build_scenario_dataset(all_stats)
    # The pickled simulated test set records the paths of the container it was
    # built in (/workspace/...); point them at this checkout.
    if getattr(dataset, "data_points", None):
        dataset.data_points = [
            (str(fp).replace("/workspace/shallow-vessel-palpation-simulator-and-AI", REPO), *rest)
            if len(dp) == 3 else dp
            for dp in dataset.data_points for (fp, *rest) in [dp]
        ]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    n_nodes = SYSTEM_PARAMS.vitactip.num_markers
    probs_all, labels_all, pos_all, trial_all, frame_all = [], [], [], [], []
    with torch.no_grad():
        for clip_ix, (batch, _labels_images, _poses, _meta, _frame_ix) in enumerate(loader):
            batch = batch.to(device)
            x, x_mask, edge_index, _, edge_attr = model.my_prepare_data(batch, batch.num_graphs)
            out = model(x, edge_index, edge_attr, batch.batch).squeeze(-1)[x_mask]
            mask = batch.mask
            probs = torch.sigmoid(out[mask]).cpu().numpy()
            labels = batch.y[mask].cpu().numpy()
            assert probs.shape == (n_nodes,), probs.shape
            trial, frame, pixels = clip_identity(dataset, clip_ix)
            probs_all.append(probs.astype(np.float32))
            labels_all.append(labels.astype(np.int8))
            pos_all.append(pixels.astype(np.float32))
            trial_all.append(trial)
            frame_all.append(frame)

    probs = np.stack(probs_all)
    labels = np.stack(labels_all)
    pos = np.stack(pos_all)
    pred = probs >= 0.5
    lab = labels.astype(bool)
    tp, fp = int((pred & lab).sum()), int((pred & ~lab).sum())
    fn, tn = int((~pred & lab).sum()), int((~pred & ~lab).sum())
    print(f"{config}: {len(probs)} central frames; TP {tp} FP {fp} FN {fn} TN {tn}; expected {EXPECTED[config]}")
    if (tp, fp, fn, tn) != EXPECTED[config]:
        raise RuntimeError(f"{config}: pooled confusion does not match the manuscript's Table 4")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, f"frame_space_predictions_{config}.npz"),
        probs=probs, labels=labels, pos=pos,
        trial=np.array(trial_all), frame=np.array(frame_all, dtype=int),
        model=vis.weights_label,
    )


if __name__ == "__main__":
    for c in (sys.argv[1:] or CONFIGS):
        run(c)
