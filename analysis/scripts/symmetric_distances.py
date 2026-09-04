"""Symmetric nearest-neighbour distances for the poster's segmentation table.

The manuscript's Table 4 reports, for the top-view maps only, the ONE-directional
mean distance d from each predicted vessel pixel to the nearest true vessel
pixel. The poster table replaces it with a SYMMETRIC formulation, computed in
both spaces with the same rule:

    for every scored element (a marker of a central video frame, or a pixel of a
    top-view map) the distance is measured to the nearest element whose ground-
    truth label equals the PREDICTED label of that element:
        predicted vessel      -> nearest ground-truth vessel element
        predicted background  -> nearest ground-truth background element
    (so TP and TN elements contribute 0, FP and FN elements contribute > 0);
    all distances of all frames / maps of a model are pooled and averaged ONCE
    (mean, never a mean of per-frame means), and infinite distances (a frame or
    map with no ground-truth element of that label) are dropped, as in
    vessel_map.py::distance_summary.

Video-frame space: markers at 1920x1080 pixel coordinates, converted to mm with
the sensor's inter-marker spacing, 55 px = 2 mm (main.py::PX_TO_MM, and the
manuscript's domain-adaptation section). Predictions at threshold 0.5, the
threshold of Table 4. Inputs: results/frame_space_predictions_<config>.npz.

Top-view map space: prediction.png / ground_truth.png of the vessel-map runs the
manuscript's Table 4 and Fig. 7 were made from (1 px = 1 mm); the one-directional
mean is recomputed and checked against Table 4 (1.05 / 1.21 / 5.49 / 1.31 mm).

For transparency the script also reports (a) the manuscript's one-directional
value, (b) the symmetric value restricted to the mismatched elements (FP + FN
only), and (c) the standard average symmetric surface distance (predicted
vessel -> true vessel and true vessel -> predicted vessel), in
results/symmetric_distances.{md,json}. The poster uses the first definition.
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CONFIGS as RUNS, RESULTS, vessel_map_run  # noqa: E402
NAMES = {"A-to-A": "Sim→Sim", "A-to-B": "Sim→Silicone", "A-to-C": "Sim→Meat", "C-to-B": "Meat→Silicone"}
TABLE4_D = {"A-to-A": 1.05, "A-to-B": 1.21, "A-to-C": 5.49, "C-to-B": 1.31}
TABLE4_FG_IOU_FRAME = {"A-to-A": 0.19, "A-to-B": 0.24, "A-to-C": 0.17, "C-to-B": 0.16}
TABLE4_FG_IOU_MAP = {"A-to-A": 0.42, "A-to-B": 0.28, "A-to-C": 0.21, "C-to-B": 0.31}
PX_TO_MM = 2.0 / 55.0
THRESHOLD = 0.5


def finite_mean(values):
    v = np.concatenate(values) if values else np.zeros(0)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan"), int(v.size)


def nn_dist(src, dst):
    """Distance from every point of `src` to the nearest point of `dst` (inf if none)."""
    if len(src) == 0:
        return np.zeros(0)
    if len(dst) == 0:
        return np.full(len(src), np.inf)
    return cKDTree(dst).query(src)[0]


# ----------------------------------------------------------------------------- frame space
def frame_space(config):
    d = np.load(os.path.join(RESULTS, f"frame_space_predictions_{config}.npz"), allow_pickle=True)
    probs, labels, pos = d["probs"], d["labels"].astype(bool), d["pos"].astype(float)
    pred = probs >= THRESHOLD
    sym, mis, one, assd = [], [], [], []
    for p, l, xy in zip(pred, labels, pos):
        d_pos = nn_dist(xy[p], xy[l])            # predicted vessel -> true vessel
        d_neg = nn_dist(xy[~p], xy[~l])          # predicted background -> true background
        sym += [d_pos, d_neg]
        mis += [d_pos[~l[p]], d_neg[l[~p]]]      # FP part of d_pos, FN part of d_neg
        one.append(d_pos)
        assd += [d_pos, nn_dist(xy[l], xy[p])]   # + true vessel -> predicted vessel
    tp, fp = int((pred & labels).sum()), int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    out = {"fg_iou": tp / (tp + fp + fn)}
    for key, vals in (("symmetric_all", sym), ("symmetric_mismatched", mis),
                      ("one_directional", one), ("assd", assd)):
        m, n = finite_mean(vals)
        out[key + "_px"], out[key + "_mm"], out[key + "_n"] = m, m * PX_TO_MM, n
    return out


# ----------------------------------------------------------------------------- map space
def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def map_pairs(config):
    run = str(vessel_map_run(config))
    subs = sorted(s for s in os.listdir(run) if os.path.isdir(os.path.join(run, s)))
    dirs = [os.path.join(run, s) for s in subs] or [run]
    return [(load_mask(os.path.join(m, "prediction.png")), load_mask(os.path.join(m, "ground_truth.png")),
             load_mask(os.path.join(m, "swept_pixels.png"))) for m in dirs]


def map_space(config):
    sym, sym_swept, mis, one, assd = [], [], [], [], []
    tp = fp = fn = 0
    for pred, gt, swept in map_pairs(config):
        to_gt_pos = ndimage.distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
        to_gt_neg = ndimage.distance_transform_edt(gt) if (~gt).any() else np.full(gt.shape, np.inf)
        to_pred_pos = ndimage.distance_transform_edt(~pred) if pred.any() else np.full(gt.shape, np.inf)
        d_pos, d_neg = to_gt_pos[pred], to_gt_neg[~pred]
        sym += [d_pos, d_neg]
        sym_swept += [to_gt_pos[pred & swept], to_gt_neg[~pred & swept]]
        mis += [to_gt_pos[pred & ~gt], to_gt_neg[~pred & gt]]
        one.append(d_pos)
        assd += [d_pos, to_pred_pos[gt]]
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
    out = {"fg_iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan")}
    for key, vals in (("symmetric_all", sym), ("symmetric_swept", sym_swept), ("symmetric_mismatched", mis),
                      ("one_directional", one), ("assd", assd)):
        m, n = finite_mean(vals)
        out[key + "_mm"], out[key + "_n"] = m, n
    return out


def main():
    res = {}
    for c in RUNS:
        res[c] = {"frame": frame_space(c), "map": map_space(c)}
        f, m = res[c]["frame"], res[c]["map"]
        print(f"{NAMES[c]:14s} frame: FG IoU {f['fg_iou']:.3f} (tab {TABLE4_FG_IOU_FRAME[c]}) "
              f"sym {f['symmetric_all_mm']:.3f} mm | map: FG IoU {m['fg_iou']:.3f} (tab {TABLE4_FG_IOU_MAP[c]}) "
              f"one-dir {m['one_directional_mm']:.2f} (tab {TABLE4_D[c]}) sym {m['symmetric_all_mm']:.3f} mm")
        assert abs(m["one_directional_mm"] - TABLE4_D[c]) < 0.006, (c, m["one_directional_mm"])
        assert abs(f["fg_iou"] - TABLE4_FG_IOU_FRAME[c]) < 0.006 and abs(m["fg_iou"] - TABLE4_FG_IOU_MAP[c]) < 0.006

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "symmetric_distances.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    lines = [
        "# Symmetric nearest-neighbour distances (poster table)", "",
        "Definition used on the poster (`symmetric_all`): every scored element (marker of a central",
        "video frame / pixel of a top-view map) measures the distance to the nearest ground-truth",
        "element carrying the label the model PREDICTED for it (predicted vessel -> nearest true",
        "vessel; predicted background -> nearest true background). TP and TN contribute 0, FP and FN",
        "contribute > 0. All distances of a model are pooled and averaged once (mean); infinite",
        "distances (no ground-truth element of that label in the frame/map) are dropped.",
        "Frame space: threshold 0.5, markers in 1920x1080 px, 55 px = 2 mm. Map space: 1 px = 1 mm,",
        "the manuscript's runs and operating points; the recomputed one-directional means match",
        "Table 4 exactly.", "",
        "| Model | space | FG IoU | symmetric, all elements (mm) | symmetric, FP+FN only (mm) | one-directional, Table 4 (mm) | ASSD (mm) | n elements |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in RUNS:
        f, m = res[c]["frame"], res[c]["map"]
        lines.append(f"| {NAMES[c]} | video frame (marker) | {f['fg_iou']:.2f} | {f['symmetric_all_mm']:.2f} | "
                     f"{f['symmetric_mismatched_mm']:.2f} | {f['one_directional_mm']:.2f} | {f['assd_mm']:.2f} | {f['symmetric_all_n']} |")
        lines.append(f"| {NAMES[c]} | top-view map (pixel) | {m['fg_iou']:.2f} | {m['symmetric_all_mm']:.2f} | "
                     f"{m['symmetric_mismatched_mm']:.2f} | {m['one_directional_mm']:.2f} | {m['assd_mm']:.2f} | {m['symmetric_all_n']} |")
    lines += ["", "Map space, symmetric over swept pixels only (pixels the sensor passed over): " +
              ", ".join(f"{NAMES[c]} {res[c]['map']['symmetric_swept_mm']:.2f} mm" for c in RUNS), ""]
    with open(os.path.join(RESULTS, "symmetric_distances.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"written {RESULTS}/symmetric_distances.{{md,json}}")


if __name__ == "__main__":
    main()
