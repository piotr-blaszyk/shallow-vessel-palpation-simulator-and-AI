"""Video-frame-space metrics of the best-of-five models, pooled once over all frames.

WHAT THIS REPORTS. For each configuration, the per-marker predictions of its
best-of-five seed instance (cnn/model_selection.py) on that configuration's
test set - every central-frame marker of every clip of every trial, POOLED
into one set - are scored exactly once at DECISION_THRESHOLD (0.5): TP, FP,
FN, TN, MCC, F1, precision, recall, foreground and background IoU, plus the
threshold-free average precision. Nothing is averaged over trials, clips or
seeds: this is the single-model, single-aggregation twin of the pooled
top-view-map statistics in vessel_map.py, so the two can sit in one table.

The per-marker probabilities and labels are read from the `scores_<config>.npz`
that every sweep run saved beside its checkpoint (written by
segmentation_gnn.score_ranking_metrics), so no inference is repeated.

Entrypoint: python -m difftactile.scripts.script_frame_space_metrics
Output: FRAME_SPACE_METRICS.md at the repository root, plus the same in JSON
beside the published sweep (frame_space_metrics.json).
"""

import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

from difftactile.cnn.curve_plots import DECISION_THRESHOLD
from difftactile.cnn.model_selection import best_model, published_sweep_dir
from difftactile.main.paths import repo_path

CONFIGS = ("A-to-A", "A-to-B", "A-to-C", "C-to-B")
MANUSCRIPT_NAMES = {"A-to-A": "Sim→Sim", "A-to-B": "Sim→Silicone",
                    "A-to-C": "Sim→Meat", "C-to-B": "Meat→Silicone"}


def pooled_metrics(probs, labels, threshold=DECISION_THRESHOLD):
    """All metrics from one pooled (probs, labels) pair. NaN where undefined."""
    probs = np.asarray(probs, float).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    pred = probs >= threshold
    tp = int(np.sum(pred & labels)); fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels)); tn = int(np.sum(~pred & ~labels))
    r = lambda a, b: float(a) / b if b > 0 else float("nan")
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "threshold": threshold, "n": int(labels.size), "n_positive": int(labels.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mcc": (float(tp) * tn - float(fp) * fn) / denom if denom > 0 else 0.0,
        "f1": r(2 * tp, 2 * tp + fp + fn),
        "precision": r(tp, tp + fp), "recall": r(tp, tp + fn),
        "iou_fg": r(tp, tp + fp + fn), "iou_bg": r(tn, tn + fp + fn),
        "ap": float(average_precision_score(labels, probs)) if labels.any() else float("nan"),
        "chance": r(labels.sum(), labels.size),
    }


def scores_for(config):
    """(probs, labels, spec) of `config`'s best-of-five instance."""
    spec = best_model(config)
    directory = os.path.dirname(spec["checkpoint"])
    path = os.path.join(directory, f"scores_{config}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - the sweep run did not save its scores")
    with np.load(path) as d:
        return d["probs"], d["labels"], spec


def _fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    rows = {}
    for config in CONFIGS:
        probs, labels, spec = scores_for(config)
        rows[config] = {"model": spec["description"], **pooled_metrics(probs, labels)}
        m = rows[config]
        print(f"{config}: n={m['n']} pos={m['n_positive']} TP {m['tp']} FP {m['fp']} FN {m['fn']} "
              f"TN {m['tn']} MCC {m['mcc']:.3f} F1 {m['f1']:.3f} P {m['precision']:.3f} "
              f"R {m['recall']:.3f} FG IoU {m['iou_fg']:.3f} BG IoU {m['iou_bg']:.3f} AP {m['ap']:.3f}")

    lines = [
        "# Video-frame-space metrics (best-of-five instance per model, pooled once)", "",
        "Per-marker predictions of every central frame of every clip of every trial, pooled",
        f"into one set and scored once at threshold {DECISION_THRESHOLD}; AP is threshold-free.",
        "No averaging over trials, clips or seeds. The instance is the best-of-five by AP",
        "(cnn/model_selection.py) - the same one the top-view maps use.", "",
        "| Model | n | positives | TP | FP | FN | TN | MCC | F1 | Prec. | Rec. | FG IoU | BG IoU | AP | chance |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in CONFIGS:
        m = rows[c]
        lines.append(f"| {MANUSCRIPT_NAMES[c]} ({c}) | {m['n']} | {m['n_positive']} | {m['tp']} | {m['fp']} | "
                     f"{m['fn']} | {m['tn']} | {_fmt(m['mcc'])} | {_fmt(m['f1'])} | {_fmt(m['precision'])} | "
                     f"{_fmt(m['recall'])} | {_fmt(m['iou_fg'])} | {_fmt(m['iou_bg'])} | {_fmt(m['ap'])} | "
                     f"{_fmt(m['chance'], 3)} |")
    lines += ["", "Instances:"] + [f"- {rows[c]['model']}" for c in CONFIGS] + [""]
    md = repo_path("FRAME_SPACE_METRICS.md")
    with open(md, "w") as f:
        f.write("\n".join(lines))
    js = os.path.join(published_sweep_dir(), "frame_space_metrics.json")
    with open(js, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"written: {md}\n         {js}")
    return rows
