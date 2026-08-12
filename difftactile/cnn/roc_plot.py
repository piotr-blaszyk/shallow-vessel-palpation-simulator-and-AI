"""Shared ROC-curve rendering, so every AUROC figure in the project looks alike.

Two entrypoints draw ROC curves - `cnn/auroc_all_scenarios.py` (all six
scenarios) and `cnn/segmentation_gnn.evaluate_and_plot_roc()` (the A-to-B
`--eval` path). Both call `plot_roc()` here rather than carrying their own copy
of the styling.

Design decisions worth knowing:

* **No descriptive title.** Long scenario descriptions overflowed the figure
  width and were clipped. A figure is identified by its filename; only the
  AUROC (2 d.p., rounded) is printed above the axes.
* **The curve is colour-coded by decision threshold** using a perceptually
  uniform colourmap pinned to [0, 1], with a colourbar. Because the scale is
  fixed rather than per-figure, a given colour means the same threshold in
  every panel, which makes the three models directly comparable. A model whose
  probabilities are squeezed into a narrow band shows as a curve of nearly
  constant colour - the calibration story made visible instead of hidden.
* **The marked thresholds are identical across panels.** They bunch up where a
  model's probabilities bunch up; that is the honest depiction, so no per-figure
  threshold set is chosen to make the spacing look even.
"""

import os

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from difftactile.main.display import finish_plot

# Decision thresholds annotated as discrete markers on every ROC curve.
MARKED_THRESHOLDS = np.array([0.40, 0.50, 0.60])

# Colourmap encoding the decision threshold along the ROC curve, always
# spanning the full [0, 1] range so colours are comparable between figures.
THRESHOLD_CMAP = "viridis"
THRESHOLD_NORM = Normalize(vmin=0.0, vmax=1.0)

_FONTSIZE = 20


def operating_points(all_probs, all_labels, thresholds=MARKED_THRESHOLDS):
    """(fpr, tpr) of each threshold, as the confusion matrix at that cut.

    Returns two lists, parallel to `thresholds`.
    """
    tpr_list, fpr_list = [], []
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return fpr_list, tpr_list


def plot_roc(plt, fpr, tpr, all_probs, all_labels, auc, out_path,
             thresholds_roc=None):
    """Render one ROC figure to `out_path` as a PDF.

    `plt` is passed in by the caller because each module selects the matplotlib
    backend itself (Agg when headless) before importing pyplot.

    `thresholds_roc` is sklearn's per-vertex threshold array from `roc_curve()`;
    when omitted the curve is drawn in a single colour rather than inventing a
    threshold mapping.
    """
    fpr_list, tpr_list = operating_points(all_probs, all_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Chance line first, so the ROC curve is drawn over it.
    ax.plot([0, 1], [0, 1], "k-", alpha=0.5, linewidth=6.0, zorder=1)

    if thresholds_roc is None:
        ax.plot(fpr, tpr, alpha=0.8, linewidth=6.0, zorder=2)
    else:
        # sklearn prepends an artificial +inf threshold for the (0, 0) point;
        # clip so the segment colours stay finite.
        thr_v = np.clip(np.asarray(thresholds_roc, dtype=float), 0.0, 1.0)
        points = np.column_stack([fpr, tpr]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Colour each segment by the mean threshold of its two endpoints.
        seg_thr = 0.5 * (thr_v[:-1] + thr_v[1:])
        lc = LineCollection(segments, cmap=THRESHOLD_CMAP, norm=THRESHOLD_NORM,
                            linewidth=6.0, alpha=0.9, zorder=2,
                            capstyle="round", joinstyle="round")
        lc.set_array(seg_thr)
        ax.add_collection(lc)

    # The marked thresholds, filled from the same colourmap so they read as
    # samples of the curve rather than a separate series.
    ax.scatter(fpr_list, tpr_list, c=MARKED_THRESHOLDS, cmap=THRESHOLD_CMAP,
               norm=THRESHOLD_NORM, s=200, edgecolors="black", linewidths=2.0,
               zorder=3)
    for thr, x, y in zip(MARKED_THRESHOLDS, fpr_list, tpr_list):
        ax.text(x, y, f"{thr:.2f}", fontsize=_FONTSIZE, ha="left", va="bottom",
                fontweight="bold", zorder=4)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", which="major", labelsize=_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("False Positive Rate", fontsize=_FONTSIZE, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=_FONTSIZE, fontweight="bold")
    # AUROC only - no descriptive title, and rounded (not truncated) to 2 d.p.
    ax.set_title(f"AUROC = {auc:.2f}", fontsize=_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)

    cbar = fig.colorbar(
        ScalarMappable(norm=THRESHOLD_NORM, cmap=THRESHOLD_CMAP), ax=ax,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    cbar.set_label("Decision threshold", fontsize=_FONTSIZE - 2, fontweight="bold")
    cbar.ax.tick_params(labelsize=_FONTSIZE - 4)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    cbar.outline.set_linewidth(3.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    finish_plot(plt, out_path, format="pdf", dpi=300)
