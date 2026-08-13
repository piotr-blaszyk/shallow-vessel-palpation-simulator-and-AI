"""Shared ROC and precision-recall rendering, so every figure here looks alike.

Two entrypoints draw these curves - `cnn/auroc_all_scenarios.py` (all six
scenarios) and `cnn/segmentation_gnn.evaluate_and_plot_roc()` (the A-to-B
`--eval` path). Both call `plot_roc()` / `plot_pr()` here rather than carrying
their own copy of the styling.

WHY BOTH CURVES. Both are threshold-free and ranking-based: they read only the
*ordering* of the predicted probabilities, never their absolute scale, so they
measure what the model learned rather than how well it happens to be calibrated
for a given domain. That matters here because this is a sim-to-real transfer
project, where the first thing to break when moving between domains is the
output scale, not the ranking.

They differ in what they are blind to, which is why neither replaces the other:

* **AUROC** normalises false positives by the total negative count. This dataset
  is heavily imbalanced (~5% positive), so a large absolute number of false
  alarms barely moves it - AUROC can look reassuring where precision is poor.
  Its baseline is always 0.5, which makes it comparable with the literature.
* **Average precision** (area under the precision-recall curve) ignores true
  negatives entirely, so it cannot be flattered by the vast negative majority.
  Its baseline is the positive rate itself, so on a ~5% positive set an AP of
  0.20 is a genuine 4x lift over chance. This is the more honest summary of a
  needle-in-a-haystack problem, and the harder number to hide behind.

Report both: AUROC for comparability, AP because it cannot flatter.

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
* **The PR figure draws its own chance baseline** at the positive rate, as a
  dashed horizontal line, and prints that rate. Unlike ROC's fixed diagonal this
  baseline moves with the dataset, so without it drawn a PR curve cannot be read
  at all - the same curve is excellent on a 1% positive set and useless on a 50%
  one. Both figures share the threshold colourmap and marker styling so they can
  be placed side by side.
"""

import os

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from difftactile.main.display import finish_plot

# =============================================================================
# The decision thresholds
# =============================================================================
#
# THE HEADLINE METRICS USE NO THRESHOLD AT ALL. AUROC and AP are ranking-based
# (see the module docstring), which is why they are what the paper reports. The
# constants below exist only for the places that must commit to a hard yes/no:
# the IoU logged during training, the viewer's hard-prediction panel, and the
# bird's-eye vessel map. Do not read them as tuned hyperparameters.

# Threshold used when a probability must become a label.
#
# 0.5 is the conventional cut, and it is deliberately NOT tuned. On a heavily
# imbalanced problem trained with focal loss it is an arbitrary choice rather
# than a principled one - but every alternative worth having (Youden's J, the
# F1-optimal point) must be *fitted*, and fitting it on the test set is the one
# thing that would actually invalidate the reported numbers. Picking it on
# training-domain validation data and freezing it is the defensible route; until
# that is done, the honest position is an untuned conventional default plus the
# threshold-free metrics beside it.
#
# Consequence worth knowing: this is a display and monitoring choice. Changing it
# moves every IoU in the logs and repaints the vessel map, but moves NO reported
# metric, because AUROC and AP never see it.
DECISION_THRESHOLD = 0.5

# The vessel map and the viewer's hard-prediction panel historically used 0.58
# rather than 0.5, hardcoded in two places that had drifted apart from the four
# using 0.5. It is kept as its own named constant rather than silently
# normalised to DECISION_THRESHOLD, because doing so would change the published
# vessel-map figure - a real change to a paper artifact, not a refactor.
#
# It is an empirical pick: it looked best on the silicone map. That is exactly
# the "tune until the picture is nice" move that does not generalise, so it is
# confined to the qualitative figures and touches nothing that is scored. Anyone
# reproducing the map gets the published one; anyone wanting the conventional
# cut can set DIFFTACTILE_MAP_THRESHOLD=0.5.
MAP_DECISION_THRESHOLD = float(os.environ.get("DIFFTACTILE_MAP_THRESHOLD", 0.58))

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


def pr_operating_points(all_probs, all_labels, thresholds=MARKED_THRESHOLDS):
    """(recall, precision) of each threshold, as the confusion matrix at that cut.

    The precision-recall twin of `operating_points()`. Returns two lists,
    parallel to `thresholds`. Precision is defined as 0 when a threshold predicts
    nothing positive, which is the convention sklearn uses.
    """
    recall_list, precision_list = [], []
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        recall_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        precision_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return recall_list, precision_list


def plot_pr(plt, precision, recall, all_probs, all_labels, ap, out_path,
            thresholds_pr=None):
    """Render one precision-recall figure to `out_path` as a PDF.

    Deliberately styled as the ROC figure's twin - same threshold colourmap,
    same marked thresholds, same line weights - so the pair can sit side by side
    in a manuscript and be read as two views of one result.

    The one structural difference is the **chance baseline**. ROC's is the fixed
    diagonal; PR's is a horizontal line at the positive rate, which moves with
    the dataset. It is drawn and labelled because a PR curve is meaningless
    without it: the same curve is excellent on a 1% positive set and worthless on
    a 50% one.

    `thresholds_pr` is sklearn's per-vertex threshold array from
    `precision_recall_curve()`; when omitted the curve is drawn in a single
    colour rather than inventing a threshold mapping. Note sklearn returns it
    one element SHORTER than precision/recall (the final point, recall 0 and
    precision 1, corresponds to no threshold), which this handles.
    """
    recall_list, precision_list = pr_operating_points(all_probs, all_labels)
    baseline = float(np.mean(all_labels))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Chance baseline first, so the curve is drawn over it. A classifier that
    # ranks at random scores precision == the positive rate at every recall.
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.5, linewidth=4.0,
               zorder=1)
    ax.text(0.98, baseline, f"chance = {baseline:.3f}", fontsize=_FONTSIZE - 6,
            ha="right", va="bottom", fontweight="bold", zorder=4)

    if thresholds_pr is None:
        ax.plot(recall, precision, alpha=0.8, linewidth=6.0, zorder=2)
    else:
        # sklearn's thresholds array is one shorter than precision/recall, so
        # colour only the segments those thresholds actually span.
        thr_v = np.clip(np.asarray(thresholds_pr, dtype=float), 0.0, 1.0)
        n = len(thr_v)
        points = np.column_stack([recall[:n + 1], precision[:n + 1]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=THRESHOLD_CMAP, norm=THRESHOLD_NORM,
                            linewidth=6.0, alpha=0.9, zorder=2,
                            capstyle="round", joinstyle="round")
        lc.set_array(thr_v)
        ax.add_collection(lc)

    ax.scatter(recall_list, precision_list, c=MARKED_THRESHOLDS,
               cmap=THRESHOLD_CMAP, norm=THRESHOLD_NORM, s=200,
               edgecolors="black", linewidths=2.0, zorder=3)
    for thr, x, y in zip(MARKED_THRESHOLDS, recall_list, precision_list):
        ax.text(x, y, f"{thr:.2f}", fontsize=_FONTSIZE, ha="left", va="bottom",
                fontweight="bold", zorder=4)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", which="major", labelsize=_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Recall", fontsize=_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=_FONTSIZE, fontweight="bold")
    # AP only, matching the ROC figure's AUROC-only title.
    ax.set_title(f"AP = {ap:.2f}", fontsize=_FONTSIZE, fontweight="bold")
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
