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
* **The MEAN curves (`plot_mean_curve`) are laid out for a four-panel row.** The
  manuscript places the four configurations' mean PR curves side by side, each
  a quarter of the text width, so those panels carry nothing that would be
  duplicated four times or be illegible at that size: no title (the AP values
  are in a table), no colourbar (the shared legend is a separate figure from
  `plot_threshold_legend()`, placed once beside the row), no between-seed
  threshold tick marks, and fonts 1.5x the single-model figures'. The random-
  model baseline is labelled "random model" without its numeric value.
"""

import os

import matplotlib
import numpy as np
from matplotlib.cm import ScalarMappable

# Embed TrueType (Type 42) fonts in PDF output instead of matplotlib's default
# Type 3 bitmap-like fonts: publishers reject PDFs containing Type 3 fonts, and
# every curve here ends up in the manuscript.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
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

# The prediction VIEWER's hard-prediction / confusion panels use this cut. It
# was historically shared with the bird's-eye vessel map, which no longer uses
# any fixed threshold at all - vessel_map.py chooses its operating point per run
# (precision >= 0.9 within 3 mm, recall maximised) - so this now affects the
# viewer only. It is an empirical pick made by eye on the old silicone map,
# confined to that qualitative display and touching nothing that is scored;
# override with DIFFTACTILE_MAP_THRESHOLD (e.g. 0.5 for the conventional cut).
MAP_DECISION_THRESHOLD = float(os.environ.get("DIFFTACTILE_MAP_THRESHOLD", 0.58))

# Decision thresholds annotated as discrete markers on every ROC curve.
MARKED_THRESHOLDS = np.array([0.40, 0.50, 0.60])

# Colourmap encoding the decision threshold along the ROC curve, always
# spanning the full [0, 1] range so colours are comparable between figures.
THRESHOLD_CMAP = "viridis"
THRESHOLD_NORM = Normalize(vmin=0.0, vmax=1.0)

_FONTSIZE = 20

# The mean-curve panels and the standalone threshold legend are printed at a
# quarter of the manuscript's text width, so their text is 1.5x the single-model
# figures' to stay legible at that size.
_MEAN_FONTSIZE = int(round(1.5 * _FONTSIZE))


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


def mean_curve_with_band(curves, grid=None):
    """Vertically average a set of (x, y) curves onto a common x grid.

    `curves` is a list of (x, y) arrays, one per seed. Returns
    (grid, mean, std) with mean and std taken across seeds at each grid point.

    VERTICAL AVERAGING is the standard way to combine ROC curves (Fawcett 2006):
    interpolate each curve onto a shared x grid, then average y. The alternative
    - averaging the raw (x, y) vertices - is meaningless, because the curves have
    different numbers of vertices at different places.

    `np.interp` needs an increasing x, and both ROC (fpr, tpr) and PR
    (recall, precision) are supplied in an order that may not be, so each curve
    is sorted first. For PR the curve is not a function of recall in general
    (precision can revisit a recall value); taking the max precision at each
    recall is the usual convention and is what the sort-then-interp does here
    once duplicates are collapsed.
    """
    grid = np.linspace(0.0, 1.0, 201) if grid is None else np.asarray(grid)
    stacked = []
    for x, y in curves:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        # Collapse duplicate x values, keeping the best (highest) y - the
        # convention for both ROC and PR.
        uniq_x, inverse = np.unique(x, return_inverse=True)
        uniq_y = np.zeros_like(uniq_x)
        np.maximum.at(uniq_y, inverse, y)
        stacked.append(np.interp(grid, uniq_x, uniq_y))
    stacked = np.vstack(stacked)
    return grid, stacked.mean(axis=0), stacked.std(axis=0, ddof=1 if len(stacked) > 1 else 0)


def mean_threshold_along_grid(curves_thr, grid=None):
    """Mean decision threshold at each point of the shared x grid.

    This is what keeps the threshold colour-coding meaningful on a mean curve.
    Each seed reaches a given false-positive rate at its OWN threshold, so a
    single fixed threshold cannot be attached to a point of the averaged curve.
    Interpolating each seed's threshold onto the same grid and averaging gives
    "the typical threshold at which a model reaches this operating point", which
    is the only reading the mean curve supports.

    `curves_thr` is a list of (x, thresholds) arrays parallel to the curves
    passed to `mean_curve_with_band`. Returns (grid, mean_threshold, std).
    """
    grid = np.linspace(0.0, 1.0, 201) if grid is None else np.asarray(grid)
    stacked = []
    for x, thr in curves_thr:
        x = np.asarray(x, dtype=float)
        thr = np.clip(np.asarray(thr, dtype=float), 0.0, 1.0)
        order = np.argsort(x)
        x, thr = x[order], thr[order]
        uniq_x, index = np.unique(x, return_index=True)
        stacked.append(np.interp(grid, uniq_x, thr[index]))
    stacked = np.vstack(stacked)
    return grid, stacked.mean(axis=0), stacked.std(axis=0, ddof=1 if len(stacked) > 1 else 0)


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


def plot_mean_curve(plt, curves, curves_thr, scores, out_path, kind,
                    baseline=None):
    """Mean ROC or PR curve across seeds, with a ±1 std band.

    `kind` is "roc" or "pr"; `curves` is a list of (x, y) per seed, `curves_thr`
    the matching (x, thresholds), and `scores` the per-seed AUROC or AP.
    `scores` is accepted for interface symmetry with the callers' bookkeeping
    but is not printed: the panels carry no title, because the manuscript
    tabulates the mean ± std separately and a title would repeat it four times.

    HOW THE THRESHOLD COLOUR-CODING SURVIVES AVERAGING. On a single-model curve
    each vertex has one exact decision threshold, and the colour is that value.
    A mean curve has no such thing: seed 3 might reach FPR 0.2 at threshold 0.55
    while seed 7 reaches it at 0.48. The colour here is therefore the MEAN
    threshold at which the models reach that operating point - the same
    vertical-averaging idea applied to the threshold axis rather than to y.

    That is a genuinely weaker claim than the single-curve colouring, which is
    why the shared legend (`plot_threshold_legend()`) is labelled "mean decision
    threshold" rather than pretending a point has one threshold. The legend is
    NOT drawn on the panel itself: four panels sit in one row in the manuscript
    and one legend to the right of the row serves all of them.

    The individual seed curves are drawn faintly underneath. The band shows the
    spread; the thin lines show whether it comes from a couple of outliers or
    from even scatter, which a band alone cannot distinguish.
    """
    grid, mean_y, std_y = mean_curve_with_band(curves)
    _, mean_thr, _ = mean_threshold_along_grid(curves_thr)

    # Square-ish: there is no colourbar beside the axes, and a quarter-width
    # panel reads best when the two unit axes get equal room.
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Random-model reference, drawn first. For PR it is the positive rate; the
    # value itself is not printed (it is quoted in the results table), only
    # what the line means.
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "k-", alpha=0.5, linewidth=6.0, zorder=1)
    elif baseline is not None:
        ax.axhline(baseline, color="k", linestyle="--", alpha=0.5, linewidth=4.0,
                   zorder=1)
        ax.text(0.98, baseline, "random model", fontsize=_MEAN_FONTSIZE - 9,
                ha="right", va="bottom", fontweight="bold", zorder=5)

    # Every seed, faintly: shows whether the band is even scatter or an outlier.
    for x, y in curves:
        order = np.argsort(np.asarray(x, dtype=float))
        ax.plot(np.asarray(x)[order], np.asarray(y)[order], color="0.55",
                linewidth=1.2, alpha=0.7, zorder=2)

    # +/- 1 std band around the mean.
    ax.fill_between(grid, np.clip(mean_y - std_y, 0, 1), np.clip(mean_y + std_y, 0, 1),
                    color="0.35", alpha=0.22, linewidth=0, zorder=3)

    # The mean curve, coloured by the MEAN threshold at each operating point.
    points = np.column_stack([grid, mean_y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=THRESHOLD_CMAP, norm=THRESHOLD_NORM,
                        linewidth=6.0, alpha=0.95, zorder=4,
                        capstyle="round", joinstyle="round")
    lc.set_array(0.5 * (mean_thr[:-1] + mean_thr[1:]))
    ax.add_collection(lc)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", which="major", labelsize=_MEAN_FONTSIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    if kind == "roc":
        ax.set_xlabel("False Positive Rate", fontsize=_MEAN_FONTSIZE, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=_MEAN_FONTSIZE, fontweight="bold")
    else:
        ax.set_xlabel("Recall", fontsize=_MEAN_FONTSIZE, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=_MEAN_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    finish_plot(plt, out_path, format="pdf", dpi=300)


def plot_threshold_legend(plt, out_path, label="Mean decision threshold"):
    """The threshold colour-coding legend as a figure of its own, to `out_path`.

    A vertical colourbar over the fixed [0, 1] threshold scale that every curve
    in this module shares, with nothing else on the page. It exists so a row of
    mean-curve panels (`plot_mean_curve()`, which draws no colourbar) can carry
    ONE legend placed beside the row instead of four copies. Fonts are the
    mean-curve panels' (1.5x the single-model figures'), so it reads at the
    same size as the panels it sits next to.

    `label` defaults to the mean-curve wording; pass "Decision threshold" if
    the legend is to accompany single-model curves instead.
    """
    # Shorter than the 6 in panels on purpose: set at the same scale beside a
    # row of them, the bar then spans the panels' AXES (which start above the
    # x-axis label) instead of overshooting the top of the row. The bar's axes
    # are placed by hand (a colourbar-only figure defeats tight_layout) and the
    # page is cropped to the drawn content when saved.
    fig = plt.figure(figsize=(2.6, 4.8))
    ax = fig.add_axes([0.05, 0.03, 0.22, 0.94])
    cbar = fig.colorbar(
        ScalarMappable(norm=THRESHOLD_NORM, cmap=THRESHOLD_CMAP), cax=ax,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    cbar.set_label(label, fontsize=_MEAN_FONTSIZE - 3, fontweight="bold")
    cbar.ax.tick_params(labelsize=_MEAN_FONTSIZE - 6, width=3.0, length=8.0)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    cbar.outline.set_linewidth(3.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    finish_plot(plt, out_path, format="pdf", dpi=300, bbox_inches="tight",
                pad_inches=0.05)


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
