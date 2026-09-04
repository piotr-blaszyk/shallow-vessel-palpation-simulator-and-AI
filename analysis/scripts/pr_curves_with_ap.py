"""Re-render the manuscript's Fig. 6 PR panels with the mean AP added to each title.

The manuscript's four precision-recall panels (mean over the five seeds of the
published sweep, +-1 std band, colour = mean decision threshold) are drawn by
`cnn/seed_sweep.py::_plot_mean_curves` -> `cnn/curve_plots.py::plot_mean_curve`
in the code repository. This script re-uses exactly those functions and the
same per-seed score files, changing ONE thing: each panel title gains a third
line "AP=0.xx" (the seed-mean average precision that the manuscript tabulates),
so the poster can drop the table. The panel geometry is taken from
curve_plots' constants, with the title band enlarged for the third line.

Outputs (analysis/figures/pr/):
  mean_pr_curve_<config>.pdf  x4   (A-to-A carries the y axis, as in the manuscript)
  curve_legend_vertical.pdf, xlabel_recall.pdf   the shared legend (vertical, drawn at its
                                        printed size) and the shared x label
  ap_values.json                        the AP mean/std used in the titles

Run:  python analysis/scripts/pr_curves_with_ap.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGURES, REPO as _REPO  # noqa: E402

REPO = str(_REPO)
OUT = os.path.join(FIGURES, "pr")
sys.path.insert(0, REPO)
os.chdir(REPO)   # the sweep loader resolves saved_models_sweeps/ against the working directory

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from difftactile.cnn import curve_plots  # noqa: E402
from difftactile.cnn.curve_plots import plot_mean_curve, plot_curve_legend, plot_axis_label  # noqa: E402
from difftactile.cnn.model_selection import load_sweep, published_sweep_dir  # noqa: E402
from difftactile.cnn.seed_sweep import PANEL_TITLES, Y_AXIS_CONFIG  # noqa: E402

# Poster scale. The manuscript panels are ~4.5 cm wide on the page; on the
# poster the same row is ~32 cm wide, i.e. each panel ~8 cm, so the fonts must
# be ~2.5x larger RELATIVE to the axes box for the ticks to stay legible. The
# axes box is kept at the manuscript's size and the text-bearing margins are
# widened to fit the bigger fonts; three title lines instead of two.
SCALE = 2.6
_BASE_MEAN_FONTSIZE = curve_plots._MEAN_FONTSIZE   # the manuscript's value, before scaling
curve_plots._MEAN_FONTSIZE = int(round(curve_plots._MEAN_FONTSIZE * SCALE))
curve_plots._PANEL_LEFT_WITH_Y = 1.6 * SCALE
curve_plots._PANEL_LEFT_NO_Y = 0.40 * SCALE
curve_plots._PANEL_RIGHT = 0.35 * SCALE
curve_plots._PANEL_BOTTOM_NO_X = 0.55 * SCALE
curve_plots._PANEL_TOP_WITH_TITLE = 1.0 * SCALE * 1.5


def poster_legend_vertical(out_path):
    """Vertical poster legend, drawn at its printed size, to sit BESIDE the panel row.

    The poster used to carry the manuscript's one-line legend above the four panels
    (supervisor feedback, 2026-09-04: put the legend beside the figure and lay it out
    vertically). Stacking the two entries -- the threshold colourbar and the random-model
    baseline -- into a narrow column frees the strip above the panels, which is what lets
    the panel row grow to the full width of the results column.

    The figure is built at the size it is printed at (FIG_W x FIG_H inches) and saved with
    no tight-bbox crop, so `FS` below is the point size that ends up on the poster and the
    poster includes the PDF at its natural size. Colours and the baseline line style are
    imported from the code repository's own plotting module, so the sample cannot drift
    from the line the panels draw.
    """
    from matplotlib.cm import ScalarMappable
    from difftactile.cnn.curve_plots import THRESHOLD_CMAP, THRESHOLD_NORM, BASELINE_STYLE

    FS = 18                       # printed point size of the legend captions
    FIG_W, FIG_H = 2.15, 3.15     # inches; 2.15 in = 155 pt wide, and no taller than the panel
                                  # row it stands beside, so it never drives the block height
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # --- entry 1: the mean-decision-threshold colourbar, caption above, ticks to its right.
    fig.text(0.03, 0.995, "Mean\ndecision\nthreshold", fontsize=FS, fontweight="bold",
             ha="left", va="top")
    cax = fig.add_axes([0.10, 0.410, 0.15, 0.295])
    cbar = fig.colorbar(ScalarMappable(norm=THRESHOLD_NORM, cmap=THRESHOLD_CMAP), cax=cax,
                        orientation="vertical", ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=FS - 3, width=1.6, length=6.0)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    cbar.outline.set_linewidth(1.6)

    # --- entry 2: a sample of the random-model baseline, caption under it.
    lax = fig.add_axes([0.05, 0.335, 0.34, 0.02])
    lax.axis("off")
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axhline(0.5, **BASELINE_STYLE)
    fig.text(0.03, 0.295, "random model\n(precision $=$\npositive rate)", fontsize=FS,
             fontweight="bold", ha="left", va="top")

    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)


def main():
    sweep_dir = published_sweep_dir()
    sweep = load_sweep(sweep_dir)
    os.makedirs(OUT, exist_ok=True)
    ap_out = {}
    for summary in sweep["summaries"]:
        config = summary["config"]
        curves, thr, aps, baselines = [], [], [], []
        for run in summary["runs"]:
            seed_dir = os.path.join(sweep_dir, os.path.basename(run["artifact_dir"]))
            with np.load(os.path.join(seed_dir, f"scores_{config}.npz")) as d:
                probs, labels = d["probs"], d["labels"]
            precision, recall, pthr = precision_recall_curve(labels, probs)
            curves.append((recall, precision))
            thr.append((recall[:len(pthr)], pthr))
            aps.append(run["ap"])
            baselines.append(float(np.mean(labels)))
        ap_mean, ap_std = float(np.mean(aps)), float(np.std(aps, ddof=1))
        ap_out[config] = {"ap_mean": ap_mean, "ap_std": ap_std, "seeds": len(aps)}
        title = f"{PANEL_TITLES[config]}\nAP={ap_mean:.2f}"
        plot_mean_curve(plt, curves, thr, aps, os.path.join(OUT, f"mean_pr_curve_{config}.pdf"),
                        kind="pr", baseline=float(np.mean(baselines)),
                        show_yaxis=(config == Y_AXIS_CONFIG), show_xlabel=False, title=title)
        print(f"{config}: AP {ap_mean:.3f} +- {ap_std:.3f} -> {title!r}")
    plot_axis_label(plt, "Recall", os.path.join(OUT, "xlabel_recall.pdf"))
    poster_legend_vertical(os.path.join(OUT, "curve_legend_vertical.pdf"))
    with open(os.path.join(OUT, "ap_values.json"), "w") as f:
        json.dump(ap_out, f, indent=2)


if __name__ == "__main__":
    main()
