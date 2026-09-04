#!/usr/bin/env python3
"""Draw the poster's "Segmentation statistics" block as two side-by-side bar panels.

Since 2026-09-04 this writes TWO INDEPENDENT figures, each with its own title and its own
legend, because they sit in different places on the poster:

    Images/segmentation_iou.pdf       foreground IoU, higher is better -- under the
                                      baselines table, beside the precision-recall curves
    Images/segmentation_distance.pdf  mean distance from a PREDICTED vessel element to the
                                      nearest TRUE vessel element, lower is better -- in the
                                      narrow column beside the baselines table, because the
                                      table quotes distances

Each of those two figures is itself TWO SIDE-BY-SIDE PANELS (user, 2026-09-04), built
exactly like Images/detection_bars.pdf: the left panel is the video-frame, per-marker
evaluation and the right one the top-view-map, per-pixel evaluation, each panel titled
with the space it measures.  There is therefore NO LEGEND -- the panel titles carry what
the legend used to say, and one bar colour serves both.  The two panels of a figure SHARE
one y limit, because they are the same quantity in the same units: independent limits would
make a 5.49 mm bar look the same height as a 4.12 mm one.

DISTANCE DEFINITION.  The manuscript's (one-directional) definition is used, not
the symmetric one the poster carried before: only elements the model PREDICTS to
be vessel are scored, i.e. the TP and FP cells of the confusion matrix, each
measuring its distance to the nearest ground-truth vessel element.  The test sets
are strongly background-dominated, so a symmetric mean (which also scores the
predicted-background elements, and hence the TN cell) is pulled towards zero by
the class balance rather than by localisation quality.

Numbers come from results/symmetric_distances.json (field `one_directional_mm`,
and `fg_iou`), written by scripts/symmetric_distances.py; nothing is hard-coded
here except the model order and the labels.  The map-space distances reproduce
the manuscript's Table 4 exactly (1.05 / 1.21 / 5.49 / 1.31 mm).

Style: DejaVu Sans, matching the precision-recall panels beside it.  Since the
1 : 2 : 2 column change of 2026-09-04 the block sits in the RIGHT sub-column of the
Results column, 0.400 x \colwide = 558.8 pt = 7.73 in wide, and the figure is drawn
at exactly that width, so it is included at width=\linewidth with no rescaling and
the font sizes below are the sizes printed on the poster.  The two panels stay side
by side (the user's 2026-09-03 instruction); the narrower canvas is paid for with a
taller figure and slightly smaller type, not by stacking them.

Output: Images/segmentation_iou.pdf, Images/segmentation_distance.pdf
Run:    python scripts/segmentation_bars.py
"""

import json
import os

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGURES, RESULTS, ensure_dirs  # noqa: E402

SRC = os.path.join(RESULTS, "symmetric_distances.json")
OUT_IOU = os.path.join(FIGURES, "segmentation_iou.pdf")
OUT_DIST = os.path.join(FIGURES, "segmentation_distance.pdf")

# Model order and two-line x labels (the arrow renders in DejaVu Sans).
CONFIGS = ["A-to-A", "A-to-B", "A-to-C", "C-to-B"]
XLABELS = ["Sim→\nSim", "Sim→\nSilicone", "Sim→\nMeat", "Meat→\nSilicone"]

# The two evaluation spaces, one per panel; the panel title says which is which.
SERIES = [
    ("Video frame, per marker", "frame"),
    ("Top-view map, per pixel", "map"),
]
# Colour encodes the MODEL, not the panel: the poster's shared code (light Sim->Sim to
# dark Meat->Silicone), whose legend is the baselines table's caption and nowhere else.
from model_colours import COLOURS as BAR_COLOURS  # noqa: E402

# Poster-scale typography. The figure is 7.73 in wide = the width of the Results
# column's right sub-panel, so these point sizes are the printed ones.
FS_TICK = 17      # x tick labels
FS_TITLE = 18     # panel titles
FS_VALUE = 14     # the number printed on top of each bar

# Each figure is drawn at the true width of the slot it goes in, so it is included
# unscaled and the point sizes above are the printed ones.
# Each figure keeps the footprint it had when it was one grouped panel: the Results column
# has ~20 pt of slack, and the space the legend used to occupy is given back to the bars.
W_IOU, W_DIST, FIG_H = 9.84, 10.93, 2.20
BAR_W = 0.56      # bar width, as in detection_bars.py -- one series per panel now
INK = "#1A1A1A"


def load():
    """Return {panel_key: {series_key: [value per config]}} from the results JSON."""
    with open(SRC) as fh:
        res = json.load(fh)
    return {
        "iou": {s: [res[c][s]["fg_iou"] for c in CONFIGS] for _, s in SERIES},
        "dist": {s: [res[c][s]["one_directional_mm"] for c in CONFIGS] for _, s in SERIES},
    }


def panel(ax, values, title, fmt, top):
    """One bar panel: four models in one evaluation space, every bar labelled.

    `top` is passed in rather than taken from `values` so that the two panels of a figure
    share a y limit and their bar heights stay comparable.
    """
    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, values, BAR_W, color=BAR_COLOURS, linewidth=0)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.02 * top,
                fmt.format(value), ha="center", va="bottom",
                fontsize=FS_VALUE, color=INK)
    ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=FS_TICK, color=INK)
    ax.set_ylim(0, top * 1.26)
    # No y axis: every bar carries its own number, so ticks and grid would be
    # redundant ink. Only the baseline is kept.
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="x", length=0, pad=6)


def figure(values, suffix, fmt, path, width):
    """One standalone figure: the two evaluation spaces as two side-by-side bar panels.

    `suffix` is appended to each panel title and carries the unit and the direction of
    goodness, e.g. " [mm] v" -- the metric's own name is the poster's \figtitle above it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(width, FIG_H))
    top = max(max(v) for v in values.values())
    for ax, (name, key) in zip(axes, SERIES):
        panel(ax, values[key], name + suffix, fmt, top)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.845, bottom=0.275, wspace=0.10)
    fig.savefig(path)
    print("written", path)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    ensure_dirs()
    data = load()
    figure(data["iou"], " \u2191", "{:.2f}", OUT_IOU, W_IOU)
    figure(data["dist"], " [mm] \u2193", "{:.2f}", OUT_DIST, W_DIST)
    for key, fmt in (("iou", "{:.2f}"), ("dist", "{:.2f}")):
        for name, s in SERIES:
            print(f"  {key:5s} {name:26s} " + "  ".join(fmt.format(v) for v in data[key][s]))


if __name__ == "__main__":
    main()
