#!/usr/bin/env python3
"""Draw the poster's "Centreline detection and localisation" block as two bar panels.

Frame space only, centrelines only (the user's 2026-09-04 instruction), all four models:

    left   detection F1 at tau = 5 mm, higher is better. TP = a matched centreline pair
           within tau, FP = an unmatched prediction, FN = an unmatched true vessel -- the
           same confusion the count matrix beside it is built from.
    right  localisation error, OSPA at c = tau = 5 mm, lower is better. The penalties are
           already inside it: a false alarm costs 2 x 5 mm and a miss 1 x 5 mm, so a model
           cannot buy a low distance by predicting nothing or by predicting everywhere.
           It is capped at 10 mm, which is why Meat->Silicone sits so close to the top.

One series, so no legend: the panel titles name what is plotted. Numbers come from
analysis/results/detection_ospa.json (written by detection_ospa.py); nothing is hard-coded
here but the model order and the labels. Each bar is drawn in its model's colour (analysis/scripts/model_colours.py), the code the whole
poster uses; the legend for that code is the baselines table's caption and nowhere else.

It also writes the 2x2 "vessel present or absent" confusion figure and the vessel-count
confusion table that sit beside it: the 4x4 matrix of
Sim->Meat's per-frame centreline counts over {0, 1, 2, 3+}, as a LaTeX tabular the poster source
inputs, so the poster's numbers cannot drift from the evaluation either.

Output: analysis/figures/{detection_bars.pdf, binary_confusion.pdf, detection_confusion.tex}
Run:    python scripts/detection_bars.py
"""
import json
import os

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGURES, RESULTS, ensure_dirs  # noqa: E402

SRC = os.path.join(RESULTS, "detection_ospa.json")
OUT = os.path.join(FIGURES, "detection_bars.pdf")

CONFIGS = ["Sim->Sim", "Sim->Silicone", "Sim->Meat", "Meat->Silicone"]
XLABELS = ["Sim→\nSim", "Sim→\nSilicone", "Sim→\nMeat", "Meat→\nSilicone"]
CUTOFF = "5"
# One colour per model, from the poster's shared code (light Sim->Sim to dark
# Meat->Silicone); the legend for it lives only in the baselines-table caption.
from model_colours import COLOURS as BAR_COLOURS  # noqa: E402
INK = "#1A1A1A"

# The figure is drawn at exactly the width it is included at, so these are printed sizes.
FIG_W, FIG_H = 10.93, 2.55   # the narrow column beside the baselines table, so drawn unscaled
FS_TICK, FS_TITLE, FS_VALUE = 17, 18, 14
BAR_W = 0.56


def load():
    with open(SRC) as fh:
        res = json.load(fh)["cutoffs"][CUTOFF]
    frame = [res[c]["centreline"]["frame"] for c in CONFIGS]
    return {"f1": [d["f1"] for d in frame], "ospa": [d["ospa_mean_mm"] for d in frame]}


def panel(ax, values, title, fmt, headroom=1.26):
    """One bar panel: four models, every bar labelled with its own number."""
    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, values, BAR_W, color=BAR_COLOURS, linewidth=0)
    top = max(values)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02 * top, fmt.format(v),
                ha="center", va="bottom", fontsize=FS_VALUE, color=INK)
    ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=FS_TICK, color=INK)
    ax.set_ylim(0, top * headroom)
    # No y axis: every bar carries its own number, so ticks and a grid would be redundant ink.
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="x", length=0, pad=6)


# --------------------------------------------------------------------- 2-class confusion
# "Vessel present or absent", per VIDEO FRAME, exactly the rule the baselines table's
# 2-class rows are built on (scripts/comparable_metrics.py, per_frame): a frame HAS a vessel
# if any marker is labelled one, and the model SAYS vessel if any marker scores >= THRESH.
# Frame space, not the top-view map: this metric is per frame.
BIN_CFG = "A-to-C"          # Sim->Meat (user's choice)
BIN_OUT = os.path.join(FIGURES, "binary_confusion.pdf")
BIN_W, BIN_H = 4.9, 3.5


def binary_confusion(config=BIN_CFG):
    """[[TN, FP], [FN, TP]] over the frames of one configuration."""
    import numpy as np
    z = np.load(os.path.join(RESULTS, f"frame_space_predictions_{config}.npz"))
    labels, probs = z["labels"].astype(bool), z["probs"].astype(float)
    present = labels.any(axis=1)
    said = (probs >= 0.5).any(axis=1)
    return np.array([[int((~present & ~said).sum()), int((~present & said).sum())],
                     [int((present & ~said).sum()), int((present & said).sum())]])


def binary_figure(m, path):
    """The 2x2 matrix as a row-normalised heat map with the frame counts written in."""
    fig, ax = plt.subplots(figsize=(BIN_W, BIN_H))
    row = m.sum(1, keepdims=True)
    frac = np.divide(m, np.where(row == 0, 1, row))
    # BLUE, not the model green (user, 2026-09-04): a green heat map beside the green model
    # code would read as if the shade meant a model. The ramp ends at ICLBlue!45, the same
    # tint the vessel-count table's fullest cell uses, so the numbers stay BLACK -- no text
    # on this poster is white.
    ax.imshow(frac, cmap=LinearSegmentedColormap.from_list(
        "iclblue", ["#FFFFFF", "#8C8CE8"]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=26,
                    color=INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks([0, 1], ["absent", "present"], fontsize=FS_TICK)
    ax.set_yticks([0, 1], ["absent", "present"], fontsize=FS_TICK)
    ax.set_xlabel("predicted", fontsize=FS_TITLE)
    ax.set_ylabel("true", fontsize=FS_TITLE)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    print("written", path)
    return m


CONF_CFG = "Sim->Meat"          # the one model whose test set spans all four vessel counts
CONF_OUT = os.path.join(FIGURES, "detection_confusion.tex")


def confusion_tex():
    """Sim->Meat's per-frame vessel-count confusion as a poster tabular, cells shaded by row.

    detection_ospa.py stores a 5x5 matrix over {0, 1, 2, 3, 4+}; the last two columns are
    folded into one "3+" column, because three is the most vessels any label holds and
    everything above it is the same kind of over-count.
    """
    with open(SRC) as fh:
        m = json.load(fh)["cutoffs"][CUTOFF][CONF_CFG]["centreline"]["frame"][
            "count_confusion_gt_by_pred"]
    m = [[r[0], r[1], r[2], r[3] + r[4]] for r in m[:4]]
    rows = []
    for i, r in enumerate(m):
        tot = sum(r) or 1
        cells = []
        for j, v in enumerate(r):
            tint = int(round(45 * v / tot))
            bold = r"\textbf{%d}" % v if i == j else str(v)
            cells.append((r"\cellcolor{ICLBlue!%d}" % tint if tint else "") + bold)
        rows.append(f"\t\t\t{i} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    with open(CONF_OUT, "w") as fh:
        fh.write(f"""% GENERATED by scripts/detection_bars.py -- do not edit by hand.
% Sim->Meat, centrelines, video-frame space: how many vessels the model finds in a frame
% against how many the ground truth holds. Rows are the truth, columns the prediction,
% cells are numbers of frames shaded by their share of the row.
		\\begin{{tabular}}{{@{{}}l r r r r@{{}}}}
			\\toprule
			& \\multicolumn{{4}}{{c}}{{\\textbf{{predicted}}}}\\\\
			\\cmidrule(l){{2-5}}
			\\textbf{{true}} & 0 & 1 & 2 & 3$+$\\\\
			\\midrule
{body}
			\\bottomrule
		\\end{{tabular}}
""")
    print("written", CONF_OUT)
    return m


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    ensure_dirs()
    d = load()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    panel(axes[0], d["f1"], "Detection $F_1$ at 5 mm ↑", "{:.2f}")
    panel(axes[1], d["ospa"], "Localisation error, OSPA [mm] ↓", "{:.2f}")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.845, bottom=0.275, wspace=0.10)
    fig.savefig(OUT)
    print("written", OUT)
    for k, fmt in (("f1", "{:.2f}"), ("ospa", "{:.2f}")):
        print(f"  {k:5s} " + "  ".join(fmt.format(v) for v in d[k]))
    for r in confusion_tex():
        print("  conf " + "  ".join(f"{v:3d}" for v in r))
    b = binary_figure(binary_confusion(), BIN_OUT)
    print(f"  2-class Sim->Meat: TN {b[0,0]} FP {b[0,1]} FN {b[1,0]} TP {b[1,1]}  "
          f"acc {(b[0,0]+b[1,1])/b.sum():.2f}  recall {b[1,1]/b[1].sum():.2f}  "
          f"false-alarm {b[0,1]/b[0].sum():.2f}")


if __name__ == "__main__":
    main()
