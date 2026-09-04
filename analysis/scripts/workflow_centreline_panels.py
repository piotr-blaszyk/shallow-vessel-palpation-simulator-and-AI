#!/usr/bin/env python3
"""Pick and render the centreline/centroid panels that go on the poster's workflow figure.

Two blocks are produced, both drawn from exactly the objects `scripts/detection_ospa.py`
scores (same extraction, same near-duplicate filtering, same Hungarian matching):

    frame-space vessel centrelines and centroids   4 panels + a legend panel
    top-view-space vessel centrelines and centroids 4 panels + a legend panel

Choosing the frame-space panels
-------------------------------
A 4x4 grid: one row per ground-truth vessel count (0, 1, 2, 3) and one column per model,
so the reader can walk down a column and see the same model on progressively harder
frames, or across a row and compare the four models on the same difficulty. Three cells
are empty by construction -- only Sim->Meat has any 3-vessel frame in its test set -- and
those are drawn as an explicit "none in this test set" placeholder rather than left blank.

The composite loss on a frame f of configuration g whose ground truth holds n vessels is

    L(f) = 0.5 * |OSPA(f)  - median_{g,n} OSPA| / s_{g,n}(OSPA)
         + 0.5 * |F1(f)    - median_{g,n} F1|   / s_{g,n}(F1)

The medians and spreads are STRATIFIED BY GROUND-TRUTH COUNT: a frame with n vessels is
compared only against that model's frames that also hold n vessels (so there are up to
four medians per metric per model, for n = 0, 1, 2, 3). Without the stratification an
"easy" panel would be judged against a pool containing the difficult frames, and the
0-vessel frames -- 379 of Sim->Sim's 675 -- would set a median no 1-vessel frame can
sit near.

* OSPA is the localisation error with the penalties already baked in (distances capped at
  the cutoff c = tau, plus c per unmatched object with the false positive charged
  ALPHA_FP x and the false negative ALPHA_FN x, normalised by max(#pred, #GT)).
* F1 at tau is the confusion-matrix score (TP = matched pairs within tau, FP = unmatched
  predictions, FN = unmatched ground-truth objects).
* s_{g,n} is the robust scale 1.4826 * MAD over that model's n-vessel frames, which puts
  both terms in comparable units of "typical deviation" and so gives the two halves equal
  weight. Degenerate spreads fall back to the mean absolute deviation and then to 1.0 (a
  constant metric contributes 0 to the loss anyway).

The frame panels share one set of axis limits and one figure size, so their heights are
equal by construction. Titles are deliberately absent (the poster captions the panels);
axis labels and ticks stay, because the millimetre scale is the point of the figure.

Usage:  python scripts/workflow_centreline_panels.py [--tau 5.0]
Writes: analysis/figures/workflow/cl_grid_<n>_<config>.png, cl_legend.png,
        analysis/figures/workflow/cl_map_<config>.png, frame_grid.tex,
        analysis/results/workflow_panel_choice.json
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGURES, RESULTS  # noqa: E402
from detection_ospa import C_MM, PRETTY, evaluate, frame_points, map_points  # noqa: E402
from model_colours import MODELS  # noqa: E402
from detection_overlays import CONF, GT_COLOUR, PR_COLOUR, STYLE, label_objects, states  # noqa: E402

# Ground-truth vessel count each frame-space panel must show (user's specification):
# an "easy" row of 1-vessel frames and a "difficult" row of the rare counts.
COUNTS = (0, 1, 2, 3)          # one grid row per ground-truth vessel count
SHORT = {"A-to-A": "sim_to_sim", "A-to-B": "sim_to_silicone",
         "A-to-C": "sim_to_meat", "C-to-B": "meat_to_silicone"}
OUT_IMG = FIGURES / "workflow"
FS_TICK, FS_LABEL, FS_LEG = 19, 21, 20


# --------------------------------------------------------------------------- the loss
def f1_at(rec, tau):
    """F1 with TP = pairs within tau, FP = unmatched predictions, FN = unmatched GT.

    A frame with no ground-truth vessel and no prediction scores 1.0: there was nothing to
    find and nothing was invented. (Scoring it 0 would make the vessel-free frames -- 379 of
    Sim->Sim's 675 -- drag the model's median down and misdescribe a correct silence.)
    """
    tp = sum(1 for _, _, d in rec["matched"] if d <= tau)
    fp, fn = rec["n_pred"] - tp, rec["n_gt"] - tp
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    return 2 * tp / (2 * tp + fp + fn)


def robust_scale(v):
    """1.4826 * MAD, falling back to the mean absolute deviation and then to 1.0.

    Both loss terms are divided by this, which is what makes their weights equal: each
    contributes "how many typical deviations from the model's median this frame is".
    """
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad > 1e-9:
        return 1.4826 * mad
    aad = float(np.mean(np.abs(v - med)))
    return aad if aad > 1e-9 else 1.0


def score_config(cfg, tau, c):
    """Per-frame OSPA / F1 / loss for one configuration, with STRATIFIED reference values.

    Returns (rows, stats) where `stats[n]` holds the median and robust scale of each metric
    over that configuration's frames whose ground truth holds exactly n vessels. Every frame's
    loss is measured against its own stratum, so an easy frame is judged against easy frames.
    """
    rows = []
    for name, pr, gt, ctx in frame_points(cfg, "centreline"):
        rec = evaluate(pr, gt, "centreline", c=c, tau=tau)
        rows.append(dict(name=name, pred=pr, gt=gt, ctx=ctx, rec=rec, n_gt=rec["n_gt"],
                         ospa=float(rec["ospa"]), f1=float(f1_at(rec, tau))))
    stats = {}
    for n in sorted({r["n_gt"] for r in rows}):
        grp = [r for r in rows if r["n_gt"] == n]
        o = np.array([r["ospa"] for r in grp])
        f = np.array([r["f1"] for r in grp])
        stats[n] = {"n_frames": len(grp),
                    "median_ospa": float(np.median(o)), "median_f1": float(np.median(f)),
                    "scale_ospa": robust_scale(o), "scale_f1": robust_scale(f)}
    for r in rows:
        st = stats[r["n_gt"]]
        r["loss"] = float(0.5 * abs(r["ospa"] - st["median_ospa"]) / st["scale_ospa"]
                          + 0.5 * abs(r["f1"] - st["median_f1"]) / st["scale_f1"])
    return rows, stats


# --------------------------------------------------------------------------- rendering
def draw_objs(ax, objs, labels, state, colour, extend, ms, lw, dy, show_labels):
    """Poster variant of `detection_overlays.draw`: both the segment and the centroid, with the
    index labels optional (the top-view maps carry 20+ objects and the labels would collide)."""
    for i, o in enumerate(objs):
        st = STYLE[state[i]]
        if o["ends"] is not None:
            a, b = o["ends"]
            d = b - a
            n = np.linalg.norm(d)
            if n > 0:
                a, b = a - d / n * extend, b + d / n * extend
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=lw * st["lw"] / 2.6,
                    linestyle=st["dash"] or "-", zorder=6, solid_capstyle="round")
            if show_labels:
                ax.annotate(st["fmt"].format(labels[i]), b, textcoords="offset points",
                            xytext=(9 * np.sign(d[0] or 1), dy), color=colour,
                            fontsize=FS_TICK, fontweight="bold", zorder=9)
        ax.plot(*o["centroid"], marker="o", ms=ms, zorder=7,
                mfc=colour if st["fill"] else "none",
                mec="k" if st["fill"] else colour, mew=1.2 if st["fill"] else 2.4)
        if show_labels and o["ends"] is None:
            ax.annotate(st["fmt"].format(labels[i]), o["centroid"], textcoords="offset points",
                        xytext=(9, -dy), color=colour, fontsize=FS_TICK, fontweight="bold",
                        zorder=9)


def style_axes(ax, xlabel="x (mm)", ylabel="y (mm)", xon=True, yon=True):
    """Ticks and titles only where the grid asks for them.

    In the 4x4 grid the x axis is drawn on the bottom row and the y axis on the left column;
    sixteen repeats of both would take more of the block than the plots do. Every panel shares
    one set of limits, so the axes of the edge panels apply to the whole grid.
    """
    ax.tick_params(labelsize=FS_TICK, length=5, width=1.2, pad=2,
                   labelbottom=xon, labelleft=yon, bottom=xon, left=yon)
    ax.set_xticks([30, 40, 50]); ax.set_yticks([10, 20, 30])
    if xon:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=1)
    if yon:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)


# Panel geometry, in inches. Every panel's PLOT BOX is exactly SIDE x SIDE; only the margin
# that carries an axis grows, so the figure size differs by cell and the four cell shapes are
# recombined into a flush grid by grid_tex() below.
SIDE = 3.2
M_AXIS, M_BARE = 0.86, 0.05


def cell_figure(xon, yon):
    """A figure whose axes box is SIDE x SIDE with margins that depend on which axes show."""
    ml, mb = (M_AXIS if yon else M_BARE), (M_AXIS if xon else M_BARE)
    w, h = ml + SIDE + M_BARE, mb + SIDE + M_BARE
    fig = plt.figure(figsize=(w, h), dpi=240)
    ax = fig.add_axes([ml / w, mb / h, SIDE / w, SIDE / h])
    return fig, ax, w, h


def placeholder_panel(text, xlim, ylim, xon, yon, path):
    """A cell no frame can fill: only Sim->Meat's test set holds any 3-vessel frame.

    It keeps the grid's axes and limits -- otherwise the bottom row's x axis would appear
    under one column only -- but is greyed and labelled, so it cannot be mistaken for a real
    frame on which the model simply predicted nothing.
    """
    fig, ax, _, _ = cell_figure(xon, yon)
    ax.set_facecolor("0.93")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    style_axes(ax, "sensor x (mm)", "sensor y (mm)", xon, yon)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=FS_LABEL,
            color="0.30", transform=ax.transAxes)
    fig.savefig(path)
    plt.close(fig)


def frame_panel(row, tau, xlim, ylim, xon, yon, path):
    """One frame-space panel: confusion-coloured markers + centrelines + centroids."""
    pred, gt, ctx, rec = row["pred"], row["gt"], row["ctx"], row["rec"]
    matched = rec["matched"]
    gt_lab, pr_lab = label_objects(pred, gt, matched)
    xy, gm, pm = ctx["xy"], ctx["gt_mask"], ctx["pred_mask"]

    fig, ax, _, _ = cell_figure(xon, yon)
    for key, m in (("tn", ~gm & ~pm), ("tp", gm & pm), ("fn", gm & ~pm), ("fp", ~gm & pm)):
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=95 if key != "tn" else 42, c=CONF[key],
                       edgecolors="none", zorder=3 if key != "tn" else 2)
    draw_objs(ax, gt, gt_lab, states(len(gt), matched, "gt", tau), GT_COLOUR,
              0.7, 15, 4.5, -18, True)
    draw_objs(ax, pred, pr_lab, states(len(pred), matched, "pred", tau), PR_COLOUR,
              0.7, 15, 4.5, 20, True)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)          # one set of limits for all sixteen cells
    style_axes(ax, "sensor x (mm)", "sensor y (mm)", xon, yon)
    fig.savefig(path)                               # no tight bbox: the grid must stay flush
    plt.close(fig)


def map_panel(cfg, name, pred, gt, ctx, tau, path):
    """One top-view panel, at the map's own aspect ratio (they differ per phantom)."""
    rec = evaluate(pred, gt, "centreline", c=tau, tau=tau)
    matched = rec["matched"]
    gt_lab, pr_lab = label_objects(pred, gt, matched)
    gm, pm = ctx["gt_mask"], ctx["pred_mask"]
    rgb = np.zeros((*gm.shape, 3)) + matplotlib.colors.to_rgb(CONF["tn"])
    for key, m in (("tp", gm & pm), ("fn", gm & ~pm), ("fp", ~gm & pm)):
        rgb[m] = matplotlib.colors.to_rgb(CONF[key])
    # The four maps go on the poster at one shared HEIGHT, so the figure height -- not its
    # width -- is what must be constant: otherwise a wide flat map (Sim->Meat is 157x48) comes
    # out with axis text several times larger than its neighbours' once they are equalised.
    h, w = gm.shape
    fig, ax = plt.subplots(figsize=(3.5 * w / h, 3.5), dpi=220)
    ax.imshow(rgb, interpolation="nearest")
    # No index labels here: the silicone map carries 23 objects and the labels would collide.
    draw_objs(ax, gt, gt_lab, states(len(gt), matched, "gt", tau), GT_COLOUR,
              2.0, 9, 3.4, -16, False)
    draw_objs(ax, pred, pr_lab, states(len(pred), matched, "pred", tau), PR_COLOUR,
              2.0, 9, 3.4, 18, False)
    ax.set_xlim(-0.05 * w, 1.05 * w); ax.set_ylim(1.05 * h, -0.05 * h)
    ax.set_xlabel("x (mm)", fontsize=FS_LABEL); ax.set_ylabel("y (mm)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=6, width=1.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.3)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return rec


def legend_panel(path, height_in):
    """The shared key, as its own sub-figure (one column, no axes)."""
    h = [plt.Line2D([], [], marker="o", ls="", ms=13, color=CONF["tp"], label="TP"),
         plt.Line2D([], [], marker="o", ls="", ms=13, color=CONF["fn"], label="FN, missed"),
         plt.Line2D([], [], marker="o", ls="", ms=13, color=CONF["fp"], label="FP, false alarm"),
         plt.Line2D([], [], marker="o", ls="", ms=13, color=CONF["tn"], label="TN"),
         plt.Line2D([], [], color=GT_COLOUR, lw=4, marker="o", ms=13, mec="k",
                    label="ground truth"),
         plt.Line2D([], [], color=PR_COLOUR, lw=4, marker="o", ms=13, mec="k",
                    label="prediction"),
         plt.Line2D([], [], color="0.25", lw=4, label='matched  "k"'),
         plt.Line2D([], [], color="0.25", lw=2, label='over 5 mm  "k*"'),
         plt.Line2D([], [], color="0.25", lw=3, ls=(0, (5, 4)), marker="o", ms=13, mfc="none",
                    mec="0.25", label='unmatched  "(k)"')]
    # One tall column that spans the whole grid. The figure size is FIXED (no tight bbox) and
    # the entries are spread to fill it: stretching the key vertically is what keeps it narrow
    # once LaTeX scales it to the grid's height, and keeps its text the same size as the grid's.
    fig = plt.figure(figsize=(3.15, height_in), dpi=220)
    fig.legend(handles=h, loc="center left", fontsize=FS_LEG, frameon=False,
               handlelength=2.2, labelspacing=3.6, borderpad=0.1, borderaxespad=0.15)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- grid geometry
def grid_tex(cells, side_cm, gap_cm, key_png, path):
    """Write the TikZ nodes of the 4x4 grid and its key, in cm, as a snippet the poster source inputs.

    The cell shapes differ (only the left column carries a y axis, only the bottom row an x
    axis), so a hand-written grid would drift out of alignment the moment a font or a tick
    changed. Here the plot boxes are flush by construction: a cell is placed by its plot box
    and the axis margin hangs outside the grid.

    Everything is emitted with the block's top-left corner at (0, 0) and growing right and
    down, and the corners are marked (fgtl) and (fgbr), so the poster source only has to shift a
    scope and fit a group node to those two coordinates.
    """
    extra = side_cm * M_AXIS / SIDE          # width (or height) an axis margin adds, in cm
    bare = side_cm * M_BARE / SIDE
    lab_w, hdr_h = 0.80, 0.85                # room for the rotated row labels / column headers
    colw = [side_cm + extra + bare] + [side_cm + 2 * bare] * 3
    rowh = [side_cm + 2 * bare] * 3 + [side_cm + extra + bare]
    colx, x = [], lab_w + extra + bare - (extra + bare)   # column 0's axis margin sits after the labels
    x = lab_w
    for w in colw:
        colx.append(x); x += w + gap_cm
    rowy, y = [], -hdr_h
    for h in rowh:
        rowy.append(y); y -= h + gap_cm
    right, bottom = x - gap_cm, y + gap_cm
    key_h = rowy[0] - bottom                 # the key spans the grid, axis margins and all
    kw, kh = Image.open(key_png).size
    key_x = right + gap_cm + 0.35
    key_w = key_h * kw / kh
    block_w, block_h = key_x + key_w, hdr_h + (rowy[0] - bottom)

    out = ["% GENERATED by scripts/workflow_centreline_panels.py -- do not edit by hand.",
           "% 4x4 grid of frame-space panels: rows are the ground-truth vessel count 0..3,",
           "% columns are the four models, and the key spans the grid on the right. Axes are",
           "% drawn once per edge (left column, bottom row) and their margins hang outside the",
           "% grid, so the plot boxes stay flush. The block's top-left corner is (0, 0) and it",
           "% grows right and down; (fgtl) and (fgbr) mark its corners for a fit= group node.",
           "\\coordinate (fgtl) at (0, 0);",
           f"\\coordinate (fgbr) at ({block_w:.3f}, {-block_h:.3f});"]
    for i, n in enumerate(COUNTS):
        for j, cfg in enumerate(PRETTY):
            out.append(f"\\node[bimg] (fg{i}{j}) at ({colx[j]:.3f}, {rowy[i]:.3f}) "
                       f"{{\\includegraphics[width={colw[j]:.3f}cm]{{workflow/{cells[(n, cfg)]}}}}};")
        mid = rowy[i] - bare - side_cm / 2
        lab = f"{n} vessel" + ("" if n == 1 else "s")
        out.append(f"\\node[bsub, rotate=90, anchor=south] at ({lab_w - 0.10:.3f}, "
                   f"{mid:.3f}) {{{lab}}};")
    # The column headers carry the poster's model colour code (analysis/scripts/model_colours.py):
    # a SQUARE of the model's green to the left of the name, light Sim->Sim to dark
    # Meat->Silicone. The square, never the text colour, carries the code -- no text on this
    # poster is white (user, 2026-09-04). The legend lives only in the baselines caption.
    for j, cfg in enumerate(PRETTY):
        name = PRETTY[cfg].replace("->", "$\\rightarrow$")
        cname = MODELS[j][2]
        cx = colx[j] + colw[j] - bare - side_cm / 2
        out.append(f"\\node[bsub, anchor=south] at ({cx:.3f}, {rowy[0] + 0.12:.3f}) "
                   f"{{\\msq{{{cname}}}~{name}}};")
    out.append(f"\\node[bimg] (fgkey) at ({key_x:.3f}, {rowy[0]:.3f}) "
               f"{{\\includegraphics[height={key_h:.3f}cm]{{workflow/{key_png.name}}}}};")
    path.write_text("\n".join(out) + "\n")
    return {"side_cm": side_cm, "grid_w_cm": right - colx[0], "grid_h_cm": key_h,
            "block_w_cm": block_w, "block_h_cm": block_h, "key_w_cm": key_w}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", type=float, default=C_MM)
    ap.add_argument("--side", type=float, default=4.2,
                    help="cell plot-box side in cm inside the workflow picture")
    ap.add_argument("--out-dir", default=None,
                    help="where the panels go (default: analysis/figures/workflow)")
    ap.add_argument("--all-axes", action="store_true",
                    help="draw both axes on EVERY frame-space panel, so all sixteen come out "
                         "the same size. The poster wants the opposite -- axes on the left "
                         "column and bottom row only, so the cells butt up flush -- but a "
                         "responsive web grid scales every cell to one width, and unequal "
                         "figure shapes then render the plots at visibly different sizes.")
    args = ap.parse_args()
    tau = c = args.tau
    out_img = Path(args.out_dir) if args.out_dir else OUT_IMG
    out_img.mkdir(parents=True, exist_ok=True)
    out = {"tau_mm": tau, "c_mm": c,
           "loss": "0.5*|OSPA-median|/robust_scale(OSPA) + 0.5*|F1-median|/robust_scale(F1)",
           "configs": {}}

    # ---- frame space: score every frame, then fill each cell of the 4x4 grid with the
    # lowest-loss frame of that model holding exactly that many ground-truth vessels. Ties
    # (Sim->Sim's 0-vessel frames are all identical) break on the frame name, so the choice
    # is reproducible.
    chosen, missing = {}, []
    for cfg in PRETTY:
        rows, stats = score_config(cfg, tau, c)
        out["configs"][cfg] = {"n_frames": len(rows), "strata": {}, "cells": {}}
        for n in COUNTS:
            pool = [r for r in rows if r["n_gt"] == n]
            if not pool:
                missing.append((n, cfg))
                out["configs"][cfg]["cells"][str(n)] = {"n_frames_in_stratum": 0,
                                                        "chosen": None}
                print(f"{PRETTY[cfg]:15s} gt={n}  no frame in this test set -> n/a cell")
                continue
            st = stats[n]
            out["configs"][cfg]["strata"][str(n)] = {
                "n_frames": st["n_frames"], "median_ospa_mm": st["median_ospa"],
                "median_f1": st["median_f1"], "scale_ospa_mm": st["scale_ospa"],
                "scale_f1": st["scale_f1"]}
            best = min(pool, key=lambda r: (r["loss"], r["name"]))
            chosen[(n, cfg)] = best
            out["configs"][cfg]["cells"][str(n)] = {
                "n_frames_in_stratum": len(pool),
                "chosen": {k: best[k] for k in ("name", "ospa", "f1", "loss", "n_gt")},
                "chosen_n_pred": best["rec"]["n_pred"],
                "chosen_matched": len(best["rec"]["matched"]),
                "pool_loss_min_median_max": [float(np.min([r["loss"] for r in pool])),
                                             float(np.median([r["loss"] for r in pool])),
                                             float(np.max([r["loss"] for r in pool]))]}
            print(f"{PRETTY[cfg]:15s} gt={n} stratum n={len(pool):4d} "
                  f"median OSPA {st['median_ospa']:5.2f} F1 {st['median_f1']:.2f}  ->  "
                  f"{best['name']}  OSPA {best['ospa']:5.2f}  F1 {best['f1']:.2f}  "
                  f"L {best['loss']:.3f}")

    # One set of axis limits for all sixteen cells, so the edge axes speak for the whole grid.
    pts = np.vstack([v["ctx"]["xy"] for v in chosen.values()])
    pad = 0.06 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ylim = (pts[:, 1].max() + pad, pts[:, 1].min() - pad)   # image convention: y downwards
    cells = {}
    for i, n in enumerate(COUNTS):
        for cfg in PRETTY:
            xon, yon = ((n == COUNTS[-1]), (cfg == next(iter(PRETTY)))) \
                if not args.all_axes else (True, True)
            fname = f"cl_grid_{n}_{SHORT[cfg]}.png"
            cells[(n, cfg)] = fname
            if (n, cfg) in chosen:
                frame_panel(chosen[(n, cfg)], tau, xlim, ylim, xon, yon, out_img / fname)
            else:
                placeholder_panel(f"n/a\n\nno {n}-vessel frame\nin this test set",
                                  xlim, ylim, xon, yon, out_img / fname)
    legend_panel(out_img / "cl_legend.png", 12.0)
    geo = grid_tex(cells, args.side, 0.28, out_img / "cl_legend.png",
                   out_img / "frame_grid.tex")
    out["grid"] = geo
    out["missing_cells"] = [f"{n} vessels, {PRETTY[cfg]}" for n, cfg in missing]
    print(f"\ncell side {geo['side_cm']:.2f} cm; block {geo['block_w_cm']:.2f} x "
          f"{geo['block_h_cm']:.2f} cm (key {geo['key_w_cm']:.2f} cm wide)")

    # ---- top-view space: one map per configuration, drawn at its own aspect ratio.
    # Sim->Meat has ten trial maps rather than one stitched map, so the same representativeness
    # rule picks among them: the map whose OSPA is closest to the median over that config.
    for cfg in PRETTY:
        maps = list(map_points(cfg, "centreline"))
        recs = [evaluate(pr, gt, "centreline", c=c, tau=tau) for _, pr, gt, _ in maps]
        med = float(np.median([r["ospa"] for r in recs]))
        k = int(np.argmin([abs(r["ospa"] - med) for r in recs]))
        name, pr, gt, ctx = maps[k]
        rec = map_panel(cfg, name, pr, gt, ctx, tau, out_img / f"cl_map_{SHORT[cfg]}.png")
        out["configs"][cfg]["top_view"] = {
            "map": name, "n_maps": len(maps), "median_ospa_mm": med,
            "n_gt": rec["n_gt"], "n_pred": rec["n_pred"],
            "matched": len(rec["matched"]), "ospa_mm": rec["ospa"]}

    if not args.out_dir:      # a run into another directory must not overwrite the record
        (RESULTS / "workflow_panel_choice.json").write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {out_img}")


if __name__ == "__main__":
    main()
