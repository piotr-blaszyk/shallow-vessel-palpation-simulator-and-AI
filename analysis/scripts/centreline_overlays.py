#!/usr/bin/env python3
"""Per-vessel centreline overlays for every data point behind the centreline metrics.

WHAT THE CENTRELINE METRICS ACTUALLY DO
------------------------------------------------------------------------------------
`comparable_metrics.py` never separates vessels. For one video frame it takes the
centroid of ALL vessel-labelled markers, the centroid of ALL predicted markers, and
ONE principal axis (PCA/SVD) through all vessel-labelled markers; the lateral error
is the offset between the two centroids projected onto the normal of that axis. Two
vessels in a frame are therefore merged into a single "mean vessel", not separated and
not discarded. Map space (`symmetric_distances.py`) uses no centreline at all - only a
Euclidean distance transform. The only literal centreline in the code base is the
simulator's own vein geometry, written out as `ground_truth_centreline_reference.png`
for the Sim->Sim map; it is a reference picture, not an estimate, and feeds no metric.

WHAT THIS SCRIPT ADDS
---------------------
The per-vessel decomposition the user asked to be able to eyeball. For each scored data
point it clusters the vessel elements into individual vessels, fits one centreline per
vessel, matches ground-truth vessels to predicted vessels, and renders the standard
confusion overlay with the centrelines drawn on top:

    magenta  ground-truth vessel centrelines, labelled with their vessel index
    orange   predicted vessel centrelines, labelled with the index of the ground-truth
             vessel they were matched to (so magenta 2 and orange 2 are a pair)

An index that appears in only one colour is an unmatched vessel: a ground-truth vessel
the model missed entirely, or a predicted blob with no ground-truth counterpart.

Both spaces are covered:

  frame space   markers of one central video frame, 1920x1080 sensor px, 55 px = 2 mm.
                Clustered with DBSCAN at 1.45 x the median inter-marker spacing of the
                dataset (~1.4 x connects a marker only to its immediate hexagonal
                neighbours, so two vessels separated by a clear marker gap stay apart).
                Rendered for every vessel-bearing frame, i.e. every frame that enters
                the centre / lateral / hit-rate aggregates.
                The quantities the METRIC uses are drawn too, in a lighter weight: the
                merged ground-truth centroid (magenta star), the merged predicted
                centroid (orange star), the merged principal axis (dashed magenta) and
                the lateral axis the offset is projected onto (dashed grey).

  top-view map  one image per test-set trial: the whole-phantom map for Sim->Sim,
                Sim->Silicone and Meat->Silicone, and the ten trial maps of Sim->Meat.
                1 px = 1 mm; clustered with DBSCAN(eps=3 mm, min_samples=3), which
                recovers exactly the ten sweeps of the silicone phantom map.
                Note: the silicone map saved by the manuscript's run is ALREADY the
                stitched whole-phantom map (100 x 180 mm, "ten sweeps"), and its
                run.json carries no per-sweep provenance, so the ten sweeps are
                recovered by clustering rather than by splitting and re-stitching.

Three variants of every figure are written, so the two geometries can be looked at alone or
together:  `centreline/` draws only the fitted centrelines, `centroid/` only the vessels'
centres of mass, `both/` draws them together. Every object carries its vessel index in all
three. The lighter reference geometry follows the variant too: the merged principal axis and
its normal appear in the centreline variants, the merged ground-truth and predicted centroids
(the two stars) in the centroid variants.

Output tree (timestamped, so runs never overwrite each other):

    analysis/centrelines/<YYYYmmdd-HHMMSS>/
        README.md, params.json, vessels.json
        frame_space/<Config>/<variant>/frame_<i>.png
        top_view/<Config>/<variant>/<map>.png
        contact_sheets/<space>_<Config>_<variant>.png

Usage:  python scripts/centreline_overlays.py [--out-root analysis/centrelines] [--limit N]
"""
import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ANALYSIS, RESULTS, vessel_map_run  # noqa: E402
NAMES = {"A-to-A": "Sim-to-Sim", "A-to-B": "Sim-to-Silicone",
         "A-to-C": "Sim-to-Meat", "C-to-B": "Meat-to-Silicone"}
PRETTY = {"A-to-A": "Sim→Sim", "A-to-B": "Sim→Silicone",
          "A-to-C": "Sim→Meat", "C-to-B": "Meat→Silicone"}

MM_PER_PX = 2.0 / 55.0        # frame space: 55 sensor px = 2 mm
THRESH = 0.5                  # frame space operating point (manuscript Table 4)
EPS_SPACING_FACTOR = 1.45     # frame-space DBSCAN eps, in units of marker spacing
MAP_EPS_MM = 3.0              # map-space DBSCAN eps
MAP_MIN_SAMPLES = 3
MAP_MIN_CLUSTER_PX = 5        # ignore specks smaller than this in map space
GATE_FRAME_MM = 10.0          # max centroid separation for a GT<->pred match, frame space
GATE_MAP_MM = 20.0            # ditto, map space

MODES = ("centreline", "centroid", "both")   # what each figure variant draws
MODE_TITLE = {"centreline": "centrelines only", "centroid": "centroids only",
              "both": "centrelines + centroids"}

GT_COLOUR = "#FF00FF"         # magenta: ground-truth centrelines
PR_COLOUR = "#FF8000"         # orange:  predicted centrelines
CONF = {"tp": "#00C000", "fn": "#E00000", "fp": "#0050FF", "tn": "#BFBFBF"}


# --------------------------------------------------------------------------- vessels
def cluster(points, eps, min_samples=1, min_size=1):
    """Split `points` (n, 2) into vessels with DBSCAN; returns a list of index arrays.

    Clusters are returned largest-first so that vessel indices are stable and the most
    substantial vessel is always index 0.
    """
    if len(points) == 0:
        return []
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    groups = [np.flatnonzero(lab == k) for k in sorted(set(lab)) if k != -1]
    groups = [g for g in groups if len(g) >= min_size]
    return sorted(groups, key=len, reverse=True)


def centreline(points):
    """Total-least-squares centreline of one vessel.

    Returns (centroid, unit axis, the two endpoints where the cluster's own extreme
    points project onto the axis). A single point has no direction, so `axis` is None.
    """
    c = points.mean(axis=0)
    if len(points) < 2:
        return c, None, None
    axis = np.linalg.svd(points - c, full_matrices=False)[2][0]
    # The PCA axis sign is arbitrary; fix it to point right (or down, for a vertical
    # vessel) so that a vessel's index label always sits at the same end of its line.
    dom = 0 if abs(axis[0]) >= abs(axis[1]) else 1   # the dominant component decides, so
    if axis[dom] < 0:                                # a near-vertical vessel stays stable
        axis = -axis
    t = (points - c) @ axis
    return c, axis, (c + t.min() * axis, c + t.max() * axis)


def extract(points, eps, min_samples=1, min_size=1):
    """All vessels of one data point: index arrays plus their centrelines."""
    out = []
    for idx in cluster(points, eps, min_samples, min_size):
        c, axis, ends = centreline(points[idx])
        out.append({"idx": idx, "centroid": c, "axis": axis, "ends": ends,
                    "n": int(len(idx))})
    return out


def match(gt, pred, gate):
    """Match ground-truth vessels to predicted vessels by centroid distance.

    A Hungarian assignment minimises the total centroid distance; pairs farther apart
    than `gate` are rejected. Returns (pairs, unmatched_gt, unmatched_pred) where a pair
    is (gt index, pred index) and the two share a displayed vessel index.
    """
    if not gt or not pred:
        return [], list(range(len(gt))), list(range(len(pred)))
    cost = np.linalg.norm(np.array([g["centroid"] for g in gt])[:, None, :]
                          - np.array([p["centroid"] for p in pred])[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(ri, ci) if cost[r, c] <= gate]
    ug = [i for i in range(len(gt)) if i not in {r for r, _ in pairs}]
    up = [j for j in range(len(pred)) if j not in {c for _, c in pairs}]
    return pairs, ug, up


def label_vessels(gt, pred, gate):
    """Give every vessel its displayed index.

    Ground-truth vessels are numbered 0, 1, 2, ... in spatial reading order; a predicted
    vessel matched to one of them carries that same number, so magenta k and orange k are
    a pair. Predicted vessels with no counterpart are numbered after the ground-truth ones,
    which is exactly what an index appearing in only one colour means.
    """
    pairs, ug, up = match(gt, pred, gate)

    # Vessels in one data point are near-parallel, so index them ACROSS the common vessel
    # direction: down the image for the silicone sweeps, left-to-right for the meat straws.
    # The common direction is the dominant eigenvector of sum(a a^T), which is immune to the
    # arbitrary sign of each individual axis.
    axes = [v["axis"] for v in list(gt) + list(pred) if v["axis"] is not None]
    a = (np.linalg.eigh(sum(np.outer(x, x) for x in axes))[1][:, -1] if axes
         else np.array([1.0, 0.0]))
    nrm = np.array([-a[1], a[0]])
    nrm = nrm if nrm[1] >= 0 else -nrm      # index downwards, i.e. in reading order

    def order(vessels):
        return lambda i: (float(vessels[i]["centroid"] @ nrm), float(vessels[i]["centroid"] @ a))

    gt_lab = {g: k for k, g in enumerate(sorted(range(len(gt)), key=order(gt)))}
    pr_lab = {p: gt_lab[g] for g, p in pairs}
    for k, p in enumerate(sorted(up, key=order(pred))):
        pr_lab[p] = len(gt) + k
    return gt_lab, pr_lab, pairs


# --------------------------------------------------------------------------- drawing
def draw_vessels(ax, vessels, labels, colour, lw=2.0, extend=0.0, dy=-10, mode="both"):
    """Draw one side's vessels with their indices, in the requested mode.

    `mode` is "centreline" (the centreline segment only), "centroid" (the vessel's centre of mass
    only) or "both". Every drawn object carries the vessel's index in the same colour, so a
    magenta k and an orange k are a matched pair whichever mode the figure is in.
    """
    def tag_at(point, tag, off):
        ax.annotate(tag, point, textcoords="offset points", xytext=off, color=colour,
                    fontsize=12, fontweight="bold", zorder=8)

    for i, v in enumerate(vessels):
        tag = str(labels[i])
        c = v["centroid"]
        if v["ends"] is None:
            # A lone element has no direction, so its centroid IS the whole vessel: one
            # cross, one label, in every mode.
            ax.plot(*c, marker="x", ms=9, mew=2.5, color=colour, zorder=6)
            tag_at(c, tag, (7, -dy))
            continue
        if mode in ("centreline", "both"):
            a, b = v["ends"]
            d = b - a
            n = np.linalg.norm(d)
            if n > 0:                       # a small overhang makes short vessels legible
                a, b = a - d / n * extend, b + d / n * extend
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=lw, zorder=6,
                    solid_capstyle="round")
            tag_at(b, tag, (8 * np.sign(d[0] or 1), dy))
        if mode in ("centroid", "both"):
            ax.plot(*c, marker="o", ms=8, mfc=colour, mec="k", mew=0.9, zorder=7)
            tag_at(c, tag, (8, -dy))


def confusion_scatter(ax, xy, gt, pred, size=42):
    """The standard confusion overlay for markers: TP green, FN red, FP blue, TN grey."""
    masks = {"tn": ~gt & ~pred, "tp": gt & pred, "fn": gt & ~pred, "fp": ~gt & pred}
    for key in ("tn", "tp", "fn", "fp"):
        m = masks[key]
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size if key != "tn" else size * 0.45,
                       c=CONF[key], edgecolors="none", zorder=3 if key != "tn" else 2)


def legend(ax, mode, merged=False):
    """The confusion key plus whichever vessel geometry this figure draws."""
    h = [plt.Line2D([], [], marker="o", ls="", color=CONF["tp"], label="TP"),
         plt.Line2D([], [], marker="o", ls="", color=CONF["fn"], label="FN (missed)"),
         plt.Line2D([], [], marker="o", ls="", color=CONF["fp"], label="FP (false alarm)"),
         plt.Line2D([], [], marker="o", ls="", color=CONF["tn"], label="TN")]
    if mode in ("centreline", "both"):
        h += [plt.Line2D([], [], color=GT_COLOUR, lw=2.5, label="ground-truth centreline"),
              plt.Line2D([], [], color=PR_COLOUR, lw=2.5, label="predicted centreline")]
    if mode in ("centroid", "both"):
        h += [plt.Line2D([], [], marker="o", ls="", mfc=GT_COLOUR, mec="k",
                         label="ground-truth centroid"),
              plt.Line2D([], [], marker="o", ls="", mfc=PR_COLOUR, mec="k",
                         label="predicted centroid")]
    if merged:
        if mode in ("centreline", "both"):
            h += [plt.Line2D([], [], color=GT_COLOUR, lw=1.0, ls=(0, (6, 4)),
                             label="merged axis (metric)"),
                  plt.Line2D([], [], color="0.35", lw=1.0, ls=(0, (2, 3)),
                             label="lateral axis (metric)")]
        if mode in ("centroid", "both"):
            h += [plt.Line2D([], [], marker="*", ls="", ms=11, mfc=GT_COLOUR, mec="k",
                             label="merged GT centroid (metric)"),
                  plt.Line2D([], [], marker="*", ls="", ms=11, mfc=PR_COLOUR, mec="k",
                             label="merged pred. centroid (metric)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=False, handlelength=1.6)


# --------------------------------------------------------------------------- frame space
def frame_space(config, out_dir, limit=None):
    """One overlay per vessel-bearing frame of `config`; returns the per-frame records."""
    npz = np.load(RESULTS / f"frame_space_predictions_{config}.npz", allow_pickle=True)
    probs, labels, pos = npz["probs"], npz["labels"].astype(bool), npz["pos"].astype(float)
    pred_all = probs >= THRESH

    pos = pos * MM_PER_PX                 # work in mm throughout: the metrics are in mm
    spacing = float(np.median(cKDTree(pos[0]).query(pos[0], k=2)[0][:, 1]))
    eps = EPS_SPACING_FACTOR * spacing
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    n_done = 0
    for i in range(len(probs)):
        gt, pr = labels[i], pred_all[i]
        if not gt.any():                      # vessel-free frames enter no centreline metric
            continue
        if limit is not None and n_done >= limit:
            break
        xy = pos[i]
        gts = extract(xy[gt], eps)
        prs = extract(xy[pr], eps) if pr.any() else []
        gl, pl, pairs = label_vessels(gts, prs, GATE_FRAME_MM)

        # the quantities the poster's metric actually uses, for reference
        gc = xy[gt].mean(axis=0)
        pc = xy[pr].mean(axis=0) if pr.any() else None
        merged_axis = (np.linalg.svd(xy[gt] - gc, full_matrices=False)[2][0]
                       if gt.sum() >= 3 else None)
        lat = None
        if pc is not None and merged_axis is not None:
            n = np.array([-merged_axis[1], merged_axis[0]])
            n = n if n[1] >= 0 else -n
            lat = float((pc - gc) @ n)
        err = float(np.linalg.norm(pc - gc)) if pc is not None else None

        for mode in MODES:
            fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
            confusion_scatter(ax, xy, gt, pr)
            draw_vessels(ax, gts, gl, GT_COLOUR, extend=0.35 * spacing, mode=mode)
            draw_vessels(ax, prs, pl, PR_COLOUR, lw=2.4, extend=0.35 * spacing, dy=12,
                         mode=mode)

            # The geometry the PUBLISHED metric uses, drawn lightly so it never competes with
            # the per-vessel objects: one merged axis and its normal in the line modes, the two
            # merged centroids (the stars) in the centroid modes.
            if merged_axis is not None and mode in ("centreline", "both"):
                span = np.abs((xy[gt] - gc) @ merged_axis).max()
                a, b = gc - merged_axis * span, gc + merged_axis * span
                ax.plot([a[0], b[0]], [a[1], b[1]], color=GT_COLOUR, lw=1.0, ls=(0, (6, 4)),
                        alpha=0.55, zorder=4)
                nrm = np.array([-merged_axis[1], merged_axis[0]])
                ax.plot([gc[0] - nrm[0] * span * 0.5, gc[0] + nrm[0] * span * 0.5],
                        [gc[1] - nrm[1] * span * 0.5, gc[1] + nrm[1] * span * 0.5],
                        color="0.35", lw=1.0, ls=(0, (2, 3)), alpha=0.7, zorder=4)
            if mode in ("centroid", "both"):
                ax.plot(*gc, marker="*", ms=13, color=GT_COLOUR, mec="k", mew=0.6, zorder=9)
                if pc is not None:
                    ax.plot(*pc, marker="*", ms=13, color=PR_COLOUR, mec="k", mew=0.6, zorder=9)

            ax.set_aspect("equal")
            ax.margins(0.10)              # room for the index labels just outside the markers
            ax.invert_yaxis()             # image convention: +y is down
            ax.set_xlabel("sensor x (mm)"); ax.set_ylabel("sensor y (mm)")
            head = (f"{PRETTY[config]}  frame {i}  [{MODE_TITLE[mode]}]   |   "
                    f"GT vessels {len(gts)}, predicted {len(prs)}, matched {len(pairs)}")
            sub = ("centre error " + (f"{err:.2f} mm" if err is not None else "n/a")
                   + ",  lateral " + (f"{lat:+.2f} mm" if lat is not None else "n/a")
                   + ("   [in the aggregate]" if lat is not None else "   [not aggregated]"))
            ax.set_title(head + "\n" + sub, fontsize=9)
            legend(ax, mode, merged=True)
            fig.tight_layout()
            (out_dir / mode).mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / mode / f"frame_{i:04d}.png", bbox_inches="tight")
            plt.close(fig)

        records.append({"frame": i, "n_gt_vessels": len(gts), "n_pred_vessels": len(prs),
                        "n_matched": len(pairs), "centre_err_mm": err, "lateral_mm": lat,
                        "gt_sizes": [v["n"] for v in gts],
                        "pred_sizes": [v["n"] for v in prs]})
        n_done += 1
    return {"eps_mm": eps, "marker_spacing_mm": spacing, "frames": records}


# --------------------------------------------------------------------------- map space
def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def map_dirs(config):
    run = vessel_map_run(config)
    subs = sorted(p for p in run.iterdir() if p.is_dir())
    return subs or [run]


def top_view(config, out_dir):
    """One overlay per top-view map of `config`; returns the per-map records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for d in map_dirs(config):
        gt = load_mask(d / "ground_truth.png")
        pr = load_mask(d / "prediction.png")
        gt_xy = np.column_stack(np.nonzero(gt))[:, ::-1].astype(float)   # (x, y) = (col, row)
        pr_xy = np.column_stack(np.nonzero(pr))[:, ::-1].astype(float)
        gts = extract(gt_xy, MAP_EPS_MM, MAP_MIN_SAMPLES, MAP_MIN_CLUSTER_PX)
        prs = extract(pr_xy, MAP_EPS_MM, MAP_MIN_SAMPLES, MAP_MIN_CLUSTER_PX)
        gl, pl, pairs = label_vessels(gts, prs, GATE_MAP_MM)

        rgb = np.zeros((*gt.shape, 3))
        rgb[...] = matplotlib.colors.to_rgb(CONF["tn"])
        for key, m in (("tp", gt & pr), ("fn", gt & ~pr), ("fp", ~gt & pr)):
            rgb[m] = matplotlib.colors.to_rgb(CONF[key])

        h, w = gt.shape
        name = d.name if d != vessel_map_run(config) else "whole_map"
        for mode in MODES:
            fig, ax = plt.subplots(figsize=(max(5.0, w / 22), max(3.2, h / 22)), dpi=130)
            ax.imshow(rgb, interpolation="nearest")
            draw_vessels(ax, gts, gl, GT_COLOUR, lw=1.8, extend=2.0, mode=mode)
            draw_vessels(ax, prs, pl, PR_COLOUR, lw=1.8, extend=2.0, dy=12, mode=mode)
            ax.set_xlim(-0.06 * w, 1.06 * w); ax.set_ylim(1.08 * h, -0.08 * h)  # room for labels
            ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
            ax.set_title(f"{PRETTY[config]}  {name}  [{MODE_TITLE[mode]}]   |   "
                         f"GT vessels {len(gts)}, predicted {len(prs)}, matched {len(pairs)}",
                         fontsize=9)
            legend(ax, mode)
            fig.tight_layout()
            (out_dir / mode).mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / mode / f"{name}.png", bbox_inches="tight")
            plt.close(fig)

        records.append({"map": name, "shape": [int(h), int(w)],
                        "n_gt_vessels": len(gts), "n_pred_vessels": len(prs),
                        "n_matched": len(pairs),
                        "gt_sizes": [v["n"] for v in gts],
                        "pred_sizes": [v["n"] for v in prs]})
    return records


# --------------------------------------------------------------------------- contact sheets
def contact_sheet(src_dir, out_path, cols=8, thumb=260, cap=64):
    """A grid of thumbnails, so a whole config can be scanned in one image."""
    files = sorted(src_dir.glob("*.png"))[:cap]
    if not files:
        return
    ims = [Image.open(f).convert("RGB") for f in files]
    ims = [im.resize((thumb, int(thumb * im.height / im.width))) for im in ims]
    rows = (len(ims) + cols - 1) // cols
    ch = max(im.height for im in ims)
    sheet = Image.new("RGB", (cols * thumb, rows * ch), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * thumb, (k // cols) * ch))
    sheet.save(out_path)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(ANALYSIS / "overlays/centrelines"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the frames rendered per config (for a quick look)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_root) / stamp
    (root / "contact_sheets").mkdir(parents=True, exist_ok=True)

    summary = {"timestamp": stamp, "configs": {}}
    for cfg, name in NAMES.items():
        fs = frame_space(cfg, root / "frame_space" / name, args.limit)
        tv = top_view(cfg, root / "top_view" / name)
        summary["configs"][name] = {"config": cfg, "frame_space": fs, "top_view": tv}
        for mode in MODES:
            contact_sheet(root / "frame_space" / name / mode,
                          root / "contact_sheets" / f"frame_space_{name}_{mode}.png")
            contact_sheet(root / "top_view" / name / mode,
                          root / "contact_sheets" / f"top_view_{name}_{mode}.png",
                          cols=2, thumb=700)
        print(f"{name:18s} frame-space frames {len(fs['frames']):4d}   "
              f"top-view maps {len(tv):2d}   "
              f"GT vessels/map {[t['n_gt_vessels'] for t in tv]}")

    params = {"timestamp": stamp, "threshold": THRESH, "mm_per_px_frame": MM_PER_PX,
              "eps_spacing_factor": EPS_SPACING_FACTOR, "map_eps_mm": MAP_EPS_MM,
              "map_min_samples": MAP_MIN_SAMPLES, "map_min_cluster_px": MAP_MIN_CLUSTER_PX,
              "gate_frame_mm": GATE_FRAME_MM, "gate_map_mm": GATE_MAP_MM, "modes": list(MODES),
              "runs": {c: str(vessel_map_run(c)) for c in NAMES}}
    (root / "params.json").write_text(json.dumps(params, indent=2))
    (root / "vessels.json").write_text(json.dumps(summary, indent=1))

    n_frames = sum(len(v["frame_space"]["frames"]) for v in summary["configs"].values())
    n_maps = sum(len(v["top_view"]) for v in summary["configs"].values())
    (root / "README.md").write_text(
        f"# Centreline overlays, {stamp}\n\n"
        f"{n_frames} frame-space and {n_maps} top-view data points, each drawn in "
        f"{len(MODES)} variants\n({n_frames * len(MODES)} + {n_maps * len(MODES)} images), "
        "produced by\n"
        "`analysis/scripts/centreline_overlays.py` (see that file's docstring for the method and\n"
        "`reports/vessel-centreline-derivations.md` for what the poster's own metrics do).\n\n"
        "Magenta = ground-truth vessel centrelines, orange = predicted ones; a shared index\n"
        "means the two were matched, an index in only one colour is an unmatched vessel.\n"
        "Markers/pixels carry the usual confusion colours (TP green, FN red, FP blue, TN grey).\n"
        "In the frame-space figures the dashed magenta line, dashed grey line and the two stars\n"
        "are the geometry the published metric uses: one merged principal axis, its normal, and\n"
        "the merged ground-truth and predicted centroids.\n\n"
        "Layout:\n\n"
        "```\nframe_space/<Config>/<variant>/frame_<i>.png   one per vessel-bearing test frame\n"
        "top_view/<Config>/<variant>/<map>.png          one per test-set trial map\n"
        "contact_sheets/                                thumbnail grids for a quick scan\n"
        "params.json, vessels.json                      parameters, per-data-point counts\n```\n\n"
        "<variant> is one of centreline/ (centrelines only), centroid/ (centres of mass only)\n"
        "and both/. The lighter reference geometry follows the variant: the merged principal\n"
        "axis and its normal in the centreline variants, the two merged centroids (stars) in\n"
        "the centroid variants.\n")

    latest = Path(args.out_root) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(stamp)
    print(f"\nwritten to {root}  ({n_frames} frame overlays, {n_maps} map overlays)")


if __name__ == "__main__":
    main()
