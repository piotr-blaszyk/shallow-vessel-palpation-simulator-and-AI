#!/usr/bin/env python3
"""Draw the FINAL matched state of the detection/OSPA scheme, one image per data point.

This renders exactly what `scripts/detection_ospa.py` scores -- the same extraction (heatmap
head in video-frame space, connected components in top-view map space), the same near-duplicate
filtering, and the same Hungarian matching -- so nothing here is a re-derivation. Only the final
objects are drawn: no candidate peaks, no pre-merge fragments.

Reading an image
----------------
    magenta   ground-truth object            orange   predicted object
    solid line / filled dot, label "k"       matched pair, within the tolerance
    solid line / filled dot, label "k*"      matched pair, but FARTHER apart than the tolerance
                                             (OSPA charges the full cutoff c for these)
    dashed line / hollow dot, label "(k)"    UNMATCHED: a ground-truth object the model missed,
                                             or a predicted object with no counterpart
A magenta k and an orange k are the same pair. The title lists every pair's distance.

Markers (video frame) and pixels (top-view map) keep the project's confusion colours:
TP green, FN red, FP blue, TN grey.

Three variants of every data point:
    centreline/   the centreline segments, matched by centreline distance
    centroid/     the centroids, matched by centroid distance
    both/         the centreline objects with BOTH their segment and their centroid drawn
                  (matching is the centreline one, so the indices agree with centreline/)

Output tree (timestamped, so runs never overwrite each other):

    analysis/detections/<YYYYmmdd-HHMMSS>/
        README.md, params.json, objects.json
        frame_space/<Config>/<variant>/frame_<i>.png
        top_view/<Config>/<variant>/<map>.png
        contact_sheets/<space>_<Config>_<variant>.png

Usage:  python scripts/detection_overlays.py [--tau 5.0] [--limit N]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ANALYSIS  # noqa: E402
from detection_ospa import (ALPHA_FN, ALPHA_FP, C_MM, MERGE_ANG_DEG, MERGE_MM,  # noqa: E402
                            PRETTY, R_SPLIT_MM, THRESH, VALLEY_FRAC, evaluate,
                            frame_points, map_points)

GT_COLOUR = "#FF00FF"      # magenta: ground truth
PR_COLOUR = "#FF8000"      # orange:  prediction
CONF = {"tp": "#00C000", "fn": "#E00000", "fp": "#0050FF", "tn": "#BFBFBF"}
VARIANTS = ("centreline", "centroid", "both")
KIND_OF = {"centreline": "centreline", "both": "centreline", "centroid": "centroid"}
VAR_TITLE = {"centreline": "centrelines", "centroid": "centroids",
             "both": "centrelines + centroids"}


# --------------------------------------------------------------------------- labelling
def label_objects(pred, gt, matched):
    """Displayed index and match state for every object.

    Ground-truth objects are numbered 0, 1, 2 ... in spatial reading order -- across the common
    object direction, so the numbering runs down the image for the silicone sweeps and
    left-to-right for the meat straws. A matched prediction carries its partner's number;
    unmatched predictions are numbered after the ground-truth ones, which is what an index
    appearing in only one colour means.
    """
    axes = [o["axis"] for o in list(gt) + list(pred) if o["axis"] is not None]
    a = (np.linalg.eigh(sum(np.outer(x, x) for x in axes))[1][:, -1] if axes
         else np.array([1.0, 0.0]))
    nrm = np.array([-a[1], a[0]])
    nrm = nrm if nrm[1] >= 0 else -nrm

    def order(objs):
        return lambda i: (float(objs[i]["centroid"] @ nrm), float(objs[i]["centroid"] @ a))

    gt_lab = {j: k for k, j in enumerate(sorted(range(len(gt)), key=order(gt)))}
    pr_lab = {i: gt_lab[j] for i, j, _ in matched}
    for k, i in enumerate(sorted((i for i in range(len(pred)) if i not in pr_lab),
                                 key=order(pred))):
        pr_lab[i] = len(gt) + k
    return gt_lab, pr_lab


def states(n_objs, matched, side, tau):
    """Per object: "hit" (matched within tau), "far" (matched beyond tau) or "miss"."""
    out = {i: "miss" for i in range(n_objs)}
    for i, j, d in matched:
        out[i if side == "pred" else j] = "hit" if d <= tau else "far"
    return out


# --------------------------------------------------------------------------- drawing
STYLE = {"hit":  dict(ls="-", lw=2.6, dash=None, fill=True, fmt="{}"),
         "far":  dict(ls="-", lw=1.6, dash=None, fill=True, fmt="{}*"),
         "miss": dict(ls="--", lw=2.0, dash=(0, (5, 4)), fill=False, fmt="({})")}


def draw(ax, objs, labels, state, colour, variant, extend=0.0, dy=-10, ms=8):
    """One side's final objects, styled by match state and labelled with their index."""
    for i, o in enumerate(objs):
        st = STYLE[state[i]]
        tag = st["fmt"].format(labels[i])

        def put(point, off):
            ax.annotate(tag, point, textcoords="offset points", xytext=off, color=colour,
                        fontsize=12, fontweight="bold", zorder=9)

        if variant in ("centreline", "both") and o["ends"] is not None:
            a, b = o["ends"]
            d = b - a
            n = np.linalg.norm(d)
            if n > 0:
                a, b = a - d / n * extend, b + d / n * extend
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=st["lw"],
                    linestyle=st["dash"] or "-", zorder=6, solid_capstyle="round")
            put(b, (8 * np.sign(d[0] or 1), dy))
        if variant in ("centroid", "both") or o["ends"] is None:
            # filled = matched, hollow = unmatched; a thin dark rim keeps the fill readable
            ax.plot(*o["centroid"], marker="o", ms=ms + 1, zorder=7,
                    mfc=colour if st["fill"] else "none",
                    mec="k" if st["fill"] else colour, mew=1.0 if st["fill"] else 2.2)
            if variant == "centroid" or o["ends"] is None:
                put(o["centroid"], (8, -dy))


def legend(ax, variant):
    h = [plt.Line2D([], [], marker="o", ls="", color=CONF["tp"], label="TP"),
         plt.Line2D([], [], marker="o", ls="", color=CONF["fn"], label="FN (missed)"),
         plt.Line2D([], [], marker="o", ls="", color=CONF["fp"], label="FP (false alarm)"),
         plt.Line2D([], [], marker="o", ls="", color=CONF["tn"], label="TN"),
         plt.Line2D([], [], color=GT_COLOUR, lw=2.6, label="ground truth"),
         plt.Line2D([], [], color=PR_COLOUR, lw=2.6, label="prediction"),
         plt.Line2D([], [], color="0.25", lw=2.6, label='matched  "k"'),
         plt.Line2D([], [], color="0.25", lw=1.6, label='matched, over tolerance  "k*"'),
         plt.Line2D([], [], color="0.25", lw=2.0, ls=(0, (5, 4)),
                    label='UNMATCHED  "(k)"')]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=False, handlelength=2.0)


def pair_text(matched, gt_lab, tau, cap=6):
    """"pairs (mm): 0 1.8, 1 6.4*" -- the distance behind every matched index."""
    if not matched:
        return "no matched pairs"
    parts = [f"{gt_lab[j]} {d:.1f}" + ("*" if d > tau else "")
             for _, j, d in sorted(matched, key=lambda t: gt_lab[t[1]])]
    return "pairs (mm): " + ", ".join(parts[:cap]) + (" ..." if len(parts) > cap else "")


def title(cfg, name, variant, rec, matched, gt_lab, tau, c):
    head = (f"{PRETTY[cfg]}  {name}  [{VAR_TITLE[variant]}]   |   GT {rec['n_gt']}, "
            f"predicted {rec['n_pred']}, matched {len(matched)}   |   "
            f"TP {rec['tp']}  FP {rec['fp']}  FN {rec['fn']}  at {tau:g} mm")
    return head + "\n" + pair_text(matched, gt_lab, tau) + \
        f"     OSPA {rec['ospa']:.2f} mm  (c = {c:g} mm, FP x{ALPHA_FP:g}, FN x{ALPHA_FN:g})"


# --------------------------------------------------------------------------- per space
def render_frame(cfg, name, pred, gt, ctx, variant, tau, c, out):
    rec = evaluate(pred, gt, KIND_OF[variant], c=c, tau=tau)
    matched = rec["matched"]
    gt_lab, pr_lab = label_objects(pred, gt, matched)
    xy, gm, pm = ctx["xy"], ctx["gt_mask"], ctx["pred_mask"]

    fig, ax = plt.subplots(figsize=(6.6, 4.1), dpi=110)
    masks = {"tn": ~gm & ~pm, "tp": gm & pm, "fn": gm & ~pm, "fp": ~gm & pm}
    for key in ("tn", "tp", "fn", "fp"):
        m = masks[key]
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=42 if key != "tn" else 19, c=CONF[key],
                       edgecolors="none", zorder=3 if key != "tn" else 2)
    spacing = 2.0
    draw(ax, gt, gt_lab, states(len(gt), matched, "gt", tau), GT_COLOUR, variant,
         extend=0.35 * spacing)
    draw(ax, pred, pr_lab, states(len(pred), matched, "pred", tau), PR_COLOUR, variant,
         extend=0.35 * spacing, dy=12)
    ax.set_aspect("equal"); ax.margins(0.10); ax.invert_yaxis()
    ax.set_xlabel("sensor x (mm)"); ax.set_ylabel("sensor y (mm)")
    ax.set_title(title(cfg, name, variant, rec, matched, gt_lab, tau, c), fontsize=8.5)
    legend(ax, variant)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return rec


def render_map(cfg, name, pred, gt, ctx, variant, tau, c, out):
    rec = evaluate(pred, gt, KIND_OF[variant], c=c, tau=tau)
    matched = rec["matched"]
    gt_lab, pr_lab = label_objects(pred, gt, matched)
    gm, pm = ctx["gt_mask"], ctx["pred_mask"]

    rgb = np.zeros((*gm.shape, 3))
    rgb[...] = matplotlib.colors.to_rgb(CONF["tn"])
    for key, m in (("tp", gm & pm), ("fn", gm & ~pm), ("fp", ~gm & pm)):
        rgb[m] = matplotlib.colors.to_rgb(CONF[key])
    h, w = gm.shape
    fig, ax = plt.subplots(figsize=(max(5.2, w / 22), max(3.4, h / 22)), dpi=130)
    ax.imshow(rgb, interpolation="nearest")
    draw(ax, gt, gt_lab, states(len(gt), matched, "gt", tau), GT_COLOUR, variant, extend=2.0)
    draw(ax, pred, pr_lab, states(len(pred), matched, "pred", tau), PR_COLOUR, variant,
         extend=2.0, dy=12)
    ax.set_xlim(-0.06 * w, 1.06 * w); ax.set_ylim(1.08 * h, -0.08 * h)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title(cfg, name, variant, rec, matched, gt_lab, tau, c), fontsize=8.5)
    legend(ax, variant)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return rec


# --------------------------------------------------------------------------- contact sheets
def contact_sheet(src, dst, cols=8, thumb=260, cap=64):
    files = sorted(src.glob("*.png"))[:cap]
    if not files:
        return
    ims = [Image.open(f).convert("RGB") for f in files]
    ims = [im.resize((thumb, int(thumb * im.height / im.width))) for im in ims]
    rows = (len(ims) + cols - 1) // cols
    ch = max(im.height for im in ims)
    sheet = Image.new("RGB", (cols * thumb, rows * ch), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * thumb, (k // cols) * ch))
    sheet.save(dst)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(ANALYSIS / "overlays/detections"))
    ap.add_argument("--tau", type=float, default=C_MM,
                    help="clinical usefulness threshold in mm; c = tau (default %(default)s)")
    ap.add_argument("--limit", type=int, default=None, help="cap the data points per config")
    args = ap.parse_args()
    tau = c = args.tau

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out_root) / stamp
    (root / "contact_sheets").mkdir(parents=True, exist_ok=True)
    summary = {"timestamp": stamp, "tau_mm": tau, "configs": {}}

    for cfg, name in PRETTY.items():
        cname = name.replace("->", "-to-")
        summary["configs"][cname] = {}
        for space, gen, render in (("frame_space", frame_points, render_frame),
                                   ("top_view", map_points, render_map)):
            # extraction is per kind, not per variant: centreline/ and both/ share one
            cache = {}
            for variant in VARIANTS:
                kind = KIND_OF[variant]
                if kind not in cache:
                    cache[kind] = [(dp, pr, gt, ctx) for dp, pr, gt, ctx in gen(cfg, kind)]
                    if args.limit:
                        cache[kind] = cache[kind][:args.limit]
                out = root / space / cname / variant
                out.mkdir(parents=True, exist_ok=True)
                recs = [render(cfg, dp, pr, gt, ctx, variant, tau, c, out)
                        for dp, pr, gt, ctx in cache[kind]]
                summary["configs"][cname].setdefault(space, {})[variant] = {
                    "n": len(recs),
                    "n_unmatched_gt": sum(r["n_gt"] - len(r["matched"]) for r in recs),
                    "n_unmatched_pred": sum(r["n_pred"] - len(r["matched"]) for r in recs),
                    "n_pairs_over_tau": sum(sum(1 for _, _, d in r["matched"] if d > tau)
                                            for r in recs),
                    "ospa_mean_mm": float(np.mean([r["ospa"] for r in recs])) if recs else None}
                contact_sheet(out, root / "contact_sheets" / f"{space}_{cname}_{variant}.png",
                              cols=(2 if space == "top_view" else 8),
                              thumb=(700 if space == "top_view" else 260))
            print(f"{cname:18s} {space:11s} " + "  ".join(
                f"{v}:{summary['configs'][cname][space][v]['n']}" for v in VARIANTS))

    (root / "params.json").write_text(json.dumps(
        {"timestamp": stamp, "tau_mm": tau, "c_mm": c, "alpha_fp": ALPHA_FP, "alpha_fn": ALPHA_FN,
         "threshold": THRESH, "r_split_mm": R_SPLIT_MM, "valley_frac": VALLEY_FRAC,
         "merge_mm": MERGE_MM, "merge_ang_deg": MERGE_ANG_DEG,
         "heatmap_head": {"frame_space": True, "top_view": False}}, indent=2))
    (root / "objects.json").write_text(json.dumps(summary, indent=1))
    n_img = sum(1 for _ in root.rglob("*.png"))
    (root / "README.md").write_text(
        f"# Detection overlays, {stamp}\n\n"
        f"The FINAL matched state of the detection/OSPA scheme at a clinical usefulness\n"
        f"threshold of **{tau:g} mm**, drawn by `scripts/detection_overlays.py` from exactly the\n"
        "objects `scripts/detection_ospa.py` scores. See `reports/detection-ospa.md`.\n\n"
        "Magenta = ground truth, orange = prediction; a shared index is a matched pair.\n\n"
        "| style | meaning |\n|---|---|\n"
        '| solid, label `k` | matched pair, within the tolerance |\n'
        '| solid thin, label `k*` | matched pair, FARTHER apart than the tolerance |\n'
        '| dashed / hollow, label `(k)` | **unmatched** object |\n\n'
        "Markers and pixels carry the confusion colours (TP green, FN red, FP blue, TN grey).\n"
        "Each title lists every pair's distance and the data point's OSPA.\n\n"
        "```\nframe_space/<Config>/<variant>/frame_<i>.png\n"
        "top_view/<Config>/<variant>/<map>.png\n"
        "contact_sheets/\nparams.json, objects.json\n```\n\n"
        "`<variant>` is `centreline/`, `centroid/` or `both/` (the centreline objects with both\n"
        "their segment and their centroid drawn, matched as centrelines).\n")
    latest = Path(args.out_root) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(stamp)
    print(f"\nwritten to {root}  ({n_img} images)")


if __name__ == "__main__":
    main()
