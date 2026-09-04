#!/usr/bin/env python3
"""Recompute our results under metric definitions that the baseline papers actually use.

The manuscript's comparison table puts our one-directional mean predicted-pixel-to-truth distance
in the same column as a signed bias (Beasley), a class-value RMSE (Bewley) and a per-press centre
error (Hampson). None of those measure the same thing. This script derives, from the already-saved
per-marker test-set predictions (`analysis/results/frame_space_predictions_<config>.npz`, produced by
scripts/frame_space_predictions.py), the quantities each baseline reports, so a like-for-like row
can be written for our method:

  centre error      per frame, the distance between the centroid of the markers the model calls
                    vessel and the centroid of the truly-vessel markers. Reported as MAE +/- SD
                    (unsigned 2D distance), which is Yan & Pan's estimated-to-true tumour
                    centroid distance, and as the LATERAL component of the same offset, i.e. the
                    part perpendicular to the vessel, signed (Beasley's signed across-artery bias)
                    and unsigned (Hampson's absolute lateral error of the artery centre).
                    The vessel's own direction is the principal axis of the labelled markers of
                    that frame, so the lateral axis follows the vessel instead of assuming an
                    image axis: in Sim, Silicone the vessel runs along image x (a labelled band
                    spans ~21 mm in x and ~4-8 mm in y), so the signed x offset that this script
                    used to call "lateral" was in fact the ALONG-vessel one; on Meat the vessel
                    runs along y. The signed-x fields are kept for provenance.
  per-press acc.    per frame, the binary present/absent decision scored over vessel-present and
                    vessel-free frames alike -> MiniTac's balanced per-press accuracy.
  hit rate @ tau    fraction of vessel-bearing frames whose predicted centre lies within tau mm of
                    the true centre -> Chen et al.'s "within 50 px" accuracy, swept over tau.
  detection rate    fraction of vessel-bearing frames in which the model marks any marker vessel,
                    and the false-alarm rate on vessel-free frames -> MiniTac's and DIGIT Pinki's
                    per-press / per-clip binary accuracy.
  marker P/R/F1     per-marker precision, recall, F1 at threshold 0.5, and average precision,
                    which are threshold-free / class-balance-aware summaries of the same maps.
  truth->pred       the reverse of the manuscript's distance (each true vessel marker to the
                    nearest predicted one), which the reviewer asked to see beside it: the pair
                    bounds both over- and under-segmentation.

Geometry: markers are in 1920x1080 sensor pixels and 55 px = 2 mm (main.py::PX_TO_MM), so
1 px = 0.036364 mm. Threshold 0.5, as in the manuscript's Table 4.

Usage: comparable_metrics.py [--out analysis/results/comparable_metrics.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS  # noqa: E402

MM_PER_PX = 2.0 / 55.0
THRESH = 0.5
CONFIGS = {"A-to-A": "Sim->Sim", "A-to-B": "Sim->Silicone",
           "A-to-C": "Sim->Meat", "C-to-B": "Meat->Silicone"}


def centroid(pos, mask):
    """Centroid of the selected markers, or None when none are selected."""
    return pos[mask].mean(axis=0) if mask.any() else None


def lateral_normal(points):
    """Unit vector perpendicular to the vessel, from the principal axis of `points`.

    The sign is fixed towards the +y half-plane so that the per-frame signed lateral errors of
    one dataset can be averaged into a bias (as Beasley's are); which side "+" is remains
    arbitrary, exactly as it is in the paper. None when the label set is too small for a PCA.
    """
    if len(points) < 3:
        return None
    axis = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]
    n = np.array([-axis[1], axis[0]])
    return n if n[1] >= 0 else -n


def per_frame(npz):
    """Per-frame records: true/predicted centres, whether a vessel is present and detected."""
    probs, labels, pos = npz["probs"], npz["labels"].astype(bool), npz["pos"]
    pred = probs >= THRESH
    rows = []
    for i in range(len(probs)):
        t, p = centroid(pos[i], labels[i]), centroid(pos[i], pred[i])
        n = lateral_normal(pos[i][labels[i]]) if labels[i].any() else None
        both = t is not None and p is not None
        rows.append({
            "vessel_present": bool(labels[i].any()),
            "detected": bool(pred[i].any()),
            "err_mm": float(np.linalg.norm(t - p) * MM_PER_PX) if both else None,
            "signed_x_mm": float((p[0] - t[0]) * MM_PER_PX) if both else None,
            "lat_mm": float((p - t) @ n * MM_PER_PX) if both and n is not None else None,
        })
    return rows


def nn_distance(pos, src, dst):
    """Mean distance (mm) from every marker in `src` to the nearest marker in `dst`."""
    if not src.any() or not dst.any():
        return None
    d = np.linalg.norm(pos[src][:, None, :] - pos[dst][None, :, :], axis=2).min(axis=1)
    return float(d.mean() * MM_PER_PX)


def summarise(name, npz):
    probs, labels = npz["probs"], npz["labels"].astype(bool)
    pos, pred = npz["pos"], probs >= THRESH
    rows = per_frame(npz)

    present = [r for r in rows if r["vessel_present"]]
    absent = [r for r in rows if not r["vessel_present"]]
    errs = np.array([r["err_mm"] for r in present if r["err_mm"] is not None])
    signed = np.array([r["signed_x_mm"] for r in present if r["signed_x_mm"] is not None])
    lat = np.array([r["lat_mm"] for r in present if r["lat_mm"] is not None])

    # per-marker classification at 0.5, pooled over all frames
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum()); tn = int((~pred & ~labels).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    # average precision (threshold-free), pooled over markers
    order = np.argsort(-probs.ravel())
    y = labels.ravel()[order]
    ctp = np.cumsum(y); cfp = np.cumsum(~y)
    p_curve = ctp / np.maximum(ctp + cfp, 1)
    r_curve = ctp / max(int(y.sum()), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], r_curve])) * p_curve))

    # both directions of the nearest-neighbour distance, per frame then averaged
    fwd = [d for d in (nn_distance(pos[i], pred[i], labels[i]) for i in range(len(probs))) if d is not None]
    rev = [d for d in (nn_distance(pos[i], labels[i], pred[i]) for i in range(len(probs))) if d is not None]

    out = {
        "config": name,
        "n_frames": len(rows), "n_vessel_frames": len(present), "n_vessel_free_frames": len(absent),
        "centre_mae_mm": float(errs.mean()) if errs.size else None,
        "centre_sd_mm": float(errs.std(ddof=1)) if errs.size > 1 else None,
        "centre_median_mm": float(np.median(errs)) if errs.size else None,
        "centre_signed_x_mean_mm": float(signed.mean()) if signed.size else None,
        "centre_signed_x_sd_mm": float(signed.std(ddof=1)) if signed.size > 1 else None,
        # lateral = perpendicular to the vessel of that frame; the quantity Beasley (signed) and
        # Hampson (absolute) report, unlike the signed-x fields above (see the module docstring)
        "centre_lat_signed_mean_mm": float(lat.mean()) if lat.size else None,
        "centre_lat_signed_sd_mm": float(lat.std(ddof=1)) if lat.size > 1 else None,
        "centre_lat_abs_mean_mm": float(np.abs(lat).mean()) if lat.size else None,
        "centre_lat_abs_sd_mm": float(np.abs(lat).std(ddof=1)) if lat.size > 1 else None,
        "centre_lat_abs_median_mm": float(np.median(np.abs(lat))) if lat.size else None,
        "n_centre_frames": int(errs.size), "n_lat_frames": int(lat.size),
        "detection_rate": (sum(r["detected"] for r in present) / len(present)) if present else None,
        "false_alarm_rate": (sum(r["detected"] for r in absent) / len(absent)) if absent else None,
        # MiniTac's per-press accuracy: vessel-present frames detected + vessel-free frames not
        "per_press_accuracy": (sum(r["detected"] for r in present)
                               + sum(not r["detected"] for r in absent)) / len(rows),
        "marker_precision": prec, "marker_recall": rec,
        "marker_f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "marker_ap": ap,
        "pred_to_truth_mm": float(np.mean(fwd)) if fwd else None,
        "truth_to_pred_mm": float(np.mean(rev)) if rev else None,
        "hit_rate": {str(t): float((errs <= t).mean()) if errs.size else None
                     for t in (0.5, 1.0, 2.0, 3.0, 5.0)},
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS / "comparable_metrics.json"))
    args = ap.parse_args()

    res = {}
    for cfg, label in CONFIGS.items():
        npz = np.load(RESULTS / f"frame_space_predictions_{cfg}.npz")
        res[label] = summarise(label, npz)
        r = res[label]
        print(f"\n{label}  ({r['n_frames']} frames, {r['n_vessel_frames']} with a vessel, "
              f"{r['n_vessel_free_frames']} without)")
        print(f"  centre error   MAE {r['centre_mae_mm']:.2f} +/- {r['centre_sd_mm']:.2f} mm, "
              f"median {r['centre_median_mm']:.2f}  (n={r['n_centre_frames']})")
        print(f"  signed x mean  {r['centre_signed_x_mean_mm']:+.2f} +/- {r['centre_signed_x_sd_mm']:.2f} mm"
              "  (along the vessel in Sim/Silicone -- kept for provenance only)")
        print(f"  lateral        signed {r['centre_lat_signed_mean_mm']:+.2f} +/- {r['centre_lat_signed_sd_mm']:.2f} mm,"
              f" absolute {r['centre_lat_abs_mean_mm']:.2f} +/- {r['centre_lat_abs_sd_mm']:.2f} mm"
              f" (median {r['centre_lat_abs_median_mm']:.2f}, n={r['n_lat_frames']})")
        print(f"  detection rate {r['detection_rate']:.2f}"
              + (f", false alarms {r['false_alarm_rate']:.2f}" if r['false_alarm_rate'] is not None else "")
              + f", per-press accuracy {r['per_press_accuracy']:.2f}")
        print(f"  marker P/R/F1  {r['marker_precision']:.2f}/{r['marker_recall']:.2f}/{r['marker_f1']:.2f}, AP {r['marker_ap']:.2f}")
        print(f"  pred->truth {r['pred_to_truth_mm']:.2f} mm, truth->pred {r['truth_to_pred_mm']:.2f} mm")
        print("  hit rate  " + ", ".join(f"<={k}mm {v:.2f}" for k, v in r["hit_rate"].items()))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
