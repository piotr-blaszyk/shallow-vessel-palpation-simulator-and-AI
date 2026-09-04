#!/usr/bin/env python3
"""Evaluate vessel localisation as a DETECTION problem: count, match, then OSPA.

Why this exists
---------------
The poster's distance numbers are pixel-to-pixel or centroid-to-centroid means that say
nothing about how many vessels the model found. A frame or a map holds 0, 1, 2 or 3 vessels,
and a mean distance over matched things silently rewards a model that predicts only the easy
vessel and skips the hard ones. So this script scores the task the way object detection and
multi-target tracking are scored:

  1. extract objects from the prediction and from the ground truth;
  2. match them with the Hungarian algorithm;
  3. report the count confusion matrix, precision / recall / F1 at a tolerance tau, the mean
     distance over MATCHED pairs only, and a single penalised scalar, OSPA.

Two object types are evaluated, because "where is the vessel" has two natural answers:

  centreline   one straight segment per vessel (total-least-squares fit).  Distance between
               two centrelines = symmetric mean point-to-segment distance.
  centroid     one point per vessel.  Distance = plain Euclidean, in mm.

...and two spaces: the per-frame marker field, and the top-view map of a whole trial.

OSPA (Schuhmacher, Vo & Vo, 2008), with an asymmetric penalty
-------------------------------------------------------------
        d = [ SUM_matched min(d_i, c)  +  c * (a_FP * FP + a_FN * FN) ] / max(m, n)

with p = 1, so the result is an average in millimetres.  m = number of predicted objects,
n = number of ground-truth objects.  Because a_FP, a_FN >= 1, pairing is never dearer than
leaving a pair unmatched, so the optimal assignment matches min(m, n) objects and the
cardinality error is |m - n|, split into FP = max(0, m-n) and FN = max(0, n-m).

  c = tau = 5.0 mm    the distance beyond which a localisation is useless for the application.
                      Set to 1.0 mm on 2026-09-04 (the error margin for IV needle access) and
                      raised to 5.0 mm the same day by the user.  It is NOT tuned on the data --
                      it is a clinical judgement, and the script reports both cutoffs so the
                      choice can be seen rather than assumed.
  a_FP = 2, a_FN = 1  a false vessel costs twice a missed one (user, 2026-09-04).  A phantom
                      vessel sends a needle into tissue; a missed vessel only forfeits the
                      assist.
  both sets empty -> 0.  Bounded above by max(a_FP, a_FN) * c = 2 mm.

The matched-only mean distance and the gated precision / recall are reported BESIDE it, never
instead of it: OSPA alone cannot say whether an error was localisation or detection.

Extraction
----------
Frame space carries the model's per-marker PROBABILITIES (`probs` in the saved npz), so the
prediction side uses a heatmap head, as recommended: non-maximum suppression over the score
field picks peaks, every above-threshold marker joins its nearest peak, and each object's
centroid and axis are SCORE-WEIGHTED, which localises below the 2 mm marker pitch.  NMS alone
would shred an elongated vessel into a peak per bump, so peaks are only allowed to split a
connected component when they are more than R_SPLIT apart AND the score field dips between
them -- a watershed in spirit.  R_SPLIT = 3.0 mm sits below the smallest true vessel
separation in the data (3.5 mm) and above the marker pitch (1.8-2.0 mm).

Top-view map space has only the thresholded masks on disk -- no probability map is saved -- so
there the extraction is plain connected components plus an unweighted centroid and axis, and
NO heatmap head is used (user, 2026-09-04: only use it where the logits actually exist).

Near-duplicates are merged in both spaces so a fragmented mask cannot over-count: centroids
closer than MERGE_MM, or centrelines within MERGE_ANG of each other and offset by less than
MERGE_MM.

Usage:  python analysis/scripts/detection_ospa.py [--out analysis/results/detection_ospa.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PRETTY, RESULTS, vessel_map_run  # noqa: E402

MM_PER_PX = 2.0 / 55.0     # frame space: 55 sensor px = 2 mm
THRESH = 0.5               # frame-space operating point (manuscript Table 4)

# --- OSPA parameters, all fixed by the application rather than tuned ---------------------
C_MM = 5.0                 # cutoff = tolerance, the clinical usefulness threshold (user)
CUTOFFS = (5.0, 1.0)       # the headline value first; 1.0 mm (IV needle margin) kept for contrast
ALPHA_FP = 2.0             # a false vessel costs twice a missed one
ALPHA_FN = 1.0
P_NORM = 1                 # p = 1, so OSPA is an average in mm

# --- extraction parameters ---------------------------------------------------------------
EPS_SPACING_FACTOR = 1.45  # frame-space component radius, in marker pitches
R_SPLIT_MM = 3.0           # NMS radius / minimum separation two peaks must have to split a blob
VALLEY_FRAC = 0.80         # ...and the score between them must dip below this x the weaker peak
MAP_EPS_MM = 3.0           # map-space component radius
MAP_MIN_SAMPLES = 3
MAP_MIN_PX = 5             # ignore specks
MERGE_MM = 2.0             # merge objects this close (centroids, or parallel centrelines)
MERGE_ANG_DEG = 12.0
N_SAMPLES = 25             # points sampled along a segment for the centreline distance


# ------------------------------------------------------------------ object extraction
def fit_object(points, weights=None):
    """One object: score-weighted centroid, principal axis and segment endpoints."""
    w = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    c = (points * w[:, None]).sum(axis=0)
    d = points - c
    if len(points) < 2:
        return {"centroid": c, "axis": None, "ends": None, "n": len(points), "w": w.sum()}
    # weighted total least squares: principal eigenvector of the weighted scatter matrix
    cov = (d * w[:, None]).T @ d
    axis = np.linalg.eigh(cov)[1][:, -1]
    dom = 0 if abs(axis[0]) >= abs(axis[1]) else 1
    if axis[dom] < 0:
        axis = -axis
    t = d @ axis
    return {"centroid": c, "axis": axis, "ends": (c + t.min() * axis, c + t.max() * axis),
            "n": len(points)}


def nms_peaks(points, scores, radius):
    """Indices of local maxima of the score field, no two within `radius`. Strongest first."""
    order = np.argsort(-scores)
    tree = cKDTree(points)
    taken = np.zeros(len(points), bool)
    peaks = []
    for i in order:
        if taken[i]:
            continue
        peaks.append(i)
        taken[tree.query_ball_point(points[i], radius)] = True
    return peaks


def split_component(points, scores, radius, valley_frac):
    """Split one connected component into objects using the heatmap's peaks.

    A component is split only where the evidence really is bimodal: two peaks further apart
    than `radius` whose connecting path dips below `valley_frac` x the weaker peak. Otherwise
    an elongated vessel, whose score field has many small bumps along its length, would be
    shredded into one object per bump. Returns a list of index arrays into `points`.
    """
    peaks = nms_peaks(points, scores, radius)
    if len(peaks) < 2:
        return [np.arange(len(points))]
    keep = [peaks[0]]
    for q in peaks[1:]:
        for k in keep:
            # score along the straight path between the two peaks, sampled at the markers
            # nearest to it; a real valley means the two peaks are separate objects
            ts = np.linspace(0, 1, 12)[:, None]
            path = points[k] * (1 - ts) + points[q] * ts
            near = cKDTree(points).query(path)[1]
            valley = scores[near].min()
            if valley >= valley_frac * min(scores[k], scores[q]):
                break                     # no dip: same object
        else:
            keep.append(q)
    if len(keep) < 2:
        return [np.arange(len(points))]
    owner = np.argmin(np.linalg.norm(points[:, None] - points[keep][None], axis=2), axis=1)
    return [np.flatnonzero(owner == j) for j in range(len(keep))
            if (owner == j).sum() > 0]


def merge_duplicates(objs, kind):
    """Fold objects that are really one vessel seen twice (a fragmented mask over-counts).

    `objs` arrives largest-first, so the survivor of a merge is always the bigger object.
    """
    kept = []
    for o in objs:
        dup = False
        for m in kept:
            if np.linalg.norm(o["centroid"] - m["centroid"]) < MERGE_MM:
                dup = True
            elif kind == "centreline" and o["ends"] is not None and m["ends"] is not None:
                ang = np.degrees(np.arccos(min(1.0, abs(float(o["axis"] @ m["axis"])))))
                nrm = np.array([-m["axis"][1], m["axis"][0]])
                off = abs(float((o["centroid"] - m["centroid"]) @ nrm))
                # ...and the two segments must actually overlap along the shared axis. Without
                # this, two collinear but DISJOINT vessels -- the silicone phantom's left- and
                # right-hand sweeps of the same row -- are fused into one object.
                to, tm = [sorted(float(e @ m["axis"]) for e in x["ends"]) for x in (o, m)]
                gap = max(to[0], tm[0]) - min(to[1], tm[1])
                dup = ang < MERGE_ANG_DEG and off < MERGE_MM and gap < MERGE_MM
            if dup:
                break
        if not dup:
            kept.append(o)
    return kept


def objects_from_field(points, scores, mask, eps, use_heatmap, kind):
    """All objects of one data point, from a score field and its thresholded mask."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    lab = DBSCAN(eps=eps, min_samples=1).fit_predict(points[idx])
    objs = []
    for k in sorted(set(lab)):
        comp = idx[lab == k]
        parts = (split_component(points[comp], scores[comp], R_SPLIT_MM, VALLEY_FRAC)
                 if use_heatmap else [np.arange(len(comp))])
        for part in parts:
            sel = comp[part]
            objs.append(fit_object(points[sel], scores[sel] if use_heatmap else None))
    return merge_duplicates(sorted(objs, key=lambda o: -o["n"]), kind)


# ------------------------------------------------------------------ distances
def seg_points(o, n=N_SAMPLES):
    if o["ends"] is None:
        return o["centroid"][None, :]
    a, b = o["ends"]
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def point_to_segment(p, a, b):
    ab = b - a
    L2 = ab @ ab
    t = 0.0 if L2 == 0 else np.clip((p - a) @ ab / L2, 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def dist_centreline(o1, o2):
    """Symmetric mean point-to-segment distance between two centrelines (mm)."""
    def one_way(x, y):
        a, b = (y["ends"] if y["ends"] is not None else (y["centroid"], y["centroid"]))
        return np.mean([point_to_segment(p, a, b) for p in seg_points(x)])
    return 0.5 * (one_way(o1, o2) + one_way(o2, o1))


def dist_centroid(o1, o2):
    return float(np.linalg.norm(o1["centroid"] - o2["centroid"]))


# ------------------------------------------------------------------ matching and OSPA
def evaluate(pred, gt, kind, c=C_MM, tau=C_MM, a_fp=ALPHA_FP, a_fn=ALPHA_FN):
    """Match one data point's objects and return its OSPA plus the diagnostic parts."""
    dist = dist_centreline if kind == "centreline" else dist_centroid
    m, n = len(pred), len(gt)
    rec = {"n_pred": m, "n_gt": n, "matched": [], "ospa": None,
           "tp": 0, "fp": m, "fn": n}
    if m == 0 and n == 0:
        rec["ospa"] = 0.0
        return rec
    if m == 0 or n == 0:
        rec["ospa"] = (c * (a_fp * m + a_fn * n)) / max(m, n)
        return rec

    D = np.array([[dist(p, g) for g in gt] for p in pred])
    # Pairing costs at most c, and leaving an object unmatched costs at least c, so the optimal
    # assignment always matches min(m, n) pairs: a rectangular Hungarian solve is enough.
    ri, ci = linear_sum_assignment(np.minimum(D, c))
    matched = [(int(i), int(j), float(D[i, j])) for i, j in zip(ri, ci)]
    fp, fn = m - len(matched), n - len(matched)
    num = sum(min(d, c) ** P_NORM for _, _, d in matched) + \
        c ** P_NORM * (a_fp * fp + a_fn * fn)
    rec["ospa"] = float((num / max(m, n)) ** (1.0 / P_NORM))
    rec["matched"] = matched
    # Gated detection view, reported separately: a pair beyond tau is not a detection.
    tp = sum(1 for _, _, d in matched if d <= tau)
    rec.update(tp=tp, fp=m - tp, fn=n - tp,
               matched_d=[d for _, _, d in matched],
               hit_d=[d for _, _, d in matched if d <= tau])
    return rec


# ------------------------------------------------------------------ data points
def frame_points(config, kind):
    """Every vessel-bearing test frame: predicted objects (heatmap head) and true objects."""
    npz = np.load(RESULTS / f"frame_space_predictions_{config}.npz")
    probs, labels, pos = npz["probs"], npz["labels"].astype(bool), npz["pos"].astype(float)
    pos = pos * MM_PER_PX
    spacing = float(np.median(cKDTree(pos[0]).query(pos[0], k=2)[0][:, 1]))
    eps = EPS_SPACING_FACTOR * spacing
    for i in range(len(probs)):
        xy, sc = pos[i], probs[i].astype(float)
        gt = objects_from_field(xy, np.ones(len(xy)), labels[i], eps, False, kind)
        pr = objects_from_field(xy, sc, sc >= THRESH, eps, True, kind)
        yield f"frame_{i:04d}", pr, gt, {"xy": xy, "scores": sc,
                                         "gt_mask": labels[i], "pred_mask": sc >= THRESH}


def map_points(config, kind):
    """Every top-view trial map: binary masks only, so no heatmap head."""
    run = vessel_map_run(config)
    dirs = sorted(p for p in run.iterdir() if p.is_dir()) or [run]
    for d in dirs:
        out, masks = {}, {}
        for key, f in (("gt", "ground_truth.png"), ("pr", "prediction.png")):
            mask = np.asarray(Image.open(d / f).convert("L")) > 127
            xy = np.column_stack(np.nonzero(mask))[:, ::-1].astype(float)  # (x, y), 1 px = 1 mm
            if len(xy) == 0:
                out[key] = []
                masks[key] = mask
                continue
            lab = DBSCAN(eps=MAP_EPS_MM, min_samples=MAP_MIN_SAMPLES).fit_predict(xy)
            objs = [fit_object(xy[lab == k]) for k in sorted(set(lab))
                    if k != -1 and (lab == k).sum() >= MAP_MIN_PX]
            out[key] = merge_duplicates(sorted(objs, key=lambda o: -o["n"]), kind)
            masks[key] = mask
        yield (d.name if d != run else "whole_map"), out["pr"], out["gt"], \
            {"gt_mask": masks["gt"], "pred_mask": masks["pr"]}


# ------------------------------------------------------------------ aggregation
def summarise(records):
    """Pool one config/space/kind: counts, detection rates, matched-only distance, OSPA."""
    n_gt = np.array([r["n_gt"] for r in records])
    n_pr = np.array([r["n_pred"] for r in records])
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    matched = np.array([d for r in records for d in r.get("matched_d", [])])
    hits = np.array([d for r in records for d in r.get("hit_d", [])])
    ospa = np.array([r["ospa"] for r in records])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    conf = np.zeros((5, 5), int)          # count confusion over {0,1,2,3,4+}
    for g, p in zip(n_gt, n_pr):
        conf[min(g, 4), min(p, 4)] += 1
    return {
        "n_points": len(records),
        "count_accuracy": float((n_gt == n_pr).mean()),
        "count_mae": float(np.abs(n_gt.astype(int) - n_pr.astype(int)).mean()),
        "count_confusion_gt_by_pred": conf.tolist(),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec,
        "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "matched_mean_mm": float(matched.mean()) if matched.size else None,
        "matched_sd_mm": float(matched.std(ddof=1)) if matched.size > 1 else None,
        "matched_n": int(matched.size),
        "hit_mean_mm": float(hits.mean()) if hits.size else None,
        "hit_n": int(hits.size),
        # Split out the vessel-free data points: they score OSPA 0 whenever the model also
        # says "nothing", and there are 379 of them in Sim->Sim, which flatters the pooled mean.
        "ospa_mean_mm": float(ospa.mean()),
        "ospa_mean_vessel_present_mm": float(ospa[n_gt > 0].mean()) if (n_gt > 0).any() else None,
        "n_vessel_present": int((n_gt > 0).sum()),
        "ospa_mean_vessel_free_mm": float(ospa[n_gt == 0].mean()) if (n_gt == 0).any() else None,
        "n_vessel_free": int((n_gt == 0).sum()),
        "ospa_sd_mm": float(ospa.std(ddof=1)) if ospa.size > 1 else None,
        "ospa_median_mm": float(np.median(ospa)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "detection_ospa.json"))
    ap.add_argument("--cutoffs", type=float, nargs="+", default=list(CUTOFFS),
                    help="clinical usefulness thresholds in mm; c = tau for each (default 5 1)")
    args = ap.parse_args()

    res = {"params": {"cutoffs_mm": args.cutoffs, "headline_cutoff_mm": args.cutoffs[0],
                      "alpha_fp": ALPHA_FP, "alpha_fn": ALPHA_FN, "p": P_NORM,
                      "threshold": THRESH, "r_split_mm": R_SPLIT_MM,
                      "valley_frac": VALLEY_FRAC, "merge_mm": MERGE_MM,
                      "merge_ang_deg": MERGE_ANG_DEG,
                      "heatmap_head": {"frame_space": True, "top_view": False}},
           "cutoffs": {}}
    for c in args.cutoffs:
        res["cutoffs"][f"{c:g}"] = {}
    for cfg, name in PRETTY.items():
        for kind in ("centreline", "centroid"):
            for space, gen in (("frame", frame_points), ("top_view", map_points)):
                # extraction does not depend on the cutoff, so do it once
                points = [(pr, gt) for _, pr, gt, _ in gen(cfg, kind)]
                for c in args.cutoffs:
                    s = summarise([evaluate(pr, gt, kind, c=c, tau=c) for pr, gt in points])
                    res["cutoffs"][f"{c:g}"].setdefault(name, {}).setdefault(kind, {})[space] = s
                    if c == args.cutoffs[0]:
                        print(f"{name:15s} {kind:10s} {space:8s} c={c:g}mm n={s['n_points']:4d}  "
                              f"count acc {s['count_accuracy']:.2f}  P/R/F1 "
                              f"{s['precision']:.2f}/{s['recall']:.2f}/{s['f1']:.2f}  "
                              f"OSPA {s['ospa_mean_mm']:.2f} mm")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
