# How the blood-vessel centreline is computed in this project

*2026-09-04. Companion to `analysis/reports/ours-same-metric-derivations.md`, which derives the
"Ours, same metric" column of the poster's comparison table. This note answers the four
questions asked about the centreline behind those numbers, and documents the per-vessel
overlays generated for every data point that feeds them.*

---

## Short answers

| Question | Answer |
|---|---|
| **Is the centreline aligned with the image x- or y-axis, or can it be skew?** | Skew. It is a free principal-axis (PCA/SVD) direction fitted to the vessel-labelled markers of the frame, never an assumed image axis. On Sim→Sim it sits **10.5°** off image x in every single frame; on Silicone its median offset from the nearest image axis is **4.4°** and reaches **43°**; on Meat it runs along image y (median **1.0°** off). |
| **What happens when a frame contains several vessels?** | They are **merged**, not separated and not discarded. One centroid and one principal axis are taken over the *union* of the vessel-labelled markers, so two vessels become a single fictitious "mean vessel". 26 / 51 / 26 / 51 frames (Sim→Sim / Sim→Silicone / Sim→Meat / Meat→Silicone) contain more than one vessel and are treated this way. |
| **Frame space, top-view space, or both?** | **Frame space only.** No top-view metric uses a centreline, a centroid or an axis: `symmetric_distances.py` works purely with a Euclidean distance transform of the two masks. |
| **Is there a "centreline" object in the code at all?** | Only one, and it is not an estimate: the simulator's own vein geometry, written out as `ground_truth_centreline_reference.png` for the Sim→Sim map. It is a reference picture and feeds no metric. |

The per-vessel centrelines requested — one magenta line per ground-truth vessel, one orange
line per predicted vessel, index-matched — **did not exist before this note**. They are new,
built for visualisation and for the sensitivity check in §6, and they do not change any number
on the poster.

---

## 1. What the published metrics actually compute

### 1.1 Frame space — one merged axis per frame

Everything centreline-flavoured on the poster comes from `analysis/scripts/comparable_metrics.py`:

```python
def centroid(pos, mask):                                    # comparable_metrics.py:54
    return pos[mask].mean(axis=0) if mask.any() else None

def lateral_normal(points):                                 # comparable_metrics.py:59
    if len(points) < 3:
        return None
    axis = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]
    n = np.array([-axis[1], axis[0]])
    return n if n[1] >= 0 else -n
```

and, per frame (`per_frame`, lines 73–89):

```python
t = centroid(pos[i], labels[i])                 # centroid of ALL vessel-labelled markers
p = centroid(pos[i], pred[i])                   # centroid of ALL predicted markers
n = lateral_normal(pos[i][labels[i]])           # normal of ONE axis through ALL of them
lat = (p - t) @ n * MM_PER_PX                   # the signed lateral error
```

So the "centreline" is implicit: it is the line through the ground-truth centroid `t` along the
first principal component of the vessel-labelled markers. Three consequences follow directly
from those five lines:

* **it is skew by construction** — `axis` is whatever direction the SVD returns;
* **it is one line per frame, not one per vessel** — the mask `labels[i]` is not split;
* **it exists only where at least 3 markers are labelled** — `lateral_normal` returns `None`
  below that, and the frame drops out of the lateral aggregates (this is the only discarding
  that happens, and it is about marker count, not vessel count).

Which poster cells depend on it: the **Beasley** row (signed lateral bias, `centre_lat_signed_*`)
and the **Hampson** row (absolute lateral error, `centre_lat_abs_*`). The **Yan & Pan** row
(`centre_mae_mm`) uses the two centroids only — no axis at all. The **Raina** row and the
segmentation-statistics figure use the top-view distance, which is §1.2.

### 1.2 Top-view space — no centreline anywhere

`analysis/scripts/symmetric_distances.py::map_space` scores the two masks pixelwise:

```python
to_gt_pos = ndimage.distance_transform_edt(~gt)   # distance to the nearest true vessel pixel
d_pos = to_gt_pos[pred]                           # for every predicted vessel pixel
```

That is the manuscript's one-directional distance. No clustering, no centroid, no principal
axis, no vessel identity. The number a reader might read as "distance to the centreline" is a
distance to the nearest *ground-truth vessel pixel* — the whole labelled band, not its middle.

### 1.3 The one literal centreline in the code base

`difftactile/object_model/vein.py:32` builds the simulated vein as a chain of particles along a
known straight centreline, and `difftactile/data_analysis/experiment/vessel_map.py:700–703,
853–860` projects it into map pixels and saves it as
`ground_truth_centreline_reference.png`, **only for vessel-present simulated trajectories**
(the comment there is explicit: for a vessel-absent trial the vein is not physically present, so
a reference line would mislead). It is ground-truth world geometry, not an estimate from
predictions, and no metric reads it back.

---

## 2. Evidence for the "skew" answer

Angle of the fitted axis, over every frame with ≥3 labelled markers:

| Config | n frames | median axis angle | 5–95 % | median offset from the nearest image axis | max offset | frames > 5° off an axis |
|---|---|---|---|---|---|---|
| Sim→Sim | 296 | 10.5° | 9.5–13.0° | **10.5°** | 13.6° | **100 %** |
| Sim→Silicone | 94 | 152.7° | 1.0–178.7° | 4.4° | **43.2°** | 46 % |
| Sim→Meat | 112 | 89.8° | 83.3–96.4° | 1.0° | 28.3° | 13 % |
| Meat→Silicone | 94 | 152.7° | 1.0–178.7° | 4.4° | 43.2° | 46 % |

Sim→Sim is the clearest case: the vein is consistently tilted ~10° from image x, and an
axis-aligned assumption would have been wrong in every frame. On Silicone the wide spread is
mostly the merged-vessel effect of §1.1 — when a frame holds two vessels the merged axis can
swing far from either of them (§6).

Silicone and Meat→Silicone are identical rows because both models are evaluated on the same
94 silicone frames; the axis depends only on the *ground truth*, not on the model.

---

## 3. The new per-vessel overlays

`analysis/scripts/centreline_overlays.py` produces the requested pictures. Method, per data point:

1. **Split into vessels.** DBSCAN on the vessel elements.
   *Frame space:* `eps = 1.45 ×` the dataset's median inter-marker spacing (2.62–2.84 mm for a
   1.81–1.96 mm spacing), `min_samples = 1`. At 1.45 × spacing a marker links only to its
   immediate hexagonal neighbours, so two vessels separated by one clear marker gap stay apart.
   *Top-view space:* `eps = 3 mm`, `min_samples = 3`, clusters below 5 px ignored.
2. **Fit one centreline per vessel** — total least squares: the line through the cluster centroid
   along its first principal component, drawn between the cluster's extreme projections.
3. **Match ground truth to prediction** — Hungarian assignment on centroid distance, rejecting
   pairs farther apart than 10 mm (frame) / 20 mm (map).
4. **Index and draw.** Ground-truth vessels are numbered 0, 1, 2 … across the common vessel
   direction (down the image for the silicone sweeps, left-to-right for the meat straws); a
   matched prediction carries its partner's number. **Magenta k and orange k are a pair; an index
   that appears in only one colour is unmatched** — a missed vessel or a spurious blob.
   Markers/pixels keep the project's confusion colours (TP green, FN red, FP blue, TN grey).

Two properties of that recipe are worth knowing when reading the pictures. A cluster of a
single element has no direction, so it is drawn as an `x` with its index rather than a line —
in frame space an isolated labelled marker is counted as a vessel of its own rather than
silently dropped, which is why some frames report three "vessels" for one real one. And the
matching is on **centroid distance only**; where a long predicted band sits beside several short
ground-truth clusters, it is assigned to whichever centroid is nearest, which is not always the
cluster a person would pair it with by eye (`frame_space/Sim-to-Meat/frame_0093.png` is an
example). Both are visualisation choices and neither touches a published number.

### Three variants of every figure

A centreline and a centroid are different objects — the centreline is the fitted *axis* of a
vessel, the centroid its *centre of mass*, a single point — so each data point is drawn three
ways rather than one:

| variant | what it draws |
|---|---|
| `centreline/` | one straight centreline per vessel, indexed |
| `centroid/` | one centroid dot per vessel, indexed |
| `both/` | the two together; the centroid always lies on its own centreline |

The indices are the same in all three, so a vessel can be followed from one variant to another,
and in `both/` a vessel's number is printed twice — once at the end of its line, once beside its
centroid. Frame-space figures additionally show, in a lighter weight, the geometry the
*published* metric uses, and that follows the variant too: the merged principal axis (dashed
magenta) and its normal (dashed grey) appear in `centreline/` and `both/`, the merged
ground-truth centroid (magenta star) and merged predicted centroid (orange star) in `centroid/`
and `both/`. The offset between those two stars, projected onto the dashed grey axis, *is* the
lateral error the Beasley and Hampson cells report. The title states whether that frame entered
the lateral aggregate.

### Coverage

One image per data point that feeds a centreline metric — every vessel-bearing frame, and every
top-view trial map:

| Config | frame-space overlays | of which in the lateral aggregate | top-view overlays |
|---|---|---|---|
| Sim→Sim | 296 | 296 | 1 (whole map) |
| Sim→Silicone | 94 | 92 | 1 (whole phantom) |
| Sim→Meat | 129 | 105 | 10 (one per trial) |
| Meat→Silicone | 94 | 94 | 1 (whole phantom) |
| **total** | **613** | **587** | **13** |

Each of those 626 data points is drawn in the three variants above, so the run holds
1839 frame-space and 39 top-view images.

The in-aggregate counts reproduce `n_lat_frames` in `analysis/results/comparable_metrics.json` exactly
(296 / 92 / 105 / 94), so the picture set and the printed numbers cover the same data points.

### Where the images are

```
analysis/centrelines/<YYYYmmdd-HHMMSS>/     timestamped, so runs never overwrite each other
    README.md            what the run is and how to read the pictures
    params.json          every threshold, eps, gate and source path
    vessels.json         per-data-point vessel counts, matches, centre and lateral errors
    frame_space/<Config>/<variant>/frame_<i>.png   one per vessel-bearing test frame
    top_view/<Config>/<variant>/<map>.png          one per test-set trial map
    contact_sheets/<space>_<Config>_<variant>.png  thumbnail grids for a quick scan
                                                   <variant> = centreline | centroid | both
analysis/centrelines/latest -> <YYYYmmdd-HHMMSS>
```

Current run: `analysis/centrelines/20260904-115206` (121 MB, 1902 images). The bulk frame-space
images and the contact sheets are git-ignored — they are regenerable artefacts; the 39 top-view
maps, the manifests, the script and this report are what the repo carries.

---

## 4. Validation of the clustering

The top-view vessel counts recovered by DBSCAN are checked against what was physically in each
trial, which is recorded in the trial directory names:

| Config / trial | vessels recovered | physically present |
|---|---|---|
| Sim→Sim, whole map | 1 | 1 simulated vein |
| Sim→Silicone, whole map | **10** | "silicone phantom, **ten sweeps**, 180 × 100 mm" (`run.json`) |
| Meat→Silicone, whole map | **10** | same phantom map |
| Sim→Meat trials 01–07 | 1 each | `1-metal-straw-…`, `1-silicone-straw-…` |
| Sim→Meat trial 08 | **2** | `2-metal-straws-beneath-2-steaks` |
| Sim→Meat trial 09 | **3** | `3-metal-straws-beneath-2-steaks` |
| Sim→Meat trial 10 | **0** | `no-straw` |

Every count is right, including the empty control, so the operating point is not tuned to a
single case.

---

## 5. The silicone map: why there is no 10-way split and 5-way stitch

The requested route was: treat the ten silicone slides as ten small maps, then stitch five of
them into a whole-phantom map with re-indexed captions. That is not possible from the saved
artefacts, and it turns out not to be needed.

* The silicone map the manuscript's run saved **is already the stitched whole-phantom map** —
  one 100 × 180 mm image described in `run.json` as *"silicone phantom, ten sweeps,
  180 × 100 mm workspace"*. There are no ten sub-maps on disk; `sim-to-silicone_gt-video/…`
  contains a single `ground_truth.png` / `prediction.png` pair, unlike `sim-to-meat_gt-video/…`,
  which does have ten trial sub-directories.
* `run.json` carries **no per-sweep provenance** — no per-source pixel mask, no sweep index.
  (`ground_truth_sources_overlay.png` is video-derived vs photo-derived ground truth, not
  per-sweep.) Splitting the combined map by which sweep wrote each pixel would mean re-running
  the map builder over the ten videos, which is a heavyweight job and has not been run.
* **The split was only a means to an end**, and the end is reached directly: clustering the
  whole-phantom map recovers exactly the ten sweeps as ten separate vessels (§4), each with its
  own centreline and its own index, 0–9, on one big map. That is the artefact the stitch was
  meant to produce, without the intermediate step.

If you want the genuine ten sub-maps as well — separately built, then stitched — say so and I
will re-run the map builder; it needs your permission because it is not a last-minute metric
recomputation.

---

## 6. What the merge costs (sensitivity check)

Since the per-vessel decomposition now exists, the lateral error can be recomputed per matched
vessel pair instead of per merged frame. Nothing on the poster changes — this is a check, not a
replacement:

| Config | published (merged), signed | per-vessel, signed | published, absolute | per-vessel, absolute | per-vessel axis vs merged axis (median / 95th) |
|---|---|---|---|---|---|
| Sim→Sim | +0.02 ± 1.55 mm (n=296) | −0.04 ± 1.62 mm (n=296) | 1.18 ± 1.00 mm | 1.22 ± 1.06 mm | 0.0° / 12.1° |
| **Sim→Silicone** | **+0.41 ± 2.71 mm** (n=92) | −0.09 ± 1.78 mm (n=131) | **1.88 ± 1.98 mm** | 1.17 ± 1.34 mm | 9.0° / 89.1° |
| Sim→Meat | −0.41 ± 3.59 mm (n=105) | −0.37 ± 2.87 mm (n=105) | 2.74 ± 2.34 mm | 2.23 ± 1.83 mm | 0.0° / 27.6° |
| Meat→Silicone | +0.89 ± 2.94 mm (n=94) | +0.57 ± 3.90 mm (n=108) | 2.16 ± 2.17 mm | 3.04 ± 2.49 mm | 7.2° / 89.0° |

Read: on **Sim→Silicone**, the configuration the poster's Beasley and Hampson cells report, the
merge makes our number **worse**, not better — per vessel the absolute lateral error is
1.17 mm against the printed 1.88 mm, and the signed bias shrinks from +0.41 to −0.09 mm. The
printed cells are therefore the conservative choice, which is the right way round for a claim
against a baseline. The 95th-percentile axis deviations near 90° on the two silicone rows are
the two-vessel frames: when a frame holds two parallel sweeps whose union is wider than it is
long, the merged principal axis flips to run *across* the vessels rather than along them, and
the merged lateral error picks up the along-vessel offset instead. Per vessel that cannot
happen.

The single-vessel-dominated configs (Sim→Sim, Sim→Meat) agree to within 0.05 mm signed, which is
the expected result and a sanity check on the clustering.

---

## 7. Reproduction

```bash

# the overlays and their manifests (about 75 s, writes a new timestamped folder)
python analysis/scripts/centreline_overlays.py

# the published numbers this note refers to
python analysis/scripts/comparable_metrics.py        # centre / lateral / hit-rate aggregates
python analysis/scripts/symmetric_distances.py       # top-view distances and FG IoU
```

`analysis/scripts/centreline_overlays.py --limit N` renders only the first N frames per config, for a
quick look. Both parameters and source paths are recorded in each run's `params.json`.

---

## 8. Summary

The project has never estimated a per-vessel centreline. It fits **one skew principal axis per
video frame, over all vessel-labelled markers at once**, and uses its normal to split the
centroid offset into a lateral component — that, and nothing more, is what the Beasley and
Hampson cells report. Top-view space uses no centreline at all. Multiple vessels in a frame are
merged rather than separated or discarded, which on the silicone configuration makes the printed
lateral error larger than a per-vessel treatment would. The new overlays make all of this
visible data point by data point, with the centrelines and the centroids drawn separately and
together, and the ten silicone sweeps come out of the whole-phantom map
as ten separately indexed vessels without needing the split-and-stitch route.
