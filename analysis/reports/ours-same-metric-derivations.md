# How every "Ours, same metric" cell of the poster's related-work table was derived

Written 2026-09-04. Companion to `poster.tex` (third column, "Related work"), to
`related-work-table.tex` (the fit harness holding the same block) and to
`comparison-review/review.pdf` (the review that judged whether each cell really matches its row).

The column exists because manuscript Table 2 put our one-directional map distance in the same column
as a signed bias, a class-value RMSE and a per-press centre error, none of which measure the same
thing. Each cell here is **our own predictions re-scored under the definition the baseline in that
row uses**, so that the two numbers beside each other are the same quantity.

## Where the numbers come from

Nothing in this column comes from a new training run. Two saved artefacts feed everything:

| artefact | produced by | what it holds |
|---|---|---|
| `analysis/results/frame_space_predictions_<config>.npz` | `analysis/scripts/frame_space_predictions.py` | per-marker sigmoid probabilities, ground-truth labels and 1920x1080 pixel positions for every scored (central) video frame of each test set, from the best-of-five model instance of each configuration |
| `analysis/results/comparable_metrics.json` | `analysis/scripts/comparable_metrics.py` | every frame-space quantity below, derived from the `.npz` files |
| `analysis/results/symmetric_distances.json` | `analysis/scripts/symmetric_distances.py` | map-space distances and foreground IoU, from the `prediction.png` / `ground_truth.png` of the manuscript's own vessel-map runs |

Conventions used throughout, identical to the manuscript's Table 4:

* **Threshold 0.5** on the per-marker sigmoid, in frame space.
* **Geometry** 55 px = 2 mm (`main.py::PX_TO_MM`), so 1 px = 0.036364 mm; map space is 1 px = 1 mm.
* **Scoring unit** in frame space is one central video frame of a five-frame clip. A *frame* is what
  a baseline would call a *press* or a *clip*; frames inside one trial are not independent, so the
  effective n is smaller than the frame count. Counts: 675 / 120 / 219 / 120 frames for
  Sim->Sim / Sim->Silicone / Sim->Meat / Meat->Silicone, of which 296 / 94 / 129 / 94 contain a
  vessel.
* **Centroid statistics** drop frames in which the model predicts nothing (no centroid exists):
  n = 296 / 92 / 113 / 94, and the lateral ones additionally drop frames whose labelled markers have
  no well-defined principal axis (n = 296 / 92 / 105 / 94).
* Every cell names the transfer it comes from. Where a cell quotes one transfer only it is
  **Sim->Silicone**, which `comparison-review/review.pdf` Sec. 4 checks is the worst or second-worst
  of the four on that metric, i.e. not a cherry-pick.

## Cell by cell

### MiniTac [1] --- "present / absent, per press" --- ours: 100 % Sim->Sim, 93 % Sim->Silicone

*Their definition.* Per-press binary accuracy of a linear SVM over a test split of roughly 56
presses with the two classes deliberately balanced.

*Our derivation.* `per_press_accuracy` in `comparable_metrics.json`. Per frame, a vessel-present
frame counts as correct if **any** marker passes 0.5, and a vessel-free frame counts as correct if
**none** does; the two are pooled over all frames of the test set:

```
per_press_accuracy = (#present frames with a detection + #vessel-free frames with none) / #frames
Sim->Sim        675/675 = 1.000  -> 100 %
Sim->Silicone   112/120 = 0.933  ->  93 %
```

*Caveat to state at the poster.* Our classes are unbalanced the other way (94 present against 26
vessel-free on Silicone), so the 93 % is dominated by the present class, and the 100 % is an
**in-domain** test.

### DIGIT Pinki [2] --- "hardness class, per clip; silicone, phantom" --- ours: 78 % Meat->Silicone

*Their definition.* Per-clip accuracy of a VideoMAE head over 11 hardness classes (silicone
swatches, 98 %) and 6 classes (phantom prostate, 100 %), on splits taken inside each palpation
video.

*Our derivation.* We have no multi-class analogue, so the cell reuses the same
`per_press_accuracy`, quoted on Meat->Silicone: 94/120 = 0.783 -> 78 %.

*Caveat.* A two-class rate is easier than an eleven-class one, so this cell **understates** the gap
in the baseline's favour. It is quoted on Meat->Silicone because that is the real-data-trained
model, the closest counterpart to a model trained on the material it is tested on. The full set is
100 / 93 / 71 / 78 % --- have the unquoted Sim->Meat value, 71 %, ready.

### Bewley et al. [3] --- "lump found when present, at 8 N" --- ours: 98 % Sim->Silicone, 88 % Sim->Meat

*Their definition.* The proportion of trials in which a lump's presence was detected at 8 N, read
off their Fig. 13. It is a *when-present* rate, not a balanced accuracy.

*Our derivation.* `detection_rate`: the fraction of **vessel-present** frames in which at least one
marker passes 0.5. Sim->Silicone 92/94 = 0.979 -> 98 %; Sim->Meat 113/129 = 0.876 -> 88 %.

*Caveat --- volunteer this one unprompted.* The other half of the confusion matrix is on no cell:
`false_alarm_rate`, the fraction of **vessel-free** frames in which the model fires anyway, is
0 % Sim->Sim, **23 % Sim->Silicone, 52 % Sim->Meat and 100 % Meat->Silicone**. A high when-present
rate is cheap for a detector that fires often.

### Yan & Pan [7] --- "estimated -> true tumour centroid" --- ours: 3.7 +/- 2.5 mm, Sim->Silicone

*Their definition.* The centroid clustering error: the Euclidean distance between the estimated and
the true tumour centroid, 1.4--4.7 mm over three tumours after 30 Bayesian-optimisation iterations.

*Our derivation.* `centre_mae_mm` +/- `centre_sd_mm`. Per frame, the 2-D Euclidean distance between
the centroid of the markers the model calls vessel and the centroid of the truly-vessel markers,
converted to mm, then averaged over the frames that have a prediction:
Sim->Silicone 3.690 +/- 2.472 mm over n = 92 -> **3.7 +/- 2.5 mm**. The other transfers are
1.4 / 3.7 / 3.4 / 2.7 mm, so Sim->Silicone is the **worst** of the four.

### Raina et al. [8] --- "reconstructed -> CT centreline" --- ours: 1.2 mm Sim->Silicone, 5.5 mm Sim->Meat

*Their definition.* The k-nearest-neighbour distance between the ultrasound-reconstructed vein and
artery centrelines and the CT ground-truth centrelines of the phantom, ICP-aligned; mean 2.15 mm.

*Our derivation.* `map.one_directional_mm` in `symmetric_distances.json` --- the mean distance from
each **predicted** top-view-map vessel pixel to the nearest **true** vessel pixel, 1 px = 1 mm:
Sim->Silicone 1.21 -> **1.2 mm**, Sim->Meat 5.49 -> **5.5 mm**.

*Caveat --- the weakest match in the column, so say it first.* Theirs is a 3-D centreline-to-
centreline distance against a CT ground truth; ours is a 2-D distance from a predicted pixel to the
nearest pixel of a labelled **band**, at a threshold chosen on the test set. A band is wider than a
centreline, so our figure is structurally the smaller one. It is in the table because it is the
closest published definition to ours anywhere in the comparison, not because the two are
interchangeable.

### 3D CNN [6] --- "centre within tolerance, per video" --- ours: 78 % at 5 mm, 47 % at 3 mm, Sim->Silicone

*Their definition.* The proportion of the 100 validation **videos** whose predicted **"Guan" point**
--- the TCM pulse-taking position on the radial artery, a single pixel marked by a physician --- falls
within 50 px of that label, at 1024x544 px; the paper gives no px-to-mm scale, so their tolerance
cannot be converted. It is a regressed landmark point, not a centroid of a segmentation and not a
centreline (verified in the paper on 2026-09-04).

*Our derivation.* `hit_rate`, the fraction of vessel-bearing **frames** whose predicted centroid
lies within tau mm of the true centroid, swept over tau. Sim->Silicone: tau = 5 mm 0.783 -> 78 %,
tau = 3 mm 0.467 -> 47 % (the full sweep is 1 / 10 / 27 / 47 / 78 % at 0.5 / 1 / 2 / 3 / 5 mm).

*Caveat.* Their unit is a video, ours a frame, and the two tolerances are not comparable because
theirs has no metric scale. Two values are printed rather than one so the reader sees how steeply
the rate falls with the tolerance.

### Beasley et al. [4] --- "signed lateral bias of artery position" --- ours: +0.4 +/- 2.7 mm, Sim->Silicone

*Their definition.* A **signed** across-artery position error, +1.3 mm with an SD of 7.6 mm over 33
points --- a bias, not an accuracy: it cancels errors on opposite sides of the vessel.

*Our derivation.* `centre_lat_signed_mean_mm` +/- `centre_lat_signed_sd_mm`. The centroid offset of
the frame is projected onto the **normal of the vessel**, and the vessel's direction is taken as the
principal axis of that frame's labelled markers, so the lateral axis follows the vessel instead of
assuming an image axis. Sim->Silicone +0.410 +/- 2.708 mm -> **+0.4 +/- 2.7 mm**.

*Why this is not the image x axis (a correction made on 2026-09-03).* The cell used to read
-0.1 +/- 3.1 mm, the signed offset along image x. In Sim and Silicone a frame's labelled band spans
about 21 mm in x against 4--8 mm in y (principal axis 6--11 degrees from x), so image x is the
**along**-vessel axis --- the opposite of what Beasley measures; on Meat it is the other way round.
An axis-aligned cross-check of the corrected quantity gives +0.36 +/- 3.13 mm, so the correction
does not hinge on the principal-axis estimate. See `comparison-review/review.pdf` Sec. 2.7.

### Hampson et al. [5] --- "absolute lateral error of centre, in range" --- ours: 1.9 +/- 2.0 mm, Sim->Silicone

*Their definition.* The **absolute** lateral error of the estimated artery centre per press,
0.58 +/- 0.25 mm, holding only while the artery is within range of the tactile array; out of range
their Fig. 4 shows roughly 2--8 mm. The paper does not say whether the +/- is an SD, an SE or a
range.

*Our derivation.* `centre_lat_abs_mean_mm` +/- `centre_lat_abs_sd_mm`: the unsigned value of the
same across-vessel projection as the Beasley cell. Sim->Silicone 1.880 +/- 1.983 mm ->
**1.9 +/- 2.0 mm** (median 1.41 mm). Previously 3.7 +/- 2.5 mm, which was the full 2-D distance and
therefore included the along-vessel component that Hampson does not measure.

*Reading.* This is the row we lose most clearly: 1.9 mm against 0.58 mm, about three times worse,
and their number is the in-range best case while ours is over every frame.

### Ours (last row) --- "predicted pixel -> nearest true pixel" --- 1.1, 1.2, 5.5, 1.3 mm

Not a recomputation: this is the manuscript's own Table 4 quantity, the mean distance from each
predicted vessel pixel of the top-view map to the nearest true vessel pixel, for
Sim->Sim / Sim->Silicone / Sim->Meat / Meat->Silicone. `symmetric_distances.py` recomputes it from
the manuscript's own map runs and asserts it against Table 4: 1.05 / 1.21 / 5.49 / 1.31 mm. It is
the same definition the segmentation bar panels now print, so the two Results blocks agree.

## Reproducing the column

```bash
# frame-space cells (everything except the Raina cell and the Ours row)
python analysis/scripts/comparable_metrics.py
# map-space cells (Raina cell, Ours row) and the segmentation bar panels
python analysis/scripts/symmetric_distances.py   # needs REPO on disk
python analysis/scripts/segmentation_bars.py
```

`comparable_metrics.py` reads only the saved `.npz` predictions, so it runs in seconds and needs no
GPU, no simulator and no model. `symmetric_distances.py` additionally reads the vessel-map PNGs from
the code repository.

## What no cell in the column shows

Ranked as `comparison-review/review.pdf` Sec. 5 ranks them, all presenter material:

1. The **false-alarm rates** behind the three detection cells (23 / 52 / 100 %, above).
2. **n and independence** --- no cell carries an n, and frames within one trial are not independent
   (Silicone is ten annotated videos, Meat ten trials).
3. **Two operating points in one table** --- frame-space cells at threshold 0.5, the map-space cells
   at a threshold chosen on the test data.
4. The **Meat recall ceiling**: a perfectly localised model can recover at most 53 % of the shipped
   Meat map truth, so Meat recalls are not comparable with Silicone ones. Only the Sim->Meat
   *distance* is printed, which the ceiling does not inflate.
5. The **reverse-direction distance** (truth to prediction), 0.03 / 2.24 / 0.69 / 0.18 mm in frame
   space and an ASSD of 0.75 / 1.49 / 4.36 / 1.50 mm in map space.
