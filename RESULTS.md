# Key results of the paper, and where the repository produces them

Companion to the manuscript *"Toward Trustworthy Robot-Assisted Sliding Palpation
for Shallow Vessel Localisation with a Calibrated Digital Twin"*
(`/home/psb120/Downloads/2026_ECCV_Piotr_Blaszyk/main.tex`).

For each result it records **what the paper claims**, **what the figure actually
shows** (an independent reading of the image file, combined with the caption),
and **which script in this repository produces it**. The last section verifies
the numbers against a live run.

Datasets throughout: **A** = simulation (digital twin), **B** = real silicone
vascular phantom, **C** = real raw-meat phantom with straw vessel surrogates.
Models: `compact` = `latent_dim` 64, `large` = `latent_dim` 256.

> **Excluded by request:** Table `tab:localisation-map` (the TP/FP/FN and max-L1
> binned localisation counts) was computed by hand from a manually annotated
> sub-figure and is therefore not reproducible from code. It is out of scope
> here, as are the validation-set columns of Table `tab:iou`.

---

## 1. Headline claims

| # | Claim | Value in paper | Where in paper |
|---|---|---|---|
| 1 | Marker-alignment MAE after domain adaptation | **0.49 mm** (13.5 px), down from 0.51 mm (14 px) | Abstract, §Domain Adaptation |
| 2 | AUROC, sim-trained model on silicone (A→B) | **0.72** | Fig. `roc-curve`(a) |
| 3 | AUROC, meat-trained model on silicone (C→B) | **0.68** | Fig. `roc-curve`(b) |
| 4 | AUROC, sim-trained model on meat (A→C) | **0.60** | Fig. `roc-curve`(c) |
| 5 | Per-node foreground IoU, A→B test | **0.20** (background 0.85) | Table `tab:iou` |
| 6 | Per-node foreground IoU, C→B test | **0.15** (background 0.61) | Table `tab:iou` |
| 7 | Per-node foreground IoU, A→C test | **0.03** (background 0.88) | Table `tab:iou` |
| 8 | Binned vessel-map IoU after reprojection | **0.80**, max localisation error 5 mm | Abstract, §Comparison with Baselines |
| 9 | Inter-marker spacing (context for claim 1) | 55 px ≈ 2 mm | §Domain Adaptation |
| 10 | Training set size | 500 simulated trajectories, 80:20 vessel-present:absent | §System Overview |
| 11 | Clip length | 7 frames (beat 1, 3, 5); deep supervision on all frames, test on central frame | §System Overview |

---

## 2. Figures

### Fig. `da-results` — domain-adaptation marker alignment
Files: `press_2.png`, `twist_z_2.png`, `twist_x_2.png`, `slide_2.png`

**What the images show.** Four panels, each a white background carrying the
sensor's full marker grid (~127 markers in the ViTacTip's circular dome layout).
Every marker appears twice: a **red** filled disc (simulated position) and a
**green** filled disc (real position), joined by a short **blue** segment marking
the residual offset. The red and green discs overlap heavily almost everywhere,
so the blue segments are short — visual confirmation of sub-marker-spacing
agreement. The panels differ in the *pattern* of the residuals, which is the
informative part: in `press` the offsets are small and radially organised; in
`twist_z` they form a coherent rotational swirl about the grid centre; `twist_x`
shows a similar but differently-oriented pattern; `slide` shows offsets biased
consistently in one direction, as expected from a tangential motion. Residuals
grow slightly toward the periphery of the dome in all four, which is where
projection error and contact modelling are least accurate.

**Caption's account.** "Alignment between simulated (red) and real (green) marker
positions after domain adaptation for four canonical interactions: (a) press,
(b) twist about the z-axis, (c) twist about the x-axis, and (d) slide."

**Producing script.** Drawn inline by the simulator, not by a separate plotting
stage: `Contact.generate_validation_img()` in
[`difftactile/main/main.py`](difftactile/main/main.py) (~line 172), called from
`compute_da_loss()` (~line 149), which loads the real markers for the current
trajectory and computes the MAE at the same time. Output goes to
`difftactile/output/da_overlay_{press,twist_z,twist_x,slide}.png` (config key
`files.da_overlay`). Triggered during a collection run when
`meta.load_params_from_bo == 1`. The paper's `*_2.png` filenames are these files
renamed by hand.

> The live `generate_validation_img()` draws only the red and green discs. The
> blue residual segments in the published figure come from an earlier revision of
> that function and are not reproduced by current code.

The real marker positions it compares against are prepared separately by
`extract_reorder_save_markers()` in
[`difftactile/data_analysis/experiment/domain_adaptation.py`](difftactile/data_analysis/experiment/domain_adaptation.py)
(entrypoint `difftactile/scripts/script_domain_adaptation.py`).

---

### Fig. `roc-curve` — ROC curves on the test set
Files: `roc_curve.pdf` (A→B), `roc_curve_meat.pdf` (C→B), `roc_curve_sim_to_meat.pdf` (A→C)

**What the images show.** Each panel plots a thick blue ROC curve over a grey
diagonal chance line, on bold-ticked axes from 0 to 1. Large red dots mark
sampled decision thresholds, labelled in bold (0.05, 0.11, 0.16, 0.21, 0.26,
0.32 … 0.63) and crowding together in the lower-left as the threshold rises. In
panel (a) the curve rises steeply — reaching TPR ≈ 0.65 at FPR ≈ 0.3 and TPR ≈
0.85 at FPR ≈ 0.58 — sitting clearly above chance across the range, consistent
with the stated AUROC of 0.72. Threshold 0.5 falls on the steep lower section
(TPR ≈ 0.3, FPR ≈ 0.1), showing that the operating point used for the IoU numbers
is conservative: it trades recall for a low false-positive rate.

**Caption's account.** "ROC curves on the test set. The blue curve shows
(FPR,TPR) values for thresholds in (0,1), with red markers indicating selected
thresholds. (a) Train in simulation, test on silicone phantom (AUROC=0.72).
(b) Train on meat phantom, test on silicone phantom (AUROC=0.68). (c) Train in
simulation phantom, test on meat phantom (AUROC=0.60)."

**Producing script.** `evaluate_and_plot_roc()` in
[`difftactile/cnn/segmentation_gnn.py`](difftactile/cnn/segmentation_gnn.py)
produces panel (a) only. Panels (b) and (c) had no ROC path in the repository —
`C-to-B` and `A-to-C` ran `trainer.test()` and reported IoU alone. All three, in
both pretrained and retrained variants, are now produced uniformly by
[`difftactile/cnn/auroc_all_scenarios.py`](difftactile/cnn/auroc_all_scenarios.py):

```bash
python -m difftactile.scripts.script_auroc_all_scenarios
```

writing `difftactile/output/roc_curves/roc_curve_<config>_<weights>.pdf` and
`AUROC_RESULTS.md`.

---

### Fig. `video-pred-grid` — per-frame confusion overlays
Files: `0cm.png` (background only), `1cm.png` (1 vessel), `2cm.png` (2 vessels)

**What the images show.** Three small square panels on a black background, each
the marker grid of one central video frame rendered as a disc per marker, colour
coded by confusion outcome. `0cm.png` is entirely **yellow** — every marker a
true negative, as expected for a vessel-free frame. `1cm.png` shows one
approximately horizontal band of **blue** markers across the upper-middle of the
grid with a single **red** disc at its right end, i.e. a detected vessel line
with one miss. `2cm.png` shows two separated bands: an upper band mixing **red**
and **green** with **blue** at the edges, and a lower band of **blue** and
**green** — two vessels recovered with a mixture of hits, misses and
false alarms concentrated at the vessel boundaries. Across all three, errors sit
at band edges rather than scattered randomly, which supports the paper's argument
that low foreground IoU is largely a boundary-resolution effect at ~2 mm marker
spacing rather than a failure to localise.

**Caption's account.** "Sample shallow vessel detection model predictions on
central frames of example video clips from the experimental test set. Confusion
overlay legend: green: TP, blue: FP, red: FN, yellow: TN."

**Producing script.** The interactive viewer
`Visualisation.visualise_gnn(mode='predictions', ...)` in
[`difftactile/cnn/visualise.py`](difftactile/cnn/visualise.py), which tiles
Ground Truth / Hard Prediction / Confusion Matrix / Soft Prediction / Metadata
windows and steps through frames. It previously hardcoded both the checkpoint
(via `meta.cnn_gnn`) and the dataset (an `if True:` / `if False:` toggle); it now
takes a scenario name, so all six combinations are reachable:

```bash
./docker/view_predictions.sh A-to-B              # published checkpoint
./docker/view_predictions.sh A-to-C --retrained  # locally trained checkpoint
```

> Note the code's own comment labels FP red and FN blue, but the values are
> written in OpenCV BGR order, so the **rendered** colours are FP red, FN blue —
> matching the caption's green TP / yellow TN, with FP and FN transposed relative
> to the caption text. Worth a check before the camera-ready version.

---

### Fig. `vessel-map` — bird's-eye vessel localisation map
Files: `confusion_overlay_vessel_map_conventional_big.png` (a),
`confusion_overlay_vessel_map_conventional_bbox_big.png` (b),
`confusion_overlay_vessel_map_pretty_meat.png` (c),
`photo_vs_video_ground_truth_big.png` (d)

**What the images show.**
- **(a)** A wide black canvas (1 mm per pixel) carrying about ten elongated,
  slightly diagonal streaks — one per sliding pass of the sensor. Each streak is
  a dense speckle of **red** (false negative) with **green** (true positive)
  cores at its centre and scattered **blue** (false positive) pixels along the
  margins. The green cores line up along a common axis running left-to-right
  across the canvas, which is the recovered vessel; the surrounding red is the
  ground-truth vessel region the sparse predictions did not cover. One isolated
  blue pixel sits far to the right, away from any streak.
- **(b)** The same overlay with **yellow** rectangles drawn around each streak —
  the 15 bins used for the binned localisation count. The bins tile the vessel
  axis; a small isolated yellow box at the far right encloses the stray region
  with no prediction, marking the single false-negative bin the caption calls out.
- **(c)** A much smaller, lower-resolution version of the same style of overlay
  (red/green/blue speckle streaks on black) for the meat-phantom-trained model —
  visibly sparser and noisier, with less coherent green.
- **(d)** The two ground-truth sources overlaid: broad, smooth **blue** ribbons
  (one per pass) with **green** and **red** speckle running along them. The blue
  ribbons are contiguous and much wider than the speckle, showing the two
  ground-truth derivations agree in placement but differ in extent — the
  photo-derived truth is a solid swathe, the video-derived one a sparse
  point set along the same trajectory.

**Caption's account.** "Vessel localisation map (2D→3D→2D geometric projection;
1×1 mm per pixel). (a) Confusion overlay of video ground truth vs. prediction
(green: TP, blue: FP, red: FN, black: TN), sim-trained model. (b) Same overlay
with 15 bins (yellow), sim-trained model; the small right-hand bin marks a false
negative. (c) Confusion overlay, meat-phantom-trained model. (d) Overlay of
ground truth projected from video vs. from a top-view photo."

**Producing script.** `PredictExp.go()` in
[`difftactile/data_analysis/experiment/predict_exp.py`](difftactile/data_analysis/experiment/predict_exp.py).
It lifts each per-marker prediction to 3D through the sensor dome map, applies
the sensor pose from robot kinematics, and bins into a 1 mm grid
(`predict_clip()`, threshold 0.58), then writes the confusion overlay and the
multi-panel comparison. Wrapper:

```bash
./docker/vessel_map.sh
```

Outputs: `difftactile/output/confusion_overlay_vein_map.png` (a/c),
`segmentation_mask_predicted_aggregated.png`, and
`exp_overlay_downscaled.pdf` (the panel grid behind (d)).

**Silicone only.** The workspace bounds, sensor-to-plane offset and marker
reshapes are specific to the silicone rig, which is why the paper notes the
sim→meat localisation map is omitted.

Sub-figure **(b)** is stated in the paper to be manually annotated in external
photo-editing software; the yellow bins have no code path here. The "pretty"
recolouring of (c) is
[`difftactile/data_analysis/testing/prettify_confusion_image.py`](difftactile/data_analysis/testing/prettify_confusion_image.py).

---

### Fig. `annotation-line` — annotation examples and the meat setup
Files: `raw0.png` (a), `ann0.png` (b), `papaya.jpg` (c), `meat_ground_truth_label.jpg` (d)

**What the images show.**
- **(b)** A greyscale ViTacTip camera frame: a bright circular dome covered by a
  regular grid of dark biomimetic tips. Two annotation lines cross it
  horizontally — an upper **green** line and a lower **red** line, each a dense
  chain of small dots with one large filled disc at its centre marking the
  manually clicked point. The two lines are the two vessels, and the dot chains
  are the derived centrelines spanning the dome.
- **(d)** A greyscale meat-phantom frame with the marker grid overlaid as
  coloured dots: mostly **green** (no vessel) with a vertical stripe of **red**
  (vessel present) dots down the right-of-centre column, following a bright
  specular ridge in the image where the straw lies beneath. The red stripe is
  visibly offset by roughly one marker from the brightest part of the ridge,
  which is the labelling misalignment the caption flags.

**Caption's account.** "(a),(b) Example video frames with 2 vessels: (a) without
and (b) with vessel annotations, where the large red/green circles are manually
annotated points and the small-circle line is the vessel centreline from a
manually specified orientation. (c) Meat-phantom experimental setup. (d)
Ground-truth vessel labels (red: present, green: absent) from robot kinematics on
the meat phantom; note the minor misalignment between labels and the actual
metal-straw vessel placed on the phantom."

**Producing tools — both interactive.** See §4 below; wrapper:

```bash
./docker/annotate_data_bare_metal.sh --silicone   # (b): click annotator, also replays existing points
./docker/annotate_data_bare_metal.sh --meat       # (d): marker-label review
```

---

## 3. Verification against a live run

Run in the Docker container on an RTX 3080, published Zenodo bundle restored.
Retrained rows come from `script_segmentation_gnn <config> --train`; AUROC from
`script_auroc_all_scenarios`. AUROC is computed **per marker node across video
frames** (12700 = 100 silicone clips × 127 markers; 25273 = 199 meat clips × 127)
— not from a reprojected phantom map.

| Result | Paper | This repo | Verdict |
|---|---|---|---|
| AUROC A→B (published ckpt) | 0.72 | **0.7314** | ✅ matches |
| AUROC C→B (published ckpt) | 0.68 | **0.6786** | ✅ matches |
| AUROC A→C (published ckpt) | 0.60 | **0.8183** | ❌ **disagrees — see below** |
| FG IoU A→B, silicone test | 0.20 | **0.2309** | ✅ close |
| BG IoU A→B, silicone test | 0.85 | **0.8475** | ✅ matches |
| FG IoU C→B, silicone test | 0.15 | **0.1577** | ✅ matches |
| BG IoU C→B, silicone test | 0.61 | **0.5941** | ✅ close |
| FG IoU A→C, meat test | 0.03 | **0.1921** | ❌ **disagrees — see below** |
| BG IoU A→C, meat test | 0.88 | **0.7729** | ❌ disagrees (same cause) |
| Vessel map (A→B) regenerates | — | ✅ runs, writes all figures | ✅ |
| DA overlays regenerate | — | ⚠️ code present; needs a full BO collection run | — |

Retrained checkpoints, for reference (not paper numbers — fresh stochastic runs):
A→B **0.7807**, C→B **0.6737**, A→C **0.8253**.

### The A→C discrepancy

**The repository is the correct side of this disagreement, and the paper's A→C
numbers should be revisited.**

The paper's sim→meat figures were produced by an evaluation path that fed the
checkpoint **unnormalised** inputs. On the old `sim-to-meat-test` branch the
`test_dataset.set_stats(stats)` call sat inside a dead `if False:` block, so the
dataset's `warmup` flag stayed set and normalisation was silently a no-op — even
though the model had been trained on normalised inputs. Evaluating a model on a
different input distribution than it was trained on is a bug, not a modelling
choice.

Applying the statistics the checkpoint expects moves the cross-domain numbers:

| | vein (FG) IoU | background IoU | AUROC |
|---|---|---|---|
| Old path (unnormalised) | 0.034 | 0.888 | ~0.60 |
| Corrected (normalised) | **0.198** | 0.809 | **0.82** |

This was found and fixed before this measurement run and is already flagged in
`REPRODUCTION_TEST.md`, which warned that "if 0.034 (or anything derived from it)
appears in the paper, that figure needs revisiting". It does — both the sim→meat
row of Table `tab:iou` (FG 0.03, BG 0.88) and Fig. `roc-curve`(c) (AUROC 0.60)
derive from it.

**Consequence for the manuscript.** The reported degradation under transfer to
raw meat is **overstated**. A real domain gap remains in per-node IoU (0.192 on
meat vs 0.231 on silicone), so the qualitative claim that fine marker-level
classification is harder under domain shift survives. But the AUROC ordering
changes: A→C is no longer the worst configuration, and sentences such as
"Performance decreases under the larger domain shift to the raw-meat phantom"
(abstract) and the framing of §Synthetic vs Real Train Set need rewording against
the corrected numbers.

### Results not verifiable from code

| Result | Why |
|---|---|
| MAE 0.49 mm / 13.5 px (claim 1) | Requires a full Bayesian-optimisation domain-adaptation run over the four canonical trajectories; the BO artifacts (`bo_all_params.json`, `bo_all_targets.json`) ship in the bundle but the MAE is printed during collection, not stored as a reported metric. |
| Binned vessel-map IoU 0.80, max error 5 mm (claim 8) | Table `tab:localisation-map`; computed by hand from the manually annotated sub-figure. Excluded by request. |
| Fig. `vessel-map`(b) yellow bins | Manually drawn in external software. |
| Tables `tab:comparison-1/2` | Literature comparison, no code. |

---

## 4. Interactive tools behind the annotation figures

Both are manual tools, gated behind `DIFFTACTILE_INTERACTIVE=1` so unattended
runs never block on a window. Each **creates and reviews** annotations in one
method — opening it on already-annotated data replays what is on disk.

| Dataset | Tool | Entrypoint |
|---|---|---|
| Silicone (B) | `SiliconePreprocessData.annotate()` — cv2 click annotator, loads existing `.pkl` and redraws it | `script_annotate_silicone` |
| Meat (C) | `MeatPreprocessData.browse_annotations()` — per-marker label review | `script_browse_meat_annotations` |

```bash
./docker/annotate_data_bare_metal.sh --silicone
./docker/annotate_data_bare_metal.sh --meat
```

**Data availability.**

- `--meat` works directly from the bundle and draws the labels over the **real
  camera frames**. Each trial now ships `clean/<trial>/frames.mp4` — the 26
  decimated frames preprocessing kept, H.264 CRF 26, ~2.3 MB — aligned 1:1 with
  `marker_labels.npz`, which reproduces Fig. `annotation-line`(d) directly. This
  adds ~52 MB to the bundle in place of the 1.6 GB raw archive. Built by
  `script_make_meat_clean_videos`.
- `--silicone` needs the dilated videos and annotation pickles, which are **not**
  in the bundle (intermediate preprocessing stages). The wrapper stages them from
  the author's local tree
  (`.../diff-tactile-fork/difftactile/manual_or_experimental_data/endgame/`),
  overridable with `--source`. Staged here: 10 videos, 182 annotation dots across
  8 of them (`metadata_1.0_*` and `metadata_2.0_*` hold the 2-vessel frames
  behind Fig. `annotation-line`(b)).

Meat preprocessing now starts from the shipped `frames.mp4` + `frames_poses.npz`
rather than the raw archive. Because the shipped video is lossy, that path
*regenerates* the dataset rather than reproducing it exactly: marker positions
move by a median 0.03 px (p99 0.47 px) against ~55 px marker spacing, 20 of 23
trials keep bit-identical labels, and 16 of ~76000 labels differ overall (0.02%).
The bundled `*.npz` remain authoritative; `DIFFTACTILE_MEAT_FROM_RAW=1` with the
raw archive gives an exact rebuild.

---

## 5. Reproducing everything

```bash
# data (once)
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz

# all six scenarios: AUROC table + ROC curves
python -m difftactile.scripts.script_segmentation_gnn A-to-B --train   # and C-to-B, A-to-C
python -m difftactile.scripts.script_auroc_all_scenarios

# per-node IoU for one configuration
python -m difftactile.scripts.script_segmentation_gnn A-to-B --eval

# bird's-eye vessel map (silicone only)
./docker/vessel_map.sh

# interactive: per-frame confusion overlays, any of the six scenarios
./docker/view_predictions.sh A-to-B

# interactive: annotation / annotation review
./docker/annotate_data_bare_metal.sh --silicone
./docker/annotate_data_bare_metal.sh --meat
```
