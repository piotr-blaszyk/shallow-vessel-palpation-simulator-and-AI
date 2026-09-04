# `analysis/` — every number and figure of the conference poster

This tree reproduces the **poster** that accompanies the manuscript. The poster reports
more than the manuscript does: the same four models scored as a *detection* problem
(counts, Hungarian matching, F1, OSPA), a per-vessel centreline/centroid decomposition,
and a baseline comparison in which **our own predictions are recomputed under each
baseline's own metric definition** instead of being quoted side by side with numbers that
measure different things.

Nothing here trains a model or runs the simulator. The four published checkpoints are
evaluated once, the published top-view vessel maps are read off disk, and everything else
is derived from those two. One full pass is **under a minute of GPU time and about two
minutes in total** (the optional overlay stages add ~15 minutes).

```bash
# inside the container, from the repository root
./docker/reproduce_analysis.sh                  # stages 1-12: every poster figure and number
./docker/reproduce_analysis.sh --with-overlays  # + the per-data-point diagnostic overlays
./docker/reproduce_analysis.sh --list           # the stage table
./docker/reproduce_analysis.sh --only 4,7       # re-run two stages
```

Prerequisites: the Zenodo data bundle restored (`./data/restore_data.sh`, which brings the
checkpoints, the five-seed sweep, the real datasets and the published top-view map runs) and
a GPU **for stage 1 only** — the bundle also ships stage 1's output, so stages 2–12 run on a
CPU-only machine.

## Layout

```
analysis/
├── scripts/     the 16 scripts below; paths.py resolves every input and output
├── assets/      the one non-generated input (the pipeline overview figure, cropped for
│                the ViTacTip photograph of the workflow diagram)
├── results/     machine-readable output: *.json (tracked) and *.npz (gitignored, shipped
│                in the Zenodo bundle because they need a GPU)
├── figures/     every figure the poster prints (tracked)
├── reports/     Markdown write-ups generated from results/ (tracked)
└── overlays/    optional per-data-point diagnostic renders (gitignored)
```

`scripts/paths.py` derives every path from its own location, so the tree needs no editing
after a clone. It also resolves the **published top-view map run** of each configuration:
the newest timestamp under `difftactile/output/vessel_maps/<cfg>/` if the repository has run
the maps itself, otherwise the copy the bundle restores into
`difftactile/output/manuscript_artifacts/vessel_maps/<cfg>/`. Runs are never overwritten and
`data/make_data_bundle.sh` stages the newest one, so "newest" and "published" cannot drift
apart.

## Which script produces which poster block

Column headings are the poster's own. `A / B / C` are the code's dataset names
(Sim / Silicone / Meat); `X-to-Y` reads "trained on X, tested on Y".

| Poster block | Artefact | Script (stage) |
|---|---|---|
| **Motivation** — opening illustration | — | Hand-made illustration; not derived from any measurement. |
| **Motivation** — *Vessel-detection modalities* | — | Literature values, not repository output. The fully cited version of the table is published at [`docs/poster-motivation-bibliography/`](../docs/poster-motivation-bibliography/). |
| **Motivation** — *Contributions* | — | Prose. |
| **Methods** — *Workflow*, sensor / mesh / simulated interactions / domain randomisation / real datasets / annotation blocks | `figures/workflow/{vitactip,mesh,A1,A2,B1,B2,B3,dr_heading,dr_stiffness,real_*,ann_*}.png` | `make_workflow_images.py` (12) |
| **Methods** — *Workflow*, per-frame prediction block | `figures/workflow/pred_sim_to_meat.png` | `select_sim_to_meat_frames.py` (11) → `make_workflow_images.py` (12) |
| **Methods** — *Workflow*, *Top-view vessel maps* row | `figures/workflow/map_*.png` | `make_workflow_images.py` (12), copied from the published map runs |
| **Methods** — *Workflow*, *Frame-space vessel centrelines and centroids* | `figures/workflow/cl_grid_<0..3>_<config>.png`, `cl_legend.png`, `frame_grid.tex` | `detection_ospa.py` (4) → `workflow_centreline_panels.py` (10) |
| **Methods** — *Workflow*, *Top-view-space vessel centrelines and centroids* | `figures/workflow/cl_map_<config>.png` | `detection_ospa.py` (4) → `workflow_centreline_panels.py` (10) |
| **Methods** — *ST-GNN input graph* | `figures/gnn_graph.png` | `gnn_graph_figure.py` (9) |
| **Results** — take-home box | 1.2 mm; AP 0.32 vs 0.30 | `symmetric_distances.py` (2), `pr_curves_with_ap.py` (8) |
| **Results** — *Comparison against baselines*, our half of every row | `results/comparable_metrics.json`, `results/symmetric_distances.json` | `comparable_metrics.py` (3), `symmetric_distances.py` (2) |
| **Results** — *Comparison against baselines*, the baselines' half | — | Read out of each cited paper's own full text; not repository output. |
| **Results** — *Centreline detection and localisation* | `figures/detection_bars.pdf` | `detection_ospa.py` (4) → `detection_bars.py` (7) |
| **Results** — *Segmentation distance* | `figures/segmentation_distance.pdf` | `symmetric_distances.py` (2) → `segmentation_bars.py` (6) |
| **Results** — *Vessel-count confusion* | `figures/detection_confusion.tex` | `detection_ospa.py` (4) → `detection_bars.py` (7) |
| **Results** — *Vessel present or absent* | `figures/binary_confusion.pdf` | `detection_bars.py` (7) |
| **Results** — *Precision–recall curves* | `figures/pr/mean_pr_curve_<config>.pdf`, `curve_legend_vertical.pdf`, `xlabel_recall.pdf`, `ap_values.json` | `pr_curves_with_ap.py` (8) |
| **Results** — *Foreground IoU* | `figures/segmentation_iou.pdf` | `symmetric_distances.py` (2) → `segmentation_bars.py` (6) |
| **Results** — *Limitations*, *Future work* | — | Prose. |

Model colours (light Sim→Sim to dark Meat→Silicone) are defined once in
`scripts/model_colours.py`; every figure script imports them, and running that script prints
the matching LaTeX `\definecolor` block.

## The scripts, in dependency order

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `frame_space_predictions.py` | the four published checkpoints + their test-loader pickles | `results/frame_space_predictions_<config>.npz` — per-marker probability, label, pixel position, trial and frame index of every central frame |
| 2 | `symmetric_distances.py` | (1), the published map runs | `results/symmetric_distances.{json,md}` — foreground IoU and four distance definitions, video-frame and top-view space |
| 3 | `comparable_metrics.py` | (1) | `results/comparable_metrics.json` — our predictions under each baseline's own metric |
| 4 | `detection_ospa.py` | (1), the published map runs | `results/detection_ospa.json` — counts, matching, precision/recall/F1, OSPA, at cutoffs 5 mm and 1 mm |
| 5 | `detection_ospa_report.py` | (4) | `reports/detection-ospa.md` |
| 6 | `segmentation_bars.py` | (2) | `figures/segmentation_iou.pdf`, `figures/segmentation_distance.pdf` |
| 7 | `detection_bars.py` | (1), (4) | `figures/detection_bars.pdf`, `figures/detection_confusion.tex`, `figures/binary_confusion.pdf` |
| 8 | `pr_curves_with_ap.py` | the published five-seed sweep | `figures/pr/*` |
| 9 | `gnn_graph_figure.py` | `difftactile/output/base-graph-connectivity.npz` | `figures/gnn_graph.png` |
| 10 | `workflow_centreline_panels.py` | (1), (4), the published map runs | `figures/workflow/cl_*`, `figures/workflow/frame_grid.tex`, `results/workflow_panel_choice.json` |
| 11 | `select_sim_to_meat_frames.py` | (1), the meat dataset | `overlays/sim_to_meat_frame_choice/<timestamp>/` |
| 12 | `make_workflow_images.py` | `docs/videos/`, `docs/images/sensor_mesh/`, `assets/`, the published map runs, (11) | `figures/workflow/*.png` |
| 13 | `centreline_overlays.py` *(optional)* | (1), the published map runs | `overlays/centrelines/<timestamp>/` — one per-vessel overlay per scored data point |
| 14 | `detection_overlays.py` *(optional)* | (1), (4) | `overlays/detections/<timestamp>/` — the same for the detection/OSPA extraction |
| — | `paths.py`, `model_colours.py` | — | imported by the rest; not entry points |

## Self-checks

The pipeline refuses to produce a plausible-looking wrong answer:

* **Stage 1 asserts** the pooled TP/FP/FN/TN of each configuration at threshold 0.5 against
  `FRAME_SPACE_METRICS.md` (the manuscript's Table 4, upper half). A wrong checkpoint, a
  wrong test split or a mismatched normalisation pickle fails there rather than silently
  shifting every number downstream.
* **Stage 2 prints** its recomputed one-directional map distance and foreground IoU beside
  the manuscript's Table 4 values; they agree to the printed precision
  (1.05 / 1.21 / 5.49 / 1.31 mm and 0.42 / 0.28 / 0.21 / 0.31).
* **Stages 4 and 10 validate the object extraction against physical ground truth**: the
  recovered ground-truth vessel counts are 1 on the Sim map, 10 on each silicone
  whole-phantom map (the ten sweeps recorded in the run's `run.json`) and
  1/1/1/1/1/1/1/2/3/0 across the ten meat trials — which matches every trial directory name,
  including the `no-straw` control.

## Parameters that are choices, not fits

Stated here because they are the ones a reader should be able to disagree with:

* **OSPA cutoff `c` = tolerance `τ` = 5.0 mm**, a clinical usefulness judgement, not tuned
  on the data; `detection_ospa.py` evaluates and reports **1.0 mm** (the IV-needle-access
  margin) as well, and both are in `results/detection_ospa.json`.
* **`α_FP` = 2, `α_FN` = 1** — a false vessel costs twice a missed one, because a phantom
  vessel sends a needle into tissue while a missed one only forfeits the assist.
* **Frame-space threshold 0.5** everywhere; the top-view maps use each configuration's own
  chosen operating point (precision ≥ 0.9, maximise recall), which is the manuscript's rule.
* **`R_SPLIT` = 3.0 mm**, the distance below which two score peaks may not split one
  connected component — chosen to sit above the marker pitch (1.8–2.0 mm) and below the
  smallest true vessel separation in these datasets (3.5 mm). A dataset with vessels closer
  than that needs the rule re-derived.
* **DBSCAN `eps`** — frame space 1.45 × the dataset's median marker spacing; map space 3 mm
  with `min_samples` 3 and clusters under 5 px dropped.

## Known caveats, all of them in the reports

* Every distance printed is **precision-side only**: it scores the elements the model
  predicted to be vessel. The map recalls behind those numbers are 0.79 / 0.44 / 0.86 / 0.47.
* The detection cells of the baseline comparison show the hit half. The matching
  **false-alarm rates on vessel-free frames** are 0.23 (Sim→Silicone), 0.52 (Sim→Meat) and
  1.00 (Meat→Silicone) — `results/comparable_metrics.json`, field `false_alarm_rate`.
* The lateral error is computed **per frame with all vessels merged** into one principal
  axis, not per vessel. Recomputing it per matched vessel pair gives 1.17 ± 1.34 mm on
  Sim→Silicone against the 1.88 ± 1.98 mm reported, so the merge makes the reported number
  *worse*, not better.
* `matched_mean_mm` (mean distance over matched pairs only) is reported but is misleading in
  isolation: on the silicone map the assignment pairs sweeps at opposite ends of the phantom,
  giving 13.87 mm against 1.38 mm over the pairs inside the tolerance. Read OSPA and F1
  together instead.
