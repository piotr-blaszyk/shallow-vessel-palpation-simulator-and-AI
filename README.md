# Robot-Assisted Sliding Palpation for Shallow Vessel Localisation with a Calibrated Digital Twin

Locating features **hidden beneath a soft surface** — blood vessels in a silicone phantom,
straws buried under layers of steak — from the deformation of the marker grid of a ViTacTip
optical tactile sensor sliding over it.

The core question: **can a model trained entirely in simulation localise subsurface structure
in the real world?** A digital twin (a Taichi FEM simulation of the sensor pressing on a
phantom with a stiff inclusion) is calibrated against the real sensor by Bayesian optimisation,
generates labelled marker trajectories, and a spatio-temporal graph neural network trained only
on those is evaluated on video from the physical sensor — per marker, and as a top-view vessel
map.

This is a masters project, and a fork of [DiffTactile](https://difftactile.github.io/)
(Si et al., ICLR 2024): the upstream manipulation tasks were removed and the differentiable FEM
core repurposed. The pristine upstream state is preserved at the `upstream-difftactile` tag.

### Project repositories and data

The manuscript — *"Toward Trustworthy Robot-Assisted Sliding Palpation for Shallow Vessel
Localisation with a Calibrated Digital Twin"* — is backed by two repositories and one data
archive:

| Artefact | Role |
|---|---|
| **[shallow-vessel-palpation-simulator-and-AI](https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI)** (this one), DOI [10.5281/zenodo.21958186](https://doi.org/10.5281/zenodo.21958186) | **Main repository.** Simulator, domain adaptation, dataset generation, GNN training and evaluation, all figures and tables — everything needed to reproduce the published results. |
| [shallow-vessel-palpation-robot-control](https://github.com/piotr-blaszyk/shallow-vessel-palpation-robot-control), DOI [10.5281/zenodo.21958190](https://doi.org/10.5281/zenodo.21958190) | Drives the DOBOT Magician E6 arm that collected the real tactile recordings. Needed only to *gather new* data. |
| Zenodo **shallow-vessel-palpation-dataset**, DOI [10.5281/zenodo.21958107](https://doi.org/10.5281/zenodo.21958107) | Datasets, trained checkpoints and the manuscript's figures/tables (~275 MB). See [Data](#data-the-zenodo-bundle). |
| Project page: [piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI](https://piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI/) | Abstract and all videos in one place (`docs/`). |

```
   Taichi FEM digital twin                 Real ViTacTip sensor
   sensor + phantom + vessel               video, sliding over the phantom
   (calibrated by Bayesian optimisation)             │
            │                                        ▼
   simulated marker trajectories        tracked marker trajectories
            │                                        │
            └──────────────► ST-GNN ◄────────────────┘
                        train on Sim         test on Sim / Silicone / Meat
                              │
                              ▼
        per-marker vessel probability  ──►  top-view vessel map (1 mm/px)
```

---

## Quickstart (Docker)

**Docker is the only supported way to run this repository.** The image pins the whole stack
(CUDA 12.6, Taichi, PyTorch 2.8 + PyTorch Geometric, PySide6) and passes the host display
through so the simulator's windows work.

**Requirements:** Linux with an NVIDIA GPU (a must), the NVIDIA driver, Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Developed and tested on **Ubuntu 24.04** with an **NVIDIA GeForce RTX 3080 (10 GB)**. Since the
whole stack lives inside the container (Ubuntu 22.04 base image), the host only needs the three
items above, so other Ubuntu releases — including older ones such as 18.04 / 20.04 — should work
too (untested; the optional bare-metal annotator env, `annotate_data_bare_metal.sh`, needs
PySide6 wheels and hence glibc ≥ 2.28, i.e. Ubuntu 20.04+). Less GPU RAM than 10 GB will likely
work as well: the GNN training uses little; the Taichi simulator is the main consumer, and its
budget is set by `TI_DEVICE_MEMORY_GB` (see Troubleshooting). `main` is the only branch.

```bash
# 1. Clone
git clone https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI.git
cd shallow-vessel-palpation-simulator-and-AI

# 2. Fetch the data bundle (~275 MB) and unpack it into place
wget https://zenodo.org/records/21958107/files/shallow-vessel-palpation-data.tar.gz
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz

# 3. Build the image (~10-30 min), start the container, open a shell inside it
cd docker
./docker-build.sh
./docker-run.sh
./docker-connect.sh

# 4. INSIDE the container: verify GPU, dependencies and data
cd docker
./run_pipeline.sh check
```

> **Everything below runs inside the container, from its `docker/` directory**, unless a
> command says otherwise. Every in-container command is equivalent to
> `docker exec -it vessel-palpation ./docker/<script> <args>` from the host. Three scripts are
> launched **from the host** and `docker exec` in by themselves (`view_predictions.sh`,
> `annotate_data_docker.sh`, `record_videos.sh`), and one runs on **bare metal by design**
> (`annotate_data_bare_metal.sh`).

### The pipeline at a glance

Steps 1–2 are done and their outputs ship in the bundle, as do the simulated dataset (step 4)
and the trained models (step 5), so a fresh clone can start anywhere.

1. **Preprocess the real recordings** (Silicone, Meat) → [Real datasets](#real-datasets-annotate-review-preprocess).
2. **Calibrate the digital twin** (`./domain_adaptation.sh`) and adopt the fitted parameters into
   `system-params.json`; draw the validation panels (`./alignment_figures.sh`) →
   [Domain adaptation](#domain-adaptation-calibrating-the-digital-twin).
3. **Pick the temporal window** (`./ablation_clip_len.sh`).
4. **Generate the simulated dataset** (`./run_pipeline.sh sim-full` + reorder) →
   [Regenerate the simulated dataset](#regenerate-the-simulated-dataset).
5. **Train and evaluate the four models** (`./score_all_scenarios.sh --seeds 5`, or one
   configuration with `./run_pipeline.sh <config> --train`) →
   [Reproduce the published results](#reproduce-the-published-results).
6. **Render the qualitative results**: `./vessel_map_all.sh` (top-view maps),
   `./view_predictions.sh` (per-frame viewer), `./record_videos.sh` (the videos below).

### Which script produces which figure or table in the manuscript

| Script | Produces | Manuscript |
|---|---|---|
| `./annotate_data_bare_metal.sh --silicone` / `--meat` | the annotation viewers | **Fig. 5** (annotated Silicone frame, Meat labels) |
| `./domain_adaptation.sh` → `./alignment_figures.sh` | joint BO calibration; sim (red) vs real (green) marker alignment on the four validation interactions, MAE per panel | **Fig. 4** and the 0.50 mm MAE |
| `./ablation_clip_len.sh` | foreground IoU / AP per temporal window length, mean ± std over seeds | **Table 3** |
| `./score_all_scenarios.sh --seeds 5` | five-seed sweep of every configuration: mean PR curves ± 1 std (ROC twins too), the standalone shared legend (`curve_legend.pdf`) and x label (`xlabel_recall.pdf`), `AUROC_RESULTS.md`, `sweep.json`; `--replot` redraws the curves and legend of an existing sweep without training | **Fig. 6** |
| `python -m difftactile.scripts.script_frame_space_metrics` | per-marker statistics of each best-of-five model, pooled over all central frames → `FRAME_SPACE_METRICS.md` | **Table 4**, upper half |
| `./vessel_map_all.sh` | top-view vessel maps and per-pixel statistics for every configuration | **Fig. 7**, **Table 4** lower half |
| `./record_videos.sh` | the [demonstration videos](#demonstration-videos) | supplementary |
| `./sensor_mesh_screenshots.sh` | three Gmsh screenshots of the ViTacTip mesh from the +x side (level, 45° above, 45° below), lossless WebP in `docs/images/sensor_mesh/` | project page only |

Every mean ± std quotes the five seeds of the published sweep (`saved_models_sweeps/20260815-194045`);
wherever a **single** model is shown or tabulated it is that configuration's **best-of-five
instance by average precision** (`cnn/model_selection.py`) — the convention the manuscript states.

---

## Reproduce the published results

**Naming.** The code uses **A / B / C** for the three datasets; the manuscript spells them out.
`X-to-Y` reads "trained on X, tested on Y":

| Code | Manuscript | Dataset |
|---|---|---|
| **A** | **Sim** | 500 simulated slides (250 with / 250 without the vessel), 317 frames each |
| **B** | **Silicone** | real silicone phantom with shallow vessels: 10 annotated straight-line slides |
| **C** | **Meat** | real meat phantom (metal / silicone straws under 5 mm steaks): 10 trials |

Four (train → test) configurations, each selected **by name** and run either from the published
checkpoint (`--eval`) or retrained (`--train`):

| Config | Manuscript | Model | Default | AUROC | AP | FG IoU | BG IoU |
|---|---|---|---|---|---|---|---|
| `A-to-A` | Sim→Sim | large | `--eval` | 0.958 ± 0.001 | 0.491 ± 0.006 | 0.193 ± 0.019 | 0.818 ± 0.024 |
| `A-to-B` | Sim→Silicone | large | `--eval` | 0.779 ± 0.004 | 0.321 ± 0.004 | 0.234 ± 0.007 | 0.811 ± 0.027 |
| `A-to-C` | Sim→Meat | large | `--eval` | 0.837 ± 0.002 | 0.224 ± 0.004 | 0.171 ± 0.006 | 0.730 ± 0.019 |
| `C-to-B` | Meat→Silicone | compact | `--train` | 0.717 ± 0.054 | 0.304 ± 0.038 | 0.155 ± 0.012 | 0.537 ± 0.124 |

Mean ± std over the five seeds of the published sweep; per-marker over every central frame of
every test trial; IoU at threshold 0.5. `large` is `GNN(arch="large")` (`latent_dim` 256, the
`*_large` keys of the `gnn` config block), `compact` is `latent_dim` 64; a checkpoint only loads
into the architecture it was trained with. `A-to-A`, `A-to-B` and `A-to-C` score the **same**
sim-trained checkpoint on three test sets — the A→A → A→B gap is the sim-to-real transfer cost.

```bash
./run_pipeline.sh A-to-B                 # evaluate the published checkpoint (default for A-to-*)
./run_pipeline.sh C-to-B                 # train on meat, test on silicone (default for C-to-B)
./run_pipeline.sh A-to-C --train         # retrain any configuration from scratch
./run_pipeline.sh all-scenarios          # the four in sequence

./score_all_scenarios.sh                 # one table for all configurations -> AUROC_RESULTS.md
./score_all_scenarios.sh --pretrained    # published checkpoints only
./score_all_scenarios.sh --seeds 5       # retrain each configuration per seed: mean ± std,
                                         # mean PR/ROC curves, saved_models_sweeps/<TS>/
./score_all_scenarios.sh --replot        # redraw the published sweep's mean curves + the
                                         # shared threshold legend from saved scores (no training)
```

Each run prints AUROC, AP (with its chance level and lift) and both IoUs, and writes
`roc_curve_<config>.pdf` / `pr_curve_<config>.pdf` under `difftactile/output/`.
The older names `sim-to-silicone`, `sim-to-meat`, `silicone-to-meat` are accepted as aliases
(the last is a misnomer for A→C). The configuration and mode can also come from
`DIFFTACTILE_SCENARIO` / `DIFFTACTILE_MODE`.

**Training never overwrites the published checkpoints.** A `--train` run writes
`*_retrained_<config>` artifacts (`saved_models_sim/final_segmentation_model_gnn_sim_retrained_A-to-B.pt`
and its `test_loader_gnn_sim_retrained_A-to-B.pickle`, etc.); `DIFFTACTILE_OVERWRITE_PUBLISHED=1`
replaces the published files instead. A checkpoint and its test-loader pickle always travel
together — the pickle holds the normalisation statistics the checkpoint was trained with, and
mismatching them is a silent wrong answer, not an error.

**Metrics.** AUROC and AP are threshold-free (they read only the ranking of the probabilities,
so a domain shift of the output *scale* cannot masquerade as ignorance of where the vessels
are). They are reported together because AUROC normalises false alarms by the huge negative
count while AP ignores true negatives entirely and its chance level is the positive rate
(4–11% here) — the two can disagree in rank, and did on earlier checkpoints. IoU is per class
over the same marker predictions: **foreground** = vessel-present class
(`|pred ∧ true| / |pred ∨ true|`), **background** = vessel-absent class, always the flattering
one (the negative class is ~89% of nodes on Silicone, ~93% on Meat). Unlike AUROC/AP it depends
on the 0.5 threshold. This is the per-marker IoU; the top-view map has its own.

**Seeds.** Training is deterministic (`main/seeding.py`: torch/numpy, the shared `NP_RNG`,
DataLoader workers and deterministic CUDA kernels; default seed 42, `DIFFTACTILE_SEED=N`), so a
run reproduces bit-identically on the same machine. But the spread across seeds is large where
the training set is small — C→B AUROC ranges 0.604–0.779 over seven seeds, wider than any gap
between configurations — so quote a mean ± std, never a single run, and never the best seed.
`--seeds N` runs each seed in a fresh subprocess and keeps every seed's checkpoint + pickle
under `saved_models_sweeps/<timestamp>/<config>_seedNN/` with `sweep.json`.

**Sim test split.** Dataset A is split by trajectory (filenames sorted, cut at 70% / 85% →
350 / 75 / 75), so overlapping windows never leak across splits; every trajectory holds exactly
one vessel, so there is nothing to stratify over. `A-to-A` scores the held-out **test** split
read from the test-loader pickle (never re-derived), not the validation split that drives
early stopping. The test windows are cut at dilation 24 (nine 5-frame windows per 317-frame
trajectory) — which is why the Sim→Sim video below has nine frames.

**Temporal window.** Every sample is a `clip_len` = **5** frame window; the GNN predicts every
frame in it (deep supervision) but only the **central** frame is scored (`dataset.py::get_mask()`,
applied in `segmentation_gnn.shared_step()` for val/test). `DIFFTACTILE_CLIP_LEN` overrides
the window for one process; `./ablation_clip_len.sh [--seeds N]` trains A-to-B at
{1, 3, 5, 7} and writes `CLIP_LEN_ABLATION.md` (Table 3; 5 won on foreground IoU 0.238 vs
0.214 for 7).

---

## Domain adaptation: calibrating the digital twin

> 📄 **Fig. 4** and the marker-alignment MAE (0.50 mm at deepest contact).

The premise only holds if the simulated markers move like the real ones. `./domain_adaptation.sh`
fits the sensor's Young's modulus and the sensor↔vessel contact stiffness by one **joint
Bayesian optimisation**: every iteration simulates **two slides** at the proposed parameters —
vessel-**absent**, scored by marker MAE against the real photograph (fidelity), and
vessel-**present**, scored by how far the vessel holds the sensor up (sensitivity) — and
maximises one objective that trades the two off (`main.py::domain_adaptation_joint`). The
winning configuration is then **validated, not searched**, on the four canonical interactions —
press, twist about *z*, twist about *x*, slide — against one photograph of the real sensor at
each interaction's apex. (An older two-stage design — a vessel-absent search over the sensor
material, then a vessel-present search at the sensor it chose — is obsolete and has been
removed; `DIFFTACTILE_DA_MODE` accepts only `joint`.)

```bash
./domain_adaptation.sh                                # ~35 s per iteration; DIFFTACTILE_BO_JOINT_ITERATIONS=N (default 10),
                                                      # DIFFTACTILE_BO_JOINT_RANDOM=M random iterations first (default 5)
./alignment_figures.sh                                # Fig. 4: white-background panels + MAE, from the
                                                      # published run (or the current parameters)
./score_params.sh                                     # score the CURRENT system-params.json once, no search
./record_da_trajectories.sh                           # record the four interactions to .mp4 (needs a display)
```

Every DA run gets its own `difftactile/output/domain_adaptation/<timestamp>/`
(`bo_joint_results.json`, `iteration_log.csv`, `final_joint_validation.json`,
`da_overlay_<name>.png` on the photograph, `snapshots/` when a display is available). The
published run is `difftactile/output/domain_adaptation_published/joint_bo/` (in the bundle);
`alignment_figures.sh` redraws Fig. 4 from its cached marker positions and reports MAEs
11.9 / 15.8 / 14.5 / 12.4 px (press / twist-z / twist-x / slide; ~55 px = 2 mm marker
spacing). **Adopting a result is manual, deliberately** — nothing writes back into
`system-params.json`.

Details worth knowing: both searched parameters are log-scaled, one decade each — sensor
Young's modulus in [1e5, 1e6] Pa, sensor↔vessel normal stiffness in [1e4, 1e5] (`bo_gp.py`,
`BoGp.__init__` explains the bounds and the CFL limit behind them); a diverged parameter set is
scored as the worst value rather than
aborting; the earlier *differentiable* fit through Taichi was abandoned and its machinery
removed — BO treats the simulator as a black box. The five reference photographs live in
`difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/` (keys `da_press`,
`da_twist_z`, `da_twist_x`, `da_slide`, `flat_sensor_default_state` in `system-params.json`;
the filename encodes press depth, angle and slide length) and are turned into
`difftactile/output/da_<name>.npz` marker positions automatically. They are unrelated to the
Silicone *training* videos.

Reading the simulator window: the **sensor is green**, the **phantom blue**, the **vessel
yellow** (drawn only when its contact pair is enabled). This is a different convention from
the Fig. 4 overlays, where red = simulated and green = real markers.

---

## Top-view vessel map

> 📄 **Fig. 7** and the lower half of **Table 4**.

Projects the per-marker predictions through the sensor pose onto the phantom plane at
**1 mm per pixel** and renders the top view against the ground truth, for every configuration:

```bash
./vessel_map.sh A-to-B                          # Sim→Silicone, truth reprojected from the annotated video
./vessel_map.sh A-to-B --ground-truth photo     # Sim→Silicone, truth from the phantom's top-view photo
./vessel_map.sh C-to-B                          # Meat→Silicone
./vessel_map.sh A-to-C                          # Sim→Meat: one map per meat trial
./vessel_map.sh A-to-A                          # Sim→Sim: one simulated slide with recorded poses
./vessel_map.sh A-to-A --test-trajectories      # Sim→Sim: the ten trajectories of the prediction video, one map each
./vessel_map_all.sh                             # all six, in one go
./website_vessel_maps.sh                        # the project page's 22 maps -> docs/images/vessel_maps/ (lossless WebP)
./vessel_map.sh A-to-B --model legacy           # pre-2026-08-15 checkpoint (see Legacy models)
./vessel_map.sh A-to-B --threshold 0.6          # override the chosen decision threshold
```

- **Model:** the best-of-five instance by AP (default) or `--model legacy`.
- **Sim→Sim test set:** by default the one dedicated slide (the manuscript's map). With
  `--test-trajectories` the ten held-out trajectories of the project page's Sim→Sim prediction
  video — the same ten, in the same order — each mapped separately. The published dataset
  records no sensor pose, so `./vessel_map_sim_test_trajectories.sh` (~15 min GPU) first
  re-simulates exactly those ten by replaying the dataset's seed, verifies each against the
  published file, and stores the published markers with the re-simulated poses under
  `difftactile/output/vessel_map_sim/test_trajectories/` (shipped in the bundle).
- **Ground truth:** `video` (default) reprojects the test data's own per-marker labels
  (manual annotation on Silicone, kinematics-derived on Meat, the simulator's vessel projection
  on Sim); `photo` (Silicone only) segments the phantom's top-view photograph. Video-vs-photo
  truth IoU is 0.29, which bounds what any model can score against either.
- **The decision threshold is chosen, not assumed:** the one that keeps pixel-level precision
  ≥ 0.9 (a predicted pixel counts within 3 mm of a true pixel) while maximising recall — few
  false alarms over high sensitivity, since a false vessel misdirects a needle while a missed
  one costs a re-scan. If unreachable the run falls back to the F1-optimal threshold and says so
  (measured: Sim→Meat). `--threshold` / `DIFFTACTILE_VESSEL_MAP_THRESHOLD` override.
- **Output is versioned** under `difftactile/output/vessel_maps/<train>-to-<test>_gt-<source>/<timestamp>/`
  and never overwritten: `report.md`, `run.json`, `threshold_selection.png`, and per map
  `prediction.png`, `ground_truth.png`, `confusion_rNN.{png,pdf}` with the truth grown by
  0/1/2 mm, `l2_distances_rNN.png`, `metrics_by_radius.md` (TP FP FN TN MCC F1 precision
  recall accuracy, FG/BG IoU, AP, L2 mean/median/deciles).
- **Colours** (`Visualisation.CONFUSION_COLOURS_RGB`, shared with the viewer): 🟩 green both
  say vessel, 🟥 **red = a miss** (truth says vessel, prediction does not), 🟦 blue = a false
  alarm, ⬛ black neither. Red for misses is deliberate — the missed vessel is the dangerous
  error in palpation.
- **Geometry.** Silicone: the published 180 × 100 mm workspace, plane 16 mm from the lens
  (pressed 3 mm in). Meat: each trial one straight slide along the robot's −y, sensor assumed
  undeformed on the surface (plane 19 mm), all ten maps on one grid. Sim: the recorded
  per-frame pose of one dedicated vessel-present slide (`./vessel_map_sim_trajectory.sh`, seed
  2026, ~2 min; shipped in the bundle) — the published dataset records no poses — with the
  simulator's ×5 length scale (`meta.distance_scaling_factor`).

The **upper half of Table 4** — the same statistics per marker in video-frame space, pooled once
over all central frames of the best-of-five models — is `python -m difftactile.scripts.script_frame_space_metrics`
→ `FRAME_SPACE_METRICS.md`.

### Legacy models

`saved_models_legacy/{sim,meat}/` are the checkpoints (clip_len 7, previous unseeded simulated
dataset) of the version first accepted for publication. They are kept only because they made
that version's vessel-map figure and table; **every current result uses the five-seed sweep at
clip_len 5**. `--model legacy` / `--legacy` load them (they set `DIFFTACTILE_CLIP_LEN=7`); they
cannot be retrained. See `saved_models_legacy/README.md`.

---

## Inspect predictions frame by frame

An interactive viewer over the test set of any configuration: five panels in one Qt window —
Ground Truth, Hard Prediction, Confusion (green both / **red miss** / blue false alarm / grey
neither), Soft Prediction, Metadata — using the best-of-five model by default.

```bash
# from the docker/ directory on the HOST - it execs into the container itself
./view_predictions.sh A-to-B                    # central frames (default)
./view_predictions.sh A-to-C --all              # every frame of every window (debugging view)
./view_predictions.sh A-to-A --trials interleaved:7:3       # 7 vessel-present + 3 vessel-absent held-out trajectories
./view_predictions.sh A-to-A --trials random:10             # ten random held-out simulated trajectories
./view_predictions.sh C-to-B --retrained        # a locally trained model
./view_predictions.sh C-to-B --sweep 20260815-194045 --seed 1   # one seed of a sweep
./view_predictions.sh A-to-B --record out.mp4   # record instead of opening a window
./view_predictions.sh A-to-B --x11              # force X11 instead of Wayland (choppy)
```

| Keys | `--central` (default) | `--all` |
|---|---|---|
| `i` / `o` | previous / next trial | previous / next trial |
| `j` / `k` | previous / next **central frame** | previous / next **clip** |
| `n` / `m` | — | previous / next frame within the clip |
| `q` | quit | quit |

`--central` shows exactly what is scored (one prediction per sliding window, its centre; the
first and last `clip_len // 2` frames of a trial have none). `--all` shows every off-centre
prediction too, on sequential clips. Trials are meat trial directories, silicone videos or
simulated trajectory files; `--trials` takes comma-separated trial-id substrings,
`first-vessel-present`, `random:N` (fixed seed) or `interleaved:P:A` (P vessel-present + A
vessel-absent trials, fixed seed, shown `a a b a a b …`; the project-page Sim→Sim video is
`interleaved:7:3`, and its bird's-eye maps use the identical selection). Trials play in the order
the selection lists them. `--record PATH` steps through everything automatically (one key press
per 500 ms of video), rendered offscreen. The Hard Prediction / Confusion panels use
`MAP_DECISION_THRESHOLD` (0.58, `DIFFTACTILE_MAP_THRESHOLD`) — a display choice only; no
reported metric depends on it. **The viewer shows one model, never an average over seeds**: an
ensemble is a different model whose numbers no table here reports.

---

## Real datasets: annotate, review, preprocess

> 📄 **Fig. 5.**

The two annotation viewers are **Qt (PySide6) applications that run on bare metal by design**,
in a small dedicated environment — hand-driven frame by frame, they need to be responsive, and
they need no Taichi, CUDA or torch:

```bash
micromamba env create -f requirements/annotator-env.yml    # once, ~500 MB

# from the docker/ directory on the HOST
./annotate_data_bare_metal.sh --silicone      # click up to 4 vessel points per frame; g unlocks
                                              # editing, click a point + Delete removes it, z undo,
                                              # d clear frame, m/n video, k/j frame, p save,
                                              # q save+quit, x x quit without saving
./annotate_data_bare_metal.sh --meat          # review-only: red = vessel present, green = absent
                                              # (labels are derived from robot kinematics + straw
                                              # geometry); m/n trial, k/j frame, q quit
./annotate_data_bare_metal.sh --meat --record out.mp4     # record offscreen instead
```

Both are native Wayland clients (PySide6 bundles the Wayland plugin; `QT_QPA_PLATFORM=xcb`
falls back to X11) and open in the container too via `./annotate_data_docker.sh --meat [--x11]`
(host-launched; a debugging twin — the container's Wayland path is as smooth as bare metal, its
X11 path is not). `--meat` works from the bundle (`clean/<trial>/frames.mp4` + `marker_labels.npz`);
`--silicone` needs the dilated videos and annotation pickles, which are **not** in the bundle
(`--source DIR`, see `data/MANIFEST.md`).

**Preprocessing from raw** (only to redo the shipped datasets):

- **Meat** — `python -m difftactile.scripts.script_preprocess_meat_data`: interpolates robot
  poses onto frames, detects and Hungarian-reorders markers, projects the straw geometry from
  [`meat_experiment_spec.md`](difftactile/manual_or_experimental_data/meat_experiment_spec.md)
  through the fisheye model into a label per marker → `clean/<description>-<timestamp>/{marker_positions,marker_labels}.npz`.
  Ten of the 23 recorded trials ship (the rest were repeats no split used); the timestamp is
  the trial's identity, the description only a label. It runs from the shipped `frames.mp4`
  (regenerating, not bit-reproducing: median 0.03 px marker shift); `DIFFTACTILE_MEAT_FROM_RAW=1`
  forces the 1.6 GB raw archive.
- **Silicone** — `python -m difftactile.scripts.script_preprocess_silicone_data`: a chain of
  directory-to-directory stages (interpolate/trim → dilate → markers → reorder → annotate →
  line points → merge → dense labels) ending at the `_dense` directory `files.exp_data_silicone`
  points to. The stages are a commented menu in `preprocess_silicone_data.main()` — uncomment
  the ones you need, in order.
- The viewers as bare modules: `DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_annotate_silicone`
  / `script_browse_meat_annotations`.

---

## Regenerate the simulated dataset

The Sim dataset ships in the bundle; run this only to extend the project.

```bash
./run_pipeline.sh sim-short                     # 1 loop (8 trials, ~2-3 min): does the simulator work?
DIFFTACTILE_TRAJECTORIES=3 DIFFTACTILE_VEIN_PAIR=1 DIFFTACTILE_NUM_LOOPS=250 \
    ./run_pipeline.sh sim-full                  # the published shape: 500 slides, half with the vessel
                                                # (seed 42; ~5.5 h on an RTX 3080)

# then Hungarian-reorder into the base-graph order the GNN expects
DIFFTACTILE_SIM_RAW_DIR=difftactile/output/training_data/pickle_<timestamp> \
    python -m difftactile.scripts.script_pre_process_sim_data
# and point files.sim_data in system-params.json at the new ..._reordered_dense directory
```

`run_pipeline.sh sim-*` runs `script_apply_scaling` → `script_pre_main` → `script_main` as
separate processes (`difftactile/scripts/run_all.sh` does the same outside Docker; a single
process would load the config before `apply_scaling` rewrites it). Trajectory types are
0 press, 1 twist about *z*, 2 twist about *x*, 3 slide (`DIFFTACTILE_TRAJECTORIES`; the same
four interactions `generate_trajectories()` builds for domain adaptation); the published
dataset is entirely type 3 with `DIFFTACTILE_VEIN_PAIR=1` (each loop's first substep with the
sensor↔vessel contact pair, so half the trajectories are vessel-present). Collection is seeded,
so the same invocation regenerates the same dataset up to GPU nondeterminism. Raw output is
`difftactile/output/training_data/pickle_<timestamp>/trajectory_XXXX.npz` (`markers`,
`markers_mask`, `vein_polyline`, `vein_classification`, …).

Mesh generation (`script_generate_vitactip_mesh_gmsh`, `script_generate_vein_mesh_gmsh`) and
the sensor-geometry artifacts the simulator loads at start-up ship in the bundle too.

> The simulator opens Taichi GGUI windows when a display is available and **may segfault at
> interpreter exit** (code 139) in GGUI teardown, *after* printing `all done` and writing
> everything — the output is complete. `DIFFTACTILE_HEADLESS=1` avoids it (and is ~35% faster);
> it is implied when `DISPLAY` is unset.

---

## Demonstration videos

All videos live in **[`videos/`](videos/)** (H.264 `.mp4`, ~22 MB in total), made by
`./docker/record_videos.sh` (host-launched) at the parameters in `system-params.json` with the
best-of-five checkpoints; the GUI tools are stepped automatically, one key press per 500 ms of
video, and rendered offscreen (see "Record mode" in `difftactile/main/qt_viewer.py`).

**Simulator** — the four sensor–phantom interactions (sensor **green**, phantom **blue**,
vessel **yellow**), from the simulator's own camera (`record_da_trajectories.sh`):

| | | |
|---|---|---|
| **Press** — [`sim_press.mp4`](videos/sim_press.mp4) | **Twist about x** — [`sim_twist_x.mp4`](videos/sim_twist_x.mp4) | **Twist about z** — [`sim_twist_z.mp4`](videos/sim_twist_z.mp4) |
| **Slide, blood vessel absent** — [`sim_slide_vessel_absent.mp4`](videos/sim_slide_vessel_absent.mp4) | **Slide, blood vessel present** — [`sim_slide_vessel_present.mp4`](videos/sim_slide_vessel_present.mp4) | |

**Domain randomisation** — one vessel-present slide per randomised configuration
(`./docker/record_domain_randomisation_videos.sh`): the slide heading at −15° / 0° / +15° from a
top-down camera (`DIFFTACTILE_CAMERA_VIEW=top`, +y up the image so the slide runs bottom → top)
with both contact coefficients at their midpoint (2.75 × 10⁴, 50), and the 2 × 2 grid of
sensor–vessel normal stiffness {5 × 10³, 5 × 10⁴} × normal damping {0, 100} at heading 0° from
the side view. The three pins — `DIFFTACTILE_SLIDE_HEADING_DEG`, `DIFFTACTILE_VEIN_NORMAL_STIFFNESS`,
`DIFFTACTILE_VEIN_NORMAL_DAMPING` — are read by `record_da_trajectories_main()` only:

| | | |
|---|---|---|
| [`dr_heading_m15.mp4`](videos/dr_heading_m15.mp4) | [`dr_heading_0.mp4`](videos/dr_heading_0.mp4) | [`dr_heading_p15.mp4`](videos/dr_heading_p15.mp4) |
| [`dr_kn5e3_cn0.mp4`](videos/dr_kn5e3_cn0.mp4) | [`dr_kn5e3_cn100.mp4`](videos/dr_kn5e3_cn100.mp4) | |
| [`dr_kn5e4_cn0.mp4`](videos/dr_kn5e4_cn0.mp4) | [`dr_kn5e4_cn100.mp4`](videos/dr_kn5e4_cn100.mp4) | |

**Annotated datasets** — every frame of every recording, stepped through with the annotation
viewers. Silicone: the clicked vessel points. Meat: per-marker labels (red = vessel present,
green = absent):

| | |
|---|---|
| **Silicone phantom** — [`dataset_annotations_silicone.mp4`](videos/dataset_annotations_silicone.mp4) | **Meat phantom** — [`dataset_annotations_meat.mp4`](videos/dataset_annotations_meat.mp4) |

**Per-frame GNN predictions** — the prediction viewer on each configuration's best-of-five
model. Panels: ground truth, hard prediction, confusion (green = both say vessel, red = missed
vessel, blue = false alarm, grey = neither), soft prediction, metadata. Central frames of every
trial for the real datasets; for Sim→Sim ten held-out trajectories (nine dilation-24 windows
each): seven vessel-present and three vessel-absent, each drawn at random with a fixed seed and
interleaved present, present, absent, … — the same ten, in the same order, as the project page's
Sim→Sim bird's-eye maps:

| | |
|---|---|
| **Sim → Sim** — [`predictions_sim_to_sim.mp4`](videos/predictions_sim_to_sim.mp4) | **Sim → Silicone** — [`predictions_sim_to_silicone.mp4`](videos/predictions_sim_to_silicone.mp4) |
| **Sim → Meat** — [`predictions_sim_to_meat.mp4`](videos/predictions_sim_to_meat.mp4) | **Meat → Silicone** — [`predictions_meat_to_silicone.mp4`](videos/predictions_meat_to_silicone.mp4) |

Each link opens the file in GitHub's own video viewer; all videos are also embedded on the
[project page](https://piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI/).
The robot-control repository carries a video of the real data collection.

---

## Reference

### Data: the Zenodo bundle

**No dataset or checkpoint is in git** (`.gitignore` excludes `*.npz`, `*.pkl`, `*.pt`,
`*.mp4` except `videos/`, `output/`, `saved_models*/`, `logs/`). Everything comes from the
bundle, DOI [10.5281/zenodo.21958107](https://doi.org/10.5281/zenodo.21958107):

```bash
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz   # unpack into place (~275 MB)
./data/restore_data.sh --verify                               # what is present / missing
```

It restores the simulated dataset, the Silicone and Meat datasets, the published checkpoints
and their test-loader pickles, the whole five-seed sweep, the legacy models, the posed Sim→Sim
slide, the sensor-geometry files the simulator loads at start-up, the system-identification
marker tracks, the published domain-adaptation run, and `difftactile/output/manuscript_artifacts/`
(the manuscript's figures and tables, with a README mapping each to its source — the one
"generable but shipped" exception). Raw videos, intermediate preprocessing stages, superseded
runs and training logs are deliberately excluded — [`data/MANIFEST.md`](data/MANIFEST.md) lists
both sides. `restore_data.sh` takes any local tarball or unpacked directory, so it never needs
Zenodo itself. Authors rebuild the bundle with `./data/make_data_bundle.sh` (bare metal; refuses
to write an incomplete archive) — see `data/ZENODO_UPLOAD.md`.

Without the bundle most entrypoints raise `FileNotFoundError` — including `script_main`, which
loads marker tracks and geometry at construction. Paths resolve against the repository root
(`difftactile/main/paths.py`, override `DIFFTACTILE_ROOT`), so scripts run from any directory.

### Configuration

There are **no command-line flags** beyond the ones documented above; behaviour comes from:

1. **`difftactile/system_params/system-params.json`** — geometry, material and contact
   parameters, trajectories, the `gnn` hyperparameter block, and every path (`files.*`),
   reached in code as `SYSTEM_PARAMS.gnn.clip_len` etc. It is **partly regenerated** by
   `script_apply_scaling` from `system-params-distances.json` (SI metres) and
   `system-params-youngs-modulus.json` — edit *those* for lengths and stiffnesses.
   `system-params-computed.json` is generated by `pre_main.py`; `system-params-units.json` and
   `system-params-literature-values.json` are documentation only; `bo-gp.json` holds BO
   parameters (inert while `meta.load_params_from_bo = 0`).
2. **`RUN_ON_LAB_MACHINE`** in `difftactile/main/main.py` — Taichi on `ti.cuda` (True) or `ti.cpu`.
3. **Environment variables**:

| Variable | Effect |
|---|---|
| `DIFFTACTILE_ROOT` / `DIFFTACTILE_DATA_ROOT` | Repository root for path resolution / keep the real meat trials outside the repository. |
| `DIFFTACTILE_HEADLESS=1` | Create no windows (implied when `DISPLAY` and `WAYLAND_DISPLAY` are unset). |
| `DIFFTACTILE_INTERACTIVE=1` | Restore blocking windows (`plt.show()`, viewers, Gmsh FLTK, the tkinter labeller). Off by default: **no script ever waits for a window**; figures go to `difftactile/output/`. |
| `DIFFTACTILE_SEED` | Seed for every RNG of a training, collection or DA run (default 42). |
| `DIFFTACTILE_NUM_LOOPS`, `DIFFTACTILE_TRAJECTORIES`, `DIFFTACTILE_VEIN_PAIR` | Simulator collection: loops (each = 2 substeps × trajectory types), types to collect, vessel on the first substep. |
| `DIFFTACTILE_SCENARIO`, `DIFFTACTILE_MODE`, `DIFFTACTILE_OVERWRITE_PUBLISHED` | Configuration / `train`\|`eval` when not given as arguments; let training overwrite published checkpoints. |
| `DIFFTACTILE_CLIP_LEN` | Temporal window for one process (positive odd integer; the ablation uses it). |
| `DIFFTACTILE_ARTIFACT_DIR` | Redirect a training run's checkpoint + pickle (the seed sweep uses it). |
| `DIFFTACTILE_MAP_THRESHOLD` / `DIFFTACTILE_VESSEL_MAP_THRESHOLD` | Viewer hard-prediction cut (display only) / vessel-map operating point. |
| `DIFFTACTILE_MAP_CONFIG`, `_GT`, `_MODEL`, `_SEED` | What `vessel_map.sh` passes to `script_vessel_map`. |
| `DIFFTACTILE_VIEW_TRIALS`, `DIFFTACTILE_RECORD_MP4`, `DIFFTACTILE_RECORD_INTERVAL_MS`, `DIFFTACTILE_RECORD_SIZE` | Viewer trial selection; record mode (path, ms of video per key press, window size). |
| `DIFFTACTILE_BO_JOINT_ITERATIONS`, `DIFFTACTILE_BO_JOINT_RANDOM`, `DIFFTACTILE_VEIN`, `DIFFTACTILE_RECORD_TRAJECTORIES`, `DIFFTACTILE_DA_MAX_TIMESTEPS` | Domain adaptation (iterations; random iterations before the acquisition function takes over) / DA recording controls. |
| `DIFFTACTILE_SIM_RAW_DIR`, `DIFFTACTILE_MEAT_FROM_RAW` | Preprocessing inputs. |
| `DIFFTACTILE_ANNOTATOR_PYTHON`, `QT_QPA_PLATFORM` | Interpreter for the bare-metal annotator; force a Qt platform (`xcb`). |

### Repository layout

```
difftactile/
├── main/                    Taichi FEM simulation core: main.py (contact, collection, DA, recording),
│                            pre_main.py, apply_scaling.py, seeding.py, paths.py, display.py, qt_viewer.py
├── sensor_model/            ViTacTip FEM model + fisheye camera projection
├── object_model/            phantom, vessel, mesh loading
├── cnn/                     the GNN: segmentation_gnn.py (the configurations), gnn.py, dataset.py,
│                            seed_sweep.py, clip_len_ablation.py, model_selection.py, curve_plots.py,
│                            frame_space_metrics.py, visualise.py (the prediction viewer)
├── data_analysis/experiment/  real-sensor data: preprocessing, marker tracking, annotation,
│                            calibration, bo_gp.py, vessel_map.py, alignment_figures.py
├── scripts/                 script_<name>.py entrypoint wrappers (each imports a main() and calls it)
├── system_params/           JSON configuration
├── meshes/, manual_or_experimental_data/   STL geometry; reference photos, calibration, specs
docker/                      Dockerfile and every user-facing shell entrypoint
data/                        restore_data.sh, make_data_bundle.sh, MANIFEST.md, ZENODO_UPLOAD.md
videos/                      the demonstration videos
requirements/                dependency scripts; annotator-env.yml
```

Result tables at the root: `AUROC_RESULTS.md`, `CLIP_LEN_ABLATION.md`, `FRAME_SPACE_METRICS.md`.
`CLAUDE.md` holds the detailed engineering notes (design decisions, pitfalls, and why things are
the way they are).

### Running outside Docker

Not supported, but possible: Python 3.10–3.12, `pip install uv && pip install -e .`, then
`bash requirements/install_dependencies_difftactile.sh` (Taichi + ML stack; pins CUDA 12.6
wheels and PyG extensions against `torch-2.8.0+cu126`, no lockfile). Set `RUN_ON_LAB_MACHINE = False`
to run the **simulator** on CPU (slowly); the **GNN has no CPU path** —
`cnn/segmentation_gnn.py` allocates on a hardcoded `cuda:0`, so even evaluation needs a GPU.
The container also needs restarting (`docker stop vessel-palpation && ./docker-run.sh`) to pick
up changes to `docker-run.sh` (mounts, ulimits, the Wayland socket); it is started with `--rm`,
so keep work in the bind-mounted repository.

### Modelling decisions to know about

- **Contact compliance is deliberately asymmetric.** The sensor↔phantom pair transfers very
  little deformation; visible sensor deformation is driven almost entirely by the
  sensor↔vessel pair. The simulator is a targeted model of the *inclusion's mechanical
  signature on the marker field* — the quantity the GNN consumes and the one validated against
  the real sensor (Fig. 4) — not a general soft-body contact solver. Treat absolute contact
  forces and phantom deformation as uncalibrated.
- **The phantom is kinematically pinned** (every particle's `is_fixed` flag is set in
  `Phantom.g2p()`), which avoids the MPM collapse and jitter a free phantom showed and does not
  affect the learned quantity.
- **The BO searches only the sensor's Young's modulus (in [1e5, 1e6] Pa) and the
  sensor↔vessel normal stiffness (in [1e4, 1e5]).** Softer sensors (≤ 1e5) and stiffer ones
  (≳ 1.5e6) both diverged, so the box is the surviving decade; the sensor's Poisson's ratio is
  not searched and is held at its `system-params.json` value. `bo_gp.py` records the CFL
  reasoning (a stiffer or more incompressible sensor raises the wave speed at the fixed
  timestep), and the section below lists what else is fixed, fitted or randomised.

### Domain adaptation vs domain randomisation: what is fitted, fixed and randomised

Domain adaptation (`domain_adaptation.sh`, `main.py::domain_adaptation_joint`) *fits* two
simulator parameters against the real sensor; dataset collection (`run_pipeline.sh sim-*`,
`main.py::collect_training_data`) then *randomises* a different pair per trial. Both drive the
same simulator, the same `generate_trajectories()` and the same config, so everything not
listed as fitted or randomised is identical in the two. Lengths are physical (the simulator's
length scale is ×5). World frame: +z is up (the phantom's top-surface normal; gravity is −z);
+x runs along the vessel's axis; +y is the slide direction across the vessel.

| Parameter / setting | Domain adaptation (joint BO) | Domain randomisation (dataset collection) |
|---|---|---|
| 🟰 **Trajectory (waypoints)** | 🟰 Per BO iteration: the **slide** twice — vessel-absent and vessel-present. Final validation (Fig. 4): **press** (phantom centre, 4 mm below the surface), **twist about z** (4 mm, then spin 30°), **twist about x** (off-centre, 2 mm, then tilt 20°), **slide** — all vessel-free. | 🟰 The types named by `DIFFTACTILE_TRAJECTORIES` (0 press, 1 twist about z, 2 twist about x, 3 slide — the same waypoint builders). The published dataset is **slide only**. |
| 🟰 **The slide itself** | 🟰 Same builder in both: two waypoints at a fixed height **3 mm below the phantom's top surface** (no descent phase, no press-depth randomisation); start 30 mm before the phantom centre, end 20 mm past it on the far side, 50 mm of travel, crossing the vessel roughly at right angles. | 🟰 Identical. |
| 🎲 **Slide heading (random)** | 🎲 Drawn **once per run** from `NP_RNG`: −90° ± U(−15°, 15°) about +z, i.e. motion roughly along +y with up to ±15° of yaw of the *path*; every iteration and the validation replay that one heading (seeded, default 42). | 🎲 Re-drawn from the same distribution for **every loop × substep**, so trials differ in heading. |
| 🛑 **Sensor rotation about world +x, +y, +z (random)** | 🛑 **None.** The sensor is held at the configured pose — pointing straight down (180° about +y) and spun by the fixed camera yaw of −10.37° about +z — for the whole slide. | 🛑 **None** (the only rotation draws in the code sit in unreachable helpers and an `if False` block). Same fixed pose. |
| 🟰 **Vessel position / orientation in the phantom** | 🟰 Fixed: centreline along +x at mid-y, **3 mm beneath the top surface** (radius 2 mm, 40 mm long, rigid). | 🟰 Identical (vessel placement randomisation is disabled: `generate_random_state_dicts()` returns `[]`). |
| 🟰 **Vessel present?** | 🟰 Both per iteration (absent → fidelity, present → sensitivity); validation vessel-free. | 🟰 Alternates per **substep** with `DIFFTACTILE_VEIN_PAIR=1` (substep 0 present, substep 1 absent); without it all trials are vessel-free. |
| ❗ **Sensor Young's modulus** | 📈 **Fitted**, log-scaled in [1e5, 1e6] Pa → adopted 881 400 Pa. | 🟰 Fixed at the adopted 881 400 Pa (`vitactip.single_material`). |
| 🟰 **Sensor Poisson's ratio** | 🟰 Fixed at the config value (0.497; not searched). | 🟰 Identical. |
| 🛑 **Phantom Young's modulus / Poisson's ratio** | 🛑 Inert: the phantom is kinematically pinned and takes part in no enabled contact pair, so its material has no effect. | 🛑 Inert, same reason. |
| 🛑 **Vessel Young's modulus / Poisson's ratio** | 🛑 None — the vessel is a rigid signed-distance body with no material parameters. | 🛑 Identical. |
| 🛑 **Sensor↔phantom contact pair** | 🛑 **Disabled** (pair 0; `enable_phantom_contact_pair: false`). | 🛑 **Disabled.** |
| 🛑 **Phantom↔vessel contact pair** | 🛑 **Disabled** (pair 1; never resolved). | 🛑 **Disabled.** |
| ❗ **Sensor↔vessel normal stiffness** | 📈 **Fitted**, log-scaled in [1e4, 1e5] → adopted 94 908. | 🎲 **Random**, U(5e3, 5e4) per loop × substep. |
| ❗ **Sensor↔vessel normal damping** | 🟰 Fixed at 100. | 🎲 **Random**, U(0, 100) per loop × substep. |
| 🛑 **Sensor↔vessel tangential stiffness** | 🛑 0 (i.e. disabled). | 🛑 0 (i.e. disabled). |
| 🛑 **Sensor↔vessel friction coefficient** | 🛑 0 (i.e. disabled). | 🛑 0 (i.e. disabled). |

Legend — one emoji per cell: 🟰 fixed value · 🎲 drawn at random · 📈 fitted by the BO ·
🛑 disabled / zero (a fixed zero counts as disabled). The left-hand column repeats the emoji when
the two regimes agree, and shows ❗ where one regime holds a value fixed that the other varies
(fits or randomises).

How to picture the three rotation axes: a rotation about **+z** spins the sensor about its own
vertical axis — the contact footprint stays put and only the marker pattern turns (the fixed
−10.37° camera yaw, and the 30° "twist about z" validation interaction, are this); a rotation
about **+x** tilts the sensor about the vessel's axis, so it leans forward or back *along the
slide direction* (the 20° "twist about x" interaction); a rotation about **+y** would tilt it
about the slide direction, leaning *along the vessel* — no trajectory in either regime does
this. Neither regime randomises any of the three; the only pose randomness anywhere is the
slide path's heading, and it is the *same* draw (same function, same distribution) in both —
domain adaptation simply takes it once per seeded run, dataset collection once per trial.

Things the table cannot show: a 0 for tangential stiffness or friction switches that term of
the contact force off entirely (`min(k_t·v_t, μ·|F_n|)` is identically zero), so the vessel
pushes on the sensor purely normally in both regimes. Because the sensor↔phantom pair is off,
"press depth" is a commanded height of the sensor relative to the phantom surface, not a
force — the phantom never pushes back, and the vessel is the only thing the sensor touches.
The sensor's Young's modulus is the one parameter that is fitted in one regime and *held at
the fitted value* in the other; the sensor↔vessel normal stiffness is fitted in one and
*randomised over a different range* in the other — the adopted 94 908 lies **above** the
U(5e3, 5e4) the dataset samples, and the fitted-stage damping of 100 is the *top* of the
U(0, 100) the dataset samples. Those two ranges are the dataset's domain randomisation and
predate the joint BO; they were left as collected so the published dataset stays reproducible.
The [project page](https://piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI/)
shows one simulated vessel-present slide per randomised configuration (heading −15°/0°/+15°,
and the 2 × 2 grid of stiffness × damping range extremes).

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError` on a `.npz` / `.pkl` / `.pt` | Data bundle not restored — `./data/restore_data.sh --verify`. |
| `received 0 items of ancdata` in training | Container started before `docker-run.sh` raised the fd limit (`ulimit -Sn` should be 65535) — restart it. |
| Viewer says "no Wayland socket" | Container started before the socket mount existed — restart it, or `--x11`. |
| Segfault (139) after `all done` | Taichi GGUI teardown; output is complete. `DIFFTACTILE_HEADLESS=1` for a clean exit. |
| CUDA out of memory in Taichi | Lower `device_memory_GB` in `main.py` (`TI_DEVICE_MEMORY_GB`). |
| Edits to `system-params.json` revert | `script_apply_scaling` regenerates them — edit `system-params-distances.json` / `-youngs-modulus.json`. |
| Different numbers than the tables | Single seed vs five-seed mean; or a `--retrained` model instead of the published one. |

### Branches and tags

`main` is the only branch, and the only one the documentation, the image and the bundle are
tested against. The `upstream-difftactile` tag marks the pristine DiffTactile fork point
(`git diff upstream-difftactile..main --stat` shows this project's changes; the history is not
linear from it, so `git log` across it is not meaningful).

---

## Citation

If you use this repository, please cite it (and, where relevant, the companion robot-control
repository and the data bundle) — GitHub's "Cite this repository" button reads the same
metadata from `CITATION.cff`:

```bibtex
@misc{blaszyk_shallow-vessel-palpation-simulator-and-ai_2026,
  title        = {shallow-vessel-palpation-simulator-and-{AI}: a calibrated digital twin and spatio-temporal {GNN} for robot-assisted sliding palpation and shallow vessel localisation},
  author       = {Blaszyk, Piotr and Fan, Wen and Deng, Kaizhong and Elson, Daniel and Zhang, Dandan},
  year         = {2026},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.21958186},
  url          = {https://doi.org/10.5281/zenodo.21958186}
}

@misc{blaszyk_shallow-vessel-palpation-robot-control_2026,
  title        = {shallow-vessel-palpation-robot-control: {DOBOT} {Magician E6} control and synchronised tactile-video capture for robot-assisted sliding palpation},
  author       = {Blaszyk, Piotr and Fan, Wen and Deng, Kaizhong and Elson, Daniel and Zhang, Dandan},
  year         = {2026},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.21958190},
  url          = {https://doi.org/10.5281/zenodo.21958190}
}

@misc{blaszyk_shallow-vessel-palpation-dataset_2026,
  title        = {shallow-vessel-palpation-dataset},
  author       = {Blaszyk, Piotr},
  year         = {2026},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.21958107},
  url          = {https://doi.org/10.5281/zenodo.21958107}
}
```

This work builds on DiffTactile; please also cite the original simulator:

```bibtex
@inproceedings{si2024difftactile,
  title     = {{DIFFTACTILE}: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation},
  author    = {Zilin Si and Gu Zhang and Qingwei Ben and Branden Romero and Zhou Xian and Chao Liu and Chuang Gan},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=eJHnSg783t}
}
```

## License

MIT — see [LICENSE](LICENSE). Inherited from upstream DiffTactile.

## Contact

Piotr Blaszyk — please open an issue on this repository. Data and weights are on Zenodo at
[10.5281/zenodo.21958107](https://doi.org/10.5281/zenodo.21958107).
