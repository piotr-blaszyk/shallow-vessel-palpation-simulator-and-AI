# Sim-to-Real Subsurface Feature Localisation with a Soft Optical Tactile Sensor

A differentiable-simulation pipeline for locating features **hidden beneath a soft surface** —
blood vessels in a silicone phantom, or plastic straws buried under layers of steak — from the
deformation of markers on a ViTacTip optical tactile sensor.

The core question: **can a model trained entirely in simulation localise subsurface structure in
the real world?** A graph neural network is trained only on synthetic marker displacements
produced by a differentiable FEM simulation, then evaluated on video from a physical sensor
pressing on real tissue.

This is a masters project. It is a fork of
[DiffTactile](https://difftactile.github.io/) (Si et al., ICLR 2024); the upstream
manipulation tasks have been removed and the differentiable Taichi FEM core repurposed for
subsurface sensing. Upstream's original README is preserved at the `upstream-difftactile` tag.

### Project repositories

This work spans two repositories, submitted together to an **ECCV 2026 workshop** as
*"Sim-to-Real Subsurface Feature Localisation with a Soft Optical Tactile Sensor"*:

| Repository | Role |
|---|---|
| **[shallow-vessel-palpation-simulator-and-AI](https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI)** (this one) | **Main repository.** Simulation, dataset generation, GNN training and evaluation — everything needed to reproduce the published results. |
| [shallow-vessel-palpation-robot-control](https://github.com/piotr-blaszyk/shallow-vessel-palpation-robot-control) | Robot control. Drives the DOBOT Magician E6 arm that collected the real tactile recordings for both phantoms. Needed only to *gather new* data, not to reproduce results. |

Data and trained model weights are published on Zenodo as **shallow-vessel-palpation-dataset**
([10.5281/zenodo.21900934](https://doi.org/10.5281/zenodo.21900934)) — see
[Quickstart](#quickstart-docker) below.

```
   Taichi FEM simulation            Real sensor
   sensor + phantom + vein          video of pressing
            │                             │
            ▼                             ▼
   synthetic marker displacements   tracked marker displacements
            │                             │
            └──────────► GNN ◄────────────┘
                     train on sim      evaluate on real
                          │
                          ▼
                 subsurface feature map (+ ROC / IoU)
```

---

## Quickstart (Docker)

**Docker is the only officially supported way to run this repository.** The image pins the
whole stack (CUDA 12.6, Taichi, PyTorch 2.8 + PyTorch Geometric) on a single Python
interpreter, and passes the host X display through so the Taichi GGUI simulator windows work.

**Requirements:** Ubuntu 20.04/22.04/24.04/26.04, an NVIDIA GPU (≥10 GB VRAM),
the NVIDIA driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

> **Use the `main` branch** — it is the only branch, and a plain `git clone` already puts you
> there. See [Branches and tags](#branches-and-tags).

```bash
# 1. Clone (main is the only branch)
git clone https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI.git
cd shallow-vessel-palpation-simulator-and-AI

# 2. Fetch the data bundle from Zenodo (~190 MB) and unpack it into place
#    (datasets + trained checkpoints; see data/MANIFEST.md for what is inside)
#    Zenodo record "shallow-vessel-palpation-dataset", DOI 10.5281/zenodo.21900934
#    -> https://doi.org/10.5281/zenodo.21900934
wget https://zenodo.org/records/21900934/files/shallow-vessel-palpation-data.tar.gz
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz

# 3. Build the image (~10-30 min, downloads several GB) and start the container
./docker/docker-build.sh
./docker/docker-run.sh

# 4. Verify GPU, dependencies and data are all in place
docker exec -it vessel-palpation ./docker/run_pipeline.sh check
```

### Reproduce the published results

The paper's three models — one per (train → test) configuration over the simulated (**A**),
silicone (**B**) and meat (**C**) datasets — are selected **by name**, with no source editing:

```bash
# Evaluate the sim-trained GNN on the real SILICONE phantom -> ROC curve
docker exec -it vessel-palpation ./docker/run_pipeline.sh A-to-B

# Cross-domain: test the sim-trained checkpoint on real MEAT (no retraining)
docker exec -it vessel-palpation ./docker/run_pipeline.sh A-to-C

# Train on real MEAT trials, test on silicone
docker exec -it vessel-palpation ./docker/run_pipeline.sh C-to-B

# ...or all three in sequence
docker exec -it vessel-palpation ./docker/run_pipeline.sh all-scenarios
```

Each configuration also takes `--train` (reproduce the model from scratch) or `--eval` (load
the published checkpoint); see [Training and evaluating the GNN](#4-training-and-evaluating-the-gnn)
for the full table. Outputs land in `difftactile/output/` (e.g. `roc_curve_A-to-B.pdf`) and `logs/`.

Any run that **trains** writes `*_retrained_<config>` artifacts
(`final_segmentation_model_gnn_meat_retrained_C-to-B.pt`,
`test_loader_gnn_meat_retrained_C-to-B.pickle`, and the `_sim` equivalents) rather than
overwriting the published checkpoints that the evaluation paths read — otherwise running the
configurations in sequence would silently change the reported AUC. The `<config>` tag also
keeps A→B and A→C apart, since they share the same underlying `*_sim` artifact names. Pass
`DIFFTACTILE_OVERWRITE_PUBLISHED=1` if you deliberately want to replace them.

### AUROC for all six scenarios

The three configurations above, each scored from either the published checkpoint or one you
trained yourself, give six scenarios. `script_auroc_all_scenarios` measures all of them in one
pass — per **marker node across video frames**, not from a reprojected phantom map:

```bash
# score every scenario whose checkpoint is present -> AUROC_RESULTS.md
docker exec -it vessel-palpation python -m difftactile.scripts.script_auroc_all_scenarios

# published checkpoints only (no training needed)
docker exec -it vessel-palpation python -m difftactile.scripts.script_auroc_all_scenarios --pretrained
```

ROC curves are written one per scenario to `difftactile/output/roc_curves/`
(`roc_curve_A-to-B_pretrained.pdf` and so on), and the AUROC table to `AUROC_RESULTS.md`.
The `retrained` rows need a `--train` run of the matching configuration first.

### Inspect predictions frame by frame

An interactive viewer that steps through the test-set frames and shows the confusion overlay
(green TP, yellow TN, red FP, blue FN) alongside the ground truth and soft predictions. The
configuration selects **both** the model weights and the test dataset, so all six scenarios are
reachable by name:

```bash
docker exec -it vessel-palpation ./docker/view_predictions.sh A-to-B              # published checkpoint
docker exec -it vessel-palpation ./docker/view_predictions.sh A-to-C --retrained  # locally trained
```

Needs a display (the container forwards the host X session). Press `q` to quit.

### Bird's-eye vessel localisation map

Projects the per-marker predictions through the sensor pose onto the phantom surface at
1 mm/pixel and renders the top view against the ground truth:

```bash
docker exec -it vessel-palpation ./docker/vessel_map.sh
```

Writes `confusion_overlay_vein_map.png`, `segmentation_mask_predicted_aggregated.png` and
`exp_overlay_downscaled.pdf` to `difftactile/output/`. **Silicone only** — the workspace bounds
and sensor offsets are specific to that rig. Add `--cached` to reuse the probabilities from a
previous run instead of re-running inference.

### Annotate or review the real-world datasets

Manual annotation and annotation review for the two real datasets. In each, one tool does both
jobs: it loads the annotations already on disk, redraws them, and lets you step through frames.

**This is the one entrypoint that runs outside Docker.** These are hand-driven, frame-by-frame
GUI tools, and inside the container every repaint crosses a forwarded X socket, which makes
stepping through frames choppy. Run them natively instead — they need no part of the Docker
stack (no Taichi, no CUDA, no torch), just a small dedicated environment created once:

```bash
micromamba env create -f requirements/annotator-env.yml   # once, ~500 MB, about a minute

./docker/annotate_data.sh --silicone   # click annotator
./docker/annotate_data.sh --meat       # marker-label review
```

The script activates that environment itself. If you would rather use your own interpreter,
point `DIFFTACTILE_ANNOTATOR_PYTHON` at it — it needs numpy, scipy, tqdm, **PySide6** (the
windows) and **av** (video decoding); the script checks for the last two and says so if they
are missing. Note that these two viewers are the **only** part of the project that does not
draw its windows with OpenCV.

**They are Qt 6 applications, so they are native Wayland clients.** The `opencv-python` wheel
ships exactly one Qt platform plugin (`xcb`), so every OpenCV window on a Wayland desktop goes
through Xwayland; the PySide6 wheels bundle the Wayland plugins, so Qt selects `wayland` by
itself and no compatibility layer is involved. Nothing is forced — set `QT_QPA_PLATFORM=xcb`
to fall back to X11, which is what to use inside the container or over X forwarding. Because
Qt needs no X server, `DISPLAY` may be unset entirely: `WAYLAND_DISPLAY` alone is enough.

Running inside the container is no longer the recommended fallback: the image ships neither
PySide6 nor PyAV, and the script will tell you to run on the host instead.

Both need a display and set `DIFFTACTILE_INTERACTIVE=1` for you. Keys are printed on start-up
(`m`/`n` video or trial, `k`/`j` frame, `q` quit; silicone additionally: left click to add a
point, `d` to clear the frame, `p` to save).

Annotation points in the silicone tool are real Qt scene objects rather than circles burned
into the image, so **clicking a point selects it and `Delete` removes that one** — alongside
the older `z` (undo last) and `d` (clear frame). The view is scaled to fit the window while the
scene stays in the video's own 1080p pixel grid, so the window is freely resizable and clicks
map back to full-resolution coordinates exactly. (This replaces the old `DIFFTACTILE_VIEW_WIDTH`
downscaling, which existed because OpenCV had to shrink the frame itself before pushing it over
an X socket.)

The meat viewer draws the labels over the **real camera frames**: the bundle ships
`clean/<trial>/frames.mp4`, the 26 decimated frames per trial that preprocessing kept, aligned
1:1 with `marker_labels.npz`. Each trial is decoded and composited once on first visit, so
frame stepping is instant afterwards.

> `--silicone` still needs the dilated videos and annotation pickles, which are **not** in the
> bundle (they are intermediate preprocessing stages — see [`data/MANIFEST.md`](data/MANIFEST.md)).
> Pass `--source DIR` to point at a tree that has them.

### Regenerate the simulated dataset (optional)

The simulated training set ships in the Zenodo bundle, so **this is not required** to
reproduce the results. Run it only if you want to extend the project:

```bash
# ~2-3 minutes: a single loop (8 trials), to check the simulator works
docker exec -it vessel-palpation ./docker/run_pipeline.sh sim-short

# ~2 h 45 m: a full 800-trial collection run
docker exec -it vessel-palpation ./docker/run_pipeline.sh sim-full
```

> **To regenerate the *published* dataset specifically, set
> `DIFFTACTILE_TRAJECTORIES=3`.** All 500 trajectories in the shipped dataset are
> type 3 ("slide (vein)") — it was collected when the collection loop read
> `range(3, 4)`, which a later commit widened to all four types. A default run
> therefore also produces types 0/1/2, and type 0 yields empty arrays by design
> (it ends in ~36 timesteps, below the `ts > 80` recording threshold).
>
> ```bash
> docker exec -it -e DIFFTACTILE_TRAJECTORIES=3 vessel-palpation \
>     ./docker/run_pipeline.sh sim-full
> ```

Open an interactive shell with `./docker/docker-connect.sh`. GUI windows (Taichi GGUI,
the cv2 annotation tool, matplotlib) appear on your desktop automatically — the image ships
Vulkan, which GGUI requires — and `DIFFTACTILE_HEADLESS=1` suppresses them when running over
SSH or in CI.

> With the GUI enabled, Taichi may segfault **during interpreter shutdown**, after the
> simulation has already printed `all done` and written its data. This is a teardown-only
> issue in Taichi's GGUI destructor; the output is complete and unaffected. Use
> `DIFFTACTILE_HEADLESS=1` for a clean exit code in scripted runs.

Everything below documents the pipeline in detail, including how to run it outside Docker.

---

## Branches and tags

> ### 👉 `main` is the only branch.
>
> Everything is on `main`: it is the only branch, the only one the documentation describes, and
> the only one the Docker image and Zenodo bundle are tested against. All three of the paper's
> models train and evaluate from it, selected by name (see [Quickstart](#quickstart-docker)).

### The `upstream-difftactile` tag

One tag is published alongside `main`:

| Tag | Commit | What it is |
|---|---|---|
| `upstream-difftactile` | `c9b348e` | The pristine [DiffTactile](https://difftactile.github.io/) state this project forked from, before any masters-project work. Upstream's original README is preserved here. |

It marks the fork point, so you can separate inherited upstream code from the contribution of
this project:

```bash
# What this project changed relative to upstream DiffTactile
git diff upstream-difftactile..main --stat

# Browse the upstream code as it was at the fork
git checkout upstream-difftactile
```

Note that `main` does not descend linearly from this tag — the history was rewritten during
development — so `git diff` is meaningful but `git log upstream-difftactile..main` is not.
The tag is a reference point only; it is not a branch and there is no need to check it out to
run anything in this README.

---

## Setup

### Requirements

- **Python 3.10–3.12.** (Upstream DiffTactile said 3.9.16; that does not apply to this fork's
  torch 2.8 / CUDA 12.6 stack.)
- **An NVIDIA GPU with CUDA is effectively mandatory.** Developed on an RTX 3080 (10 GB);
  `difftactile/main/main.py` requests `device_memory_GB=9` from Taichi. The simulation can be
  switched to CPU (see [Running without a GPU](#running-without-a-gpu)), but the **GNN cannot** —
  `difftactile/cnn/segmentation_gnn.py:570` allocates on a hardcoded `cuda:0` with no fallback, so even
  loading a checkpoint to plot a ROC curve fails on a CPU-only machine.
- **A display is optional, and nothing ever waits for one.** No script blocks on a GUI
  window: figures are written to `difftactile/output/` and the run continues, so the
  simulator, all three GNN scenarios and the preprocessing tools finish unattended over SSH
  or in CI. Set `DIFFTACTILE_INTERACTIVE=1` to get the blocking windows back (you then close
  them by hand), or `DIFFTACTILE_HEADLESS=1` to skip creating windows altogether. See
  [Interactive windows](#interactive-windows).
- Linux (developed on Ubuntu 24.04).

### Install

```bash
# main is the only branch.
git clone https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI.git
cd shallow-vessel-palpation-simulator-and-AI

conda create -n difftactile python=3.10
conda activate difftactile

pip install uv          # the install scripts use uv
pip install -e .
```

Then install the dependencies. There are two sets — the simulator and the ML stack:

```bash
bash requirements/install_dependencies_difftactile.sh   # Taichi sim + ML
bash requirements/install_dependencies_ml.sh            # GNN stack only
```

`install_dependencies_difftactile.sh` is the superset and is enough on its own for most work.

> **Notes on the dependency scripts.** They carry a `#!/bin/zsh` shebang, so invoke them with
> `bash` (as above) or install zsh. They pin CUDA 12.6 wheels and PyTorch Geometric extensions
> built against `torch-2.8.0+cu126` — if you use a different CUDA version, edit the
> `--index-url` and `-f` lines to match. There is **no lockfile**, so exact versions are not
> reproducible; `requirements/requirements_ml.in` records the intended set.
> Two transitively-required packages are not listed explicitly: install them if imports fail:
> ```bash
> uv pip install scipy seaborn
> ```

### Always run from the repository root

Configuration is loaded with a path relative to the working directory
(`difftactile/main/constants.py` reads `difftactile/system_params/system-params.json`).
Running from anywhere else fails immediately. Every command below assumes you are at the repo
root and invokes modules with `python -m`.

---

## Configuration

There are **no command-line flags anywhere in this project.** Three mechanisms control behaviour:

1. **`difftactile/system_params/system-params.json`** — the runtime source of truth for geometry,
   material properties, trajectories, GNN hyperparameters, and all input/output paths.
   Accessed in code as `SYSTEM_PARAMS.gnn.batch_size`, `SYSTEM_PARAMS.files.dataset_root`, …

   The other JSON files in that directory form a small hierarchy:

   | File | Role |
   |---|---|
   | `system-params-distances.json` | **Edit this** for any length. SI metres, unscaled; multiplied by `meta.distance_scaling_factor` into `system-params.json` by `apply_scaling`. |
   | `system-params-youngs-modulus.json` | **Edit this** for stiffnesses; same mechanism. |
   | `system-params.json` | Working config. Partly **regenerated** by `apply_scaling` — hand edits to scaled keys are lost. |
   | `system-params-computed.json` | **Generated** by `pre_main.py` (poses, MPM grid layout). Never hand-edit. |
   | `bo-gp.json` | Best Bayesian-optimisation parameters (inert while `meta.load_params_from_bo = 0`). |
   | `system-params-units.json`, `system-params-literature-values.json` | Documentation only — no code reads them. Useful provenance for the material parameters. |
2. **Module-level constants** — notably `RUN_ON_LAB_MACHINE` at `difftactile/main/main.py:30`,
   which switches Taichi between `ti.cuda` and `ti.cpu`.
3. **Commented-out call lists** — several `main()` functions are a menu of pipeline steps that
   you enable by uncommenting. `difftactile/data_analysis/experiment/preprocess_silicone_data.py` is the clearest
   case; `difftactile/scripts/script_segmentation_gnn.py` toggles between training and evaluation.

To change a parameter, prefer editing the JSON.

---

## Reproducibility — read before running

**The datasets and trained model weights are not in this repository** — they are large binary
artifacts excluded by `.gitignore`. **They are published on Zenodo instead**; see
[the fix](#the-fix-restore-the-published-data-bundle) immediately below the table.

A fresh clone does **not** contain:

| Missing | Expected at | Needed by | Regenerable? |
|---|---|---|---|
| Simulation outputs and intermediates | `difftactile/output/` | most ML and analysis steps | yes, by the sim |
| Canonical 127-marker graph | `difftactile/output/base-graph-connectivity.npz` | **every** dataset/train/eval run | **yes** — `script_base_graph_connectivity`, from an image that *is* in the repo |
| Detected initial marker positions | `difftactile/output/init-marker-positions.npz` | marker tracking, base graph | yes — `script_fisheye_model` |
| Trained GNN weights + test-loader pickle | `saved_models_meat/final_segmentation_model_gnn_meat.pt`, `difftactile/output/test_loader_gnn_meat.pickle` | evaluation / ROC (needs **both**) | only by retraining |
| Silicone dataset | `difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense` | silicone training/eval | from raw video via `script_preprocess_silicone_data` |
| Meat experiment trials | `difftactile/manual_or_experimental_data/meat_training_data/clean/` | the meat scenarios (`A-to-C`, `C-to-B`) | from raw video via `script_preprocess_meat_data` |
| Raw experiment videos + robot poses | `.../meat_training_data/raw/<id>.{avi,npz}`, `silicone_training_data/…` | all preprocessing | no — must be supplied |
| Fisheye calibration | `difftactile/output/fisheye_params.npz` | undistortion | yes, with your own checkerboard images |
| Marker-tracker trajectories | `difftactile/output/marker_tracker/domain-adaptation-vascular-markers/traj_{0..3}_out.pkl` | **`script_main` construction** | from raw video via marker tracking |

### The fix: restore the published data bundle

Everything in that table is supplied by the Zenodo archive
([10.5281/zenodo.21900934](https://doi.org/10.5281/zenodo.21900934)), so none of it has to be
regenerated:

```bash
wget https://zenodo.org/records/21900934/files/shallow-vessel-palpation-data.tar.gz
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz   # ~190 MB
./data/restore_data.sh --verify                  # list what is present / missing
```

[`data/MANIFEST.md`](data/MANIFEST.md) documents exactly what the bundle contains and what is
deliberately excluded (raw videos, intermediate preprocessing stages, training logs) — the
exclusions are what take it from 4.5 GB down to ~190 MB without affecting any published result.

Without the bundle, most entrypoints raise `FileNotFoundError`. In particular `script_main`
reads marker-tracker output (`traj_{0..3}_out.pkl`) during construction, so **even the
simulation cannot start from a bare clone**. The parts that work unaided are mesh generation
(`script_generate_*_mesh_gmsh`), `script_apply_scaling` and `script_pre_main`.

Paths are resolved against the repository root by `difftactile/main/paths.py` (override with
`DIFFTACTILE_ROOT`), so scripts run from any working directory and no absolute paths need
editing. Output directories are created on demand.

**Verified:** a clone into an empty directory, plus the bundle, reproduces all three scenarios
in Docker — see [REPRODUCTION_TEST.md](REPRODUCTION_TEST.md) for the transcript and numbers.

Results are **not bit-wise reproducible for the simulation**: `NP_RNG = np.random.default_rng()`
in `difftactile/main/constants.py` is unseeded and drives trajectory and contact-parameter
randomisation, so two runs of `script_main` produce different datasets. Seed `NP_RNG` if you
need repeatable runs. The *evaluation* scenarios are deterministic (AUC agrees to ~15
significant figures across machines).

---

## Running the pipeline

### 1. Simulation — generate synthetic training data

```bash
# scale physical units into simulation units (rewrites system-params.json in place)
python -m difftactile.scripts.script_apply_scaling

# derive poses and MPM grid layout -> writes system-params-computed.json
python -m difftactile.scripts.script_pre_main

# run the Taichi FEM simulation and collect training data
python -m difftactile.scripts.script_main
```

Or all three in order:

```bash
bash difftactile/scripts/run_all.sh   # zsh shebang; invoke with bash or install zsh
```

`run_all_loop.sh` repeats this 100× to accumulate a dataset across randomised runs.

> ⚠️ **`script_apply_scaling` rewrites `system-params.json` in place.** It multiplies the values
> in `system-params-distances.json` (SI metres) and `system-params-youngs-modulus.json` by the
> scale factors in `meta` and merges the result into `system-params.json`, overwriting whatever
> was there. **To change a geometry or material value, edit the `-distances` /
> `-youngs-modulus` file, not `system-params.json`** — otherwise your edit is silently
> discarded on the next run.
>
> ⚠️ **Run the three stages as separate processes** — that is what `run_all.sh` does. A single
> process that imports all three modules up front makes `constants.py` load `system-params.json`
> *before* `apply_scaling` rewrites it, so the simulation runs against pre-scaling constants.
> (An old `script_all.py` had exactly this bug and has been removed.)

Outputs land in `difftactile/output/`: per-trajectory training data at
`difftactile/output/training_data/pickle_<timestamp>/trajectory_XXXX.npz`, containing `markers`,
`markers_mask`, `vein_polyline`, `vein_polyline_mask`, `target_id_array` and `trajectory_type`.

`script_main` opens **two Taichi GGUI windows** unconditionally, so it needs a Vulkan-capable
display; it will not run headless without patching out `visualisation_set_up_gui()`.

**Mesh generation** (optional — regenerates sensor and vein geometry with gmsh; each opens an
interactive Gmsh window that you must close for the script to continue):

```bash
python -m difftactile.scripts.script_generate_vitactip_mesh_gmsh
python -m difftactile.scripts.script_generate_vein_mesh_gmsh
```

**System identification / calibration:**

```bash
python -m difftactile.scripts.script_fisheye_model     # detect markers -> init-marker-positions.{pkl,npz}
python -m difftactile.scripts.script_bo_gp             # Bayesian optimisation of simulation parameters
```

`difftactile/main/cfl_and_contact_params_estimation.py` (CFL timestep + Hertzian
contact-stiffness estimates) is **diagnostics only and destructive if run as an entrypoint**: it
writes contact parameters back to `system-params.json` as scalars, whereas `main.py` expects
3-element lists (one per contact pair), so running it breaks the next `script_main`. Its
`script_*` wrapper has therefore been removed; the module stays because `main.py` imports it.

### 2. Processing real sensor data

Requires the recorded videos.

```bash
python -m difftactile.scripts.script_preprocess_meat_data    # meat experiment: raw/ -> clean/
python -m difftactile.scripts.script_preprocess_silicone_data                 # silicone phantom cleaning pipeline
python -m difftactile.scripts.script_marker_tracker          # marker tracking + tkinter labelling GUI
python -m difftactile.scripts.script_domain_adaptation       # sim-vs-real marker comparison
```

**Meat (`script_preprocess_meat_data`)** — interpolates robot poses onto frames, detects and
Hungarian-reorders markers, then projects the straw geometry through the fisheye camera model to
derive a binary label per marker. Trial geometry is parsed from
[`meat_experiment_spec.md`](difftactile/manual_or_experimental_data/meat_experiment_spec.md),
which catalogues the meat trials (straw depth vs. number of 5 mm steaks above it). Produces,
per trial, `clean/<trial_id>/marker_positions.npz` `(T, 127, 2)` and `marker_labels.npz`
`(T, 127)`.

**It runs from the data bundle.** Each trial ships as `clean/<trial_id>/frames.mp4` (the 26
decimated frames) plus `frames_poses.npz` (their robot poses), so preprocessing starts from
those rather than needing the 1.6 GB raw archive. If `meat_training_data/raw/` is present, any
trial without a shipped video falls back to it — decimating full-rate 1920×1080 AVIs by 15×, as
before. Force the raw path with `DIFFTACTILE_MEAT_FROM_RAW=1`.

> Because `frames.mp4` is H.264, re-running preprocessing **regenerates** the dataset rather
> than reproducing it bit-for-bit: marker positions shift by a median 0.03 px (p99 0.47 px)
> against ~55 px marker spacing, and 16 of ~76000 labels differ (0.02%). The `*.npz` files in
> the bundle stay the authoritative artifacts. The pre-rendered `marker_labels.avi` overlays are
> no longer written by default — the annotation viewer composites the same view live from
> `frames.mp4` and the labels; set `DIFFTACTILE_MEAT_WRITE_OVERLAY=1` if you want the files.

Rebuild the shipped videos from the raw archive (author-side) with:

```bash
python -m difftactile.scripts.script_make_meat_clean_videos
```

**Silicone (`script_preprocess_silicone_data`)** — a chain of directory-to-directory stages: interpolate/trim →
dilate → extract markers → reorder → annotate (a manual cv2 click GUI) → line points → merge into
the simulation `.npz` format → add dense labels, ending at the `_dense` directory that
`exp_data_silicone` points to. Each stage is a method call in `preprocess_silicone_data.main()`. As shipped, the
whole chain is commented out and only `count_annotation_dots()` runs, because the published
`_dense` output is already in the data bundle. **Uncomment the stages you need, in order.**

**Annotation and annotation review.** The one stage of each pipeline that is a manual tool has
its own entrypoint, so it can be reached without editing the commented menu above:

```bash
DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_annotate_silicone         # click annotator
DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_browse_meat_annotations   # label review
```

Both load whatever annotations already exist and redraw them, so the same window reviews the
shipped annotations and creates new ones. `docker/annotate_data.sh --silicone|--meat` wraps them,
selects the dedicated bare-metal environment and handles staging the silicone videos, which the
bundle excludes; it is the recommended way to run these two — see
[Annotate or review the real-world datasets](#annotate-or-review-the-real-world-datasets). Note the meat labels are
derived analytically from robot kinematics and straw geometry rather than clicked, so the meat
tool is review-only.

### 3. Preparing datasets

**Run this first** — it builds the canonical 127-marker graph that *every* dataset, training run
and evaluation loads at construction time:

```bash
python -m difftactile.scripts.script_base_graph_connectivity   # -> difftactile/output/base-graph-connectivity.npz
```

Then, for simulated data:

```bash
python -m difftactile.scripts.script_pre_process_sim_data   # Hungarian-reorder sim markers into base-graph order
```

### 4. Training and evaluating the GNN

The paper uses three datasets:

| | Dataset |
|---|---|
| **A** | Simulated, collected in the differentiable tactile simulator. |
| **B** | Real **silicone** phantom — shallow veins ("easy"). |
| **C** | Real **meat** phantom — veins at varying depths ("difficult"). |

and reports three separately trained models, one per (train → test) configuration. Each is
selected **by name**, and each can be either trained from scratch or evaluated from the
published checkpoint — with **no source editing**:

```bash
# Train each of the three models from scratch
python -m difftactile.scripts.script_segmentation_gnn A-to-B --train   # train on sim,  test on silicone
python -m difftactile.scripts.script_segmentation_gnn C-to-B --train   # train on meat, test on silicone
python -m difftactile.scripts.script_segmentation_gnn A-to-C --train   # train on sim,  test on meat

# ...or reproduce the published numbers without retraining
python -m difftactile.scripts.script_segmentation_gnn A-to-B --eval    # evaluate on silicone + ROC
python -m difftactile.scripts.script_segmentation_gnn A-to-C --eval    # cross-domain, no retraining
```

| Config | Train set | Test set | `--train` | `--eval` | Default |
|---|---|---|---|---|---|
| `A-to-B` | simulation | silicone | Trains the large model on sim, tests on silicone. | Loads the published sim-trained checkpoint (`*_sim`), evaluates on silicone, writes the ROC curve (AUC 0.7314). | `--eval` |
| `C-to-B` | meat | silicone | Trains the small model on the real meat trials, tests the best checkpoint on silicone. | Same as `--train` — the published run ends by testing on silicone. | `--train` |
| `A-to-C` | simulation | meat | Trains the large model on sim, tests on meat with every trial in the test split. | Loads the published sim-trained checkpoint and tests it on meat, no retraining. | `--eval` |

Omitting the mode uses the default in the last column (evaluation wherever a published
checkpoint makes it the cheaper path). The configuration can also be given as
`DIFFTACTILE_SCENARIO` and the mode as `DIFFTACTILE_MODE`.

> The **older scenario names still work** as aliases: `sim-to-silicone` → `A-to-B`,
> `sim-to-meat` → `C-to-B`, `silicone-to-meat` → `A-to-C`. Note that `silicone-to-meat` was a
> misnomer: it loads `final_segmentation_model_gnn_sim.pt`, which is the **simulation**-trained
> checkpoint, so the configuration it actually runs is sim → meat (A→C), as the paper describes.

Two architectures exist, and a checkpoint only loads into the one it was trained with —
`GNN(arch="compact")` is the small model (`latent_dim` 64) and `GNN(arch="large")` the large one
(`latent_dim` 256), whose sizes come from the `*_large` keys of the `gnn` config block. The
dispatcher picks the right one per configuration: the two sim-trained models (A→B, A→C) use
the large architecture, the meat-trained model (C→B) the small one.

Training writes `*_retrained_<config>` artifacts rather than overwriting the published
checkpoints — see the note in
[Reproduce the published results](#reproduce-the-published-results). The configuration name is
part of the suffix because A→B and A→C share `train_on_sim()` and therefore the same `*_sim`
artifact names; tagging them keeps a later run from silently replacing an earlier one's
checkpoint. Re-running the same configuration overwrites in place.

The legacy `python -m difftactile.scripts.script_gnn` entrypoint (large model) still exists.

> ⚠️ **A CUDA GPU is required to even construct the model**, not just to train quickly:
> `difftactile/cnn/segmentation_gnn.py` allocates accumulators with a hardcoded `device='cuda:0'`
> and no CPU fallback, so evaluation fails on a CPU-only machine too.

Hyperparameters come from the `gnn` block of `system-params.json`. Outputs:

| Artifact | Path |
|---|---|
| Per-epoch metrics (CSVLogger) | `logs/my_experiment/run_<timestamp>/metrics.csv` |
| Best checkpoint (monitors `val_iou/1`) | `lightning_logs/.../best-model-sim.ckpt` (A→B, A→C), `best-model-meat.ckpt` (C→B) |
| Published weights read by `--eval` | `saved_models_sim/final_segmentation_model_gnn_sim.pt` (A→B, A→C), `saved_models_meat/final_segmentation_model_gnn_meat.pt` (C→B) |
| Pickled test set + normalisation stats | `difftactile/output/test_loader_gnn_sim.pickle` (A→B, A→C), `..._meat.pickle` (C→B) |
| Weights written by `--train` | the same paths with a `_retrained_<config>` suffix |
| ROC curve (`A-to-B --eval`) | `difftactile/output/roc_curve_A-to-B.pdf` |

`A-to-B --eval` loads the **simulation**-trained checkpoint
(`saved_models_sim/final_segmentation_model_gnn_sim.pt`) and needs **both** it and the
matching `difftactile/output/test_loader_gnn_sim.pickle` (it recovers the normalisation
statistics from the latter), plus the silicone dataset at
`SYSTEM_PARAMS.files.exp_data_silicone` — all three ship in the Zenodo bundle.

> **Corrected in this revision.** `evaluate_and_plot_roc()` previously loaded
> `final_segmentation_model_gnn_meat.pt` — the small, *meat*-trained model — with a
> default-architecture `GNN()`. That made `A-to-B --eval` compute **C→B**, duplicating the
> C-to-B configuration, so the AUC it reported was the meat-trained model on silicone rather
> than the sim-to-real result. Loading the sim checkpoint changes the reported figure from
> **0.6786 to 0.7314**. The `--train` path always built A→B correctly; only this evaluate
> shortcut was wrong.

The ROC PDF is always written to `difftactile/output/roc_curve_A-to-B.pdf`, and the script
exits on its own rather than waiting for you to close a plot window — open the PDF to inspect
the curve. See [Interactive windows](#interactive-windows) if you want the window back.

### 5. Visualisation and results

```bash
python -m difftactile.scripts.script_visualise        # predictions overlaid on sensor frames
python -m difftactile.scripts.script_visualise_mesh   # simulation mesh
python -m difftactile.scripts.script_predict_exp      # run a trained model on experimental data
```

`script_visualise` accepts a configuration name (`A-to-B`, `C-to-B`, `A-to-C`) plus
`--pretrained` / `--retrained` to pick the weights and test set together; `docker/view_predictions.sh`
is the wrapper around it. `script_predict_exp` builds the bird's-eye vessel map, wrapped by
`docker/vessel_map.sh`.

**Domain-adaptation overlay figures.** The sim-vs-real marker alignment images for the four
canonical interactions (press, twist-z, twist-x, slide) are drawn by
`Contact.generate_validation_img()` in [`difftactile/main/main.py`](difftactile/main/main.py),
called from `compute_da_loss()` in the same file — the simulator writes them inline while it
computes the alignment MAE, rather than in a separate plotting stage. They land in
`difftactile/output/da_overlay_{press,twist_z,twist_x,slide}.png` and are produced during a
collection run with `meta.load_params_from_bo == 1`. The real marker positions they are compared
against come from `extract_reorder_save_markers()` in
[`difftactile/data_analysis/experiment/domain_adaptation.py`](difftactile/data_analysis/experiment/domain_adaptation.py)
(`script_domain_adaptation`).

### Interactive windows

**No script waits for user input.** Every figure, mask and 3-D view is written to disk (mostly
under `difftactile/output/`), and the script then carries on and exits. This is what makes the
pipeline safe to run unattended — in Docker, over SSH, or in CI — where a window nobody can
close would hang the run forever. To look at a result, open the saved `.pdf` / `.png`.

Two environment variables change this:

| Variable | Effect |
|---|---|
| `DIFFTACTILE_INTERACTIVE=1` | Restore the original blocking behaviour: `plt.show()` waits, frame browsers step on your key presses (`j`/`k`/`q`), the Gmsh FLTK viewer opens, and the tkinter marker-labelling GUI runs. Requires a real display. |
| `DIFFTACTILE_HEADLESS=1` | Stronger: do not create windows at all. Implied automatically when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set. |
| `DIFFTACTILE_MAX_FRAMES=N` | How many frames a viewer loop steps through before returning when non-interactive. |
| `QT_QPA_PLATFORM=xcb` | Force the Qt annotation viewers onto X11 instead of letting Qt pick Wayland. Use inside the container or over X forwarding. |
| `DIFFTACTILE_ANNOTATOR_PYTHON` | Interpreter `docker/annotate_data.sh` should use, instead of the `vessel-palpation-annotator` micromamba env. |

Two tools exist *only* to collect manual input — the Qt click-annotator in
`preprocess_silicone_data.py::annotate()` and the tkinter labeller in
`marker_tracker.py::VideoPlayer.run()`. Without `DIFFTACTILE_INTERACTIVE=1` they print a note
and return immediately, leaving any annotations already on disk untouched.

The policy lives in `difftactile/main/paths.py`'s neighbour, `difftactile/main/display.py`;
route new GUI calls through its `wait_key()`, `imshow()`, `finish_plot()` and `prompt()`
helpers rather than calling OpenCV or pyplot directly.

### Running without a GPU

Set `RUN_ON_LAB_MACHINE = False` at `difftactile/main/main.py:30` to switch Taichi to the CPU
backend for the **simulation**. Expect a large slowdown.

The **GNN has no working CPU path**: several call sites do check `torch.cuda.is_available()`, but
`difftactile/cnn/segmentation_gnn.py:570` allocates its metric accumulators on a hardcoded `cuda:0`, so
constructing the model at all requires CUDA. Patch that line if you need CPU inference.

---

## Repository layout

```
difftactile/
├── main/                 Taichi FEM simulation core
│   ├── main.py           contact simulation + training-data collection (the big one)
│   ├── pre_main.py       trajectory/geometry precomputation
│   ├── apply_scaling.py  physical units -> simulation units
│   └── generate_*_gmsh.py  mesh generation
├── sensor_model/         ViTacTip FEM model + fisheye camera projection
├── object_model/         phantom, vein, mesh loading
├── cnn/                  GNN and CNN models, datasets, training, visualisation
│   ├── segmentation_gnn.py  the three paper configurations
│   ├── gnn.py            large (simulation-trained) model
│   └── dataset.py        dataset construction and splits
├── data_analysis/
│   ├── experiment/       real sensor data: tracking, calibration, annotation, ROC
│   ├── sim/              simulated data postprocessing
│   ├── training/ testing/  metrics and figures
├── scripts/              entrypoint wrappers — run these
├── system_params/        JSON configuration
├── meshes/               STL geometry
└── manual_or_experimental_data/   reference photos, calibration images, annotations, specs
```

Every runnable entrypoint is a thin wrapper in `difftactile/scripts/`; the logic lives in the
corresponding module. To add one, follow the existing 3-line pattern.

---

## Limitations and known issues

This is a research snapshot rather than a packaged tool, and it is published as-is. The first
two entries below are deliberate modelling decisions, and understanding them is essential to
interpreting the simulator's output correctly. The remainder are known rough edges and are
*not* worth reporting as bugs:

- **Contact compliance is deliberately asymmetric across the three contact pairs.** The
  sensor↔phantom pair is tuned to transfer very little deformation to either body, so the bulk
  of the phantom surface registers only weakly on the sensor membrane. Visible sensor
  deformation is instead driven almost entirely by the sensor↔vein pair. The simulator is
  therefore best understood as a *targeted model of the subsurface feature's mechanical
  signature* rather than a general-purpose soft-body contact solver: the quantity it is built
  to reproduce is the marker displacement field induced by a stiff inclusion beneath a
  compliant surface, not the absolute contact mechanics of the surface itself.

  This is a strong simplification of the underlying physics, and it is adopted because it
  reproduces the target signal well. The resulting marker deformations match those measured on
  the real ViTacTip across all four domain-adaptation trajectories (press, press-and-slide,
  press-and-twist-x, press-and-twist-z) and throughout training-set collection — which is the
  property the downstream GNN actually consumes. Since the network is trained purely in
  simulation and evaluated on real sensor video, the fidelity that matters is fidelity of the
  marker field, and the sim-to-real transfer results reported for the A→B and A→C
  configurations bear this out. Treat absolute contact forces and phantom-surface deformation
  magnitudes as uncalibrated; treat the marker displacements as the validated output.

- **The MPM phantom is kinematically fixed.** Every phantom material point is pinned rather
  than advected: in `Phantom.g2p()` (`difftactile/object_model/phantom.py`) each particle whose
  `is_fixed` flag is set has its velocity and affine velocity field zeroed and its position
  copied unchanged to the next substep. Despite the name, `phantom.fix_bottom_points` does not
  restrict this to the bottom layer — `is_fixed` is assigned `np.ones_like(...)`, so the flag
  is set for *all* particles and the phantom acts as a rigid, immovable body. (The commented-out
  `z_coords <= z_threshold` line and the now-unused `phantom.fixed_points_z_ratio` parameter
  are remnants of the earlier bottom-only scheme.) Note this pins the **particles**; the
  background Eulerian grid is rebuilt each substep as usual.

  This is intentional. Allowing the phantom to deform freely produced two failure modes that
  are avoided entirely by pinning it:

  1. **Collapse of the MPM body**, encountered when the grid node spacing was set too large —
     the deformation field is band-limited by the cell size, so an under-resolved grid cannot
     sustain the phantom's shape.
  2. **High-frequency jitter**, in which the phantom vibrated persistently and small clusters
     of particles were ejected far from the body.

  Because the informative signal is the sensor's response to the subsurface feature, freezing
  the phantom removes both instabilities without affecting the quantity being learned.

- **Configuration is partly "edit the source".** Enabling a pipeline stage, switching train vs.
  evaluate, or choosing the Taichi backend all mean editing Python, not passing a flag.
- **Absolute paths** to the original machine remain in a handful of files (listed above).
- **Non-interactive by default.** Nothing blocks waiting for a window to be closed; inspect
  the saved figures in `difftactile/output/` instead. `DIFFTACTILE_INTERACTIVE=1` restores the
  blocking windows — see [Interactive windows](#interactive-windows).
- **The annotation viewers need their own environment.** They are the only Qt (PySide6) windows
  in the project — everything else uses OpenCV — so they run from the dedicated
  `vessel-palpation-annotator` env on bare metal, not inside the container, which ships neither
  PySide6 nor PyAV. Being native Wayland clients is the point: it removed the stale-frame
  double-present workaround the OpenCV viewers needed, so one keypress moves exactly one frame.
- **`script_main` can segfault at exit when the Taichi GGUI window is open** (exit code 139,
  "Segmentation fault (core dumped)"). This happens *after* `main()` has finished and printed
  `all done`, during CUDA/GGUI teardown, so **the collected trajectories are complete and
  valid** — the crash cannot corrupt them. It only occurs when a display is available:
  `docker-run.sh` passes `DISPLAY` through, and `run_pipeline.sh` forces headless mode only
  when `DISPLAY` is unset. Run with `DIFFTACTILE_HEADLESS=1` to avoid it:

  ```bash
  docker exec -e DIFFTACTILE_HEADLESS=1 vessel-palpation ./docker/run_pipeline.sh sim-short
  ```

  Headless is also markedly faster (~108 s vs ~149 s for `sim-short`), so prefer it unless you
  actually want to watch the simulation.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `FileNotFoundError: difftactile/system_params/system-params.json` | Not running from the repository root. |
| `FileNotFoundError` on a `.npz` / `.pkl` / `.pt` | Missing dataset or checkpoint — see [Reproducibility](#reproducibility--read-before-running). |
| CUDA out of memory in Taichi | Lower `device_memory_GB` in `difftactile/main/main.py:2611`. |
| `ModuleNotFoundError: scipy` / `seaborn` | `uv pip install scipy seaborn`. |
| PyG extension import errors | The `pyg_lib` / `torch_scatter` wheels must match your Torch+CUDA build; edit the `-f` URL in the install scripts. |
| A pipeline step does nothing | Its call is commented out in the module's `main()` — uncomment it. |
| Taichi GGUI / Vulkan error, or hang on a headless machine | `script_main` always opens two GGUI windows and the gmsh scripts open FLTK windows. A display with a working Vulkan driver is required. |
| Edits to `system-params.json` keep reverting | `script_apply_scaling` regenerates them — edit `system-params-distances.json` instead. |
| `script_main` breaks after running the CFL script | It wrote scalar contact params where lists are expected; restore them in `system-params.json`. |

---

## Citation

This work builds on DiffTactile. If you use this code, please cite the original simulator:

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

Piotr Blaszyk — for questions about this fork please open an issue on this repository. The
experimental datasets and trained weights are not in the repository itself; they are published
on Zenodo at [10.5281/zenodo.21900934](https://doi.org/10.5281/zenodo.21900934).
