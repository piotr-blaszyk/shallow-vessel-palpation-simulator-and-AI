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

Data and trained model weights are published on Zenodo — see
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

> **Use the `main` branch** — it is the only supported one, and a plain `git clone` already
> puts you there. The other branches are frozen historical snapshots; see
> [Branches](#branches).

```bash
# 1. Clone (defaults to main - the only supported branch)
git clone https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI.git
cd shallow-vessel-palpation-simulator-and-AI

# 2. Fetch the data bundle from Zenodo (~190 MB) and unpack it into place
#    (datasets + trained checkpoints; see data/MANIFEST.md for what is inside)
wget https://zenodo.org/records/<RECORD_ID>/files/shallow-vessel-palpation-data.tar.gz
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz

# 3. Build the image (~10-30 min, downloads several GB) and start the container
./docker/docker-build.sh
./docker/docker-run.sh

# 4. Verify GPU, dependencies and data are all in place
docker exec -it difftactile ./docker/run_pipeline.sh check
```

### Reproduce the published results

The paper's three models — one per (train → test) configuration over the simulated (**A**),
silicone (**B**) and meat (**C**) datasets — are selected **by name**, with no source editing
and no branch switching:

```bash
# Evaluate the sim-trained GNN on the real SILICONE phantom -> ROC curve
docker exec -it difftactile ./docker/run_pipeline.sh A-to-B

# Cross-domain: test the sim-trained checkpoint on real MEAT (no retraining)
docker exec -it difftactile ./docker/run_pipeline.sh A-to-C

# Train on real MEAT trials, test on silicone
docker exec -it difftactile ./docker/run_pipeline.sh C-to-B

# ...or all three in sequence
docker exec -it difftactile ./docker/run_pipeline.sh all-scenarios
```

Each configuration also takes `--train` (reproduce the model from scratch) or `--eval` (load
the published checkpoint); see [Training and evaluating the GNN](#4-training-and-evaluating-the-gnn)
for the full table. Outputs land in `difftactile/output/` (e.g. `roc_curve_iros.pdf`) and `logs/`.

Any run that **trains** writes `*_retrained` artifacts
(`final_segmentation_model_gnn_iros_retrained.pt`,
`test_loader_gnn_iros_retrained.pickle`, and the `_icra` equivalents) rather than overwriting
the published checkpoints that the evaluation paths read — otherwise running the
configurations in sequence would silently change the reported AUC. Pass
`DIFFTACTILE_OVERWRITE_PUBLISHED=1` if you deliberately want to replace them.

### Regenerate the simulated dataset (optional)

The simulated training set ships in the Zenodo bundle, so **this is not required** to
reproduce the results. Run it only if you want to extend the project:

```bash
# ~2-3 minutes: a single loop (8 trials), to check the simulator works
docker exec -it difftactile ./docker/run_pipeline.sh sim-short

# ~2 h 45 m: a full 800-trial collection run
docker exec -it difftactile ./docker/run_pipeline.sh sim-full
```

> **To regenerate the *published* dataset specifically, set
> `DIFFTACTILE_TRAJECTORIES=3`.** All 500 trajectories in the shipped dataset are
> type 3 ("slide (vein)") — it was collected when the collection loop read
> `range(3, 4)`, which a later commit widened to all four types. A default run
> therefore also produces types 0/1/2, and type 0 yields empty arrays by design
> (it ends in ~36 timesteps, below the `ts > 80` recording threshold).
>
> ```bash
> docker exec -it -e DIFFTACTILE_TRAJECTORIES=3 difftactile \
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

## Branches

> ### 👉 Use `main`. Ignore every other branch.
>
> **`main` is the only supported branch.** It is the only one that is maintained, the only one
> the documentation describes, and the only one the Docker image and Zenodo bundle are tested
> against. All three of the paper's models train and evaluate from it, selected by name (see
> [Quickstart](#quickstart-docker)) — there is never a reason to switch branches.
>
> Everything below is a **historical record** of how the work developed. Those branches are
> frozen: they are not maintained, they do not receive fixes, and several are known to be
> broken in ways `main` has since repaired. Do not base new work on them, and do not merge
> them into `main`.

Development originally happened on parallel branches, one per experiment. That structure is
retained only so the exact code behind a given published result stays recoverable.

| Branch | Status | What it is |
|---|---|---|
| **`main`** | ✅ **Supported — use this.** | The unified code: all three (train → test) configurations, each in both `--train` and `--eval` mode. Everything in this README refers to `main`. |
| `iros` | ⚠️ Frozen snapshot. | The IROS submission state, sim-to-real onto a **silicone vascular phantom**. Currently the same commit as `main`, kept as a named pointer to the submission. |
| `sim-to-silicone` | ⚠️ Frozen snapshot. | The silicone-phantom experiment as submitted. Identical to `iros`. |
| `sim-to-meat-test` | ❌ **Obsolete — do not use or merge.** | *Historical.* Transfers the model to **real meat**, plastic straws standing in for vessels beneath layers of steak. Superseded by the `A-to-C` configuration on `main`. It now **predates** `main`: no path, Docker or data-bundle infrastructure, and its sim-training entrypoint is disabled by a bare `return`. Merging it would delete that infrastructure, and `main` already carries its useful content — including a normalisation fix that raises the reported cross-domain vein IoU from 0.034 to 0.198. |

The meat trials are catalogued in
[`difftactile/manual_or_experimental_data/iros_experiment_spec.md`](difftactile/manual_or_experimental_data/iros_experiment_spec.md)
(straw depth vs. number of 5 mm steaks above it).

---

## Setup

### Requirements

- **Python 3.10–3.12.** (Upstream DiffTactile said 3.9.16; that does not apply to this fork's
  torch 2.8 / CUDA 12.6 stack.)
- **An NVIDIA GPU with CUDA is effectively mandatory.** Developed on an RTX 3080 (10 GB);
  `difftactile/main/main.py` requests `device_memory_GB=9` from Taichi. The simulation can be
  switched to CPU (see [Running without a GPU](#running-without-a-gpu)), but the **GNN cannot** —
  `difftactile/cnn/iros_gnn.py:570` allocates on a hardcoded `cuda:0` with no fallback, so even
  loading a checkpoint to plot a ROC curve fails on a CPU-only machine.
- **A display is optional.** Taichi GGUI, Gmsh and `plt.show()` are used when one is
  available; `DIFFTACTILE_HEADLESS=1` skips all of them, so the simulator and all three GNN
  scenarios run over SSH or in CI. The cv2/tkinter annotation tools still require a display.
- Linux (developed on Ubuntu 24.04).

### Install

```bash
# Defaults to main - the only supported branch (see Branches).
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
   you enable by uncommenting. `difftactile/data_analysis/experiment/endgame.py` is the clearest
   case; `difftactile/scripts/script_iros_gnn.py` toggles between training and evaluation.

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
| Trained GNN weights + test-loader pickle | `saved_models_iros/final_segmentation_model_gnn_iros.pt`, `difftactile/output/test_loader_gnn_iros.pickle` | evaluation / ROC (needs **both**) | only by retraining |
| Silicone dataset | `difftactile/manual_or_experimental_data/endgame/20250901-131547_dense` | silicone training/eval | from raw video via `script_endgame` |
| Meat experiment trials | `difftactile/manual_or_experimental_data/iros_training_data/clean/` | `sim-to-meat-test` | from raw video via `script_iros_preprocess_data` |
| Raw experiment videos + robot poses | `.../iros_training_data/raw/<id>.{avi,npz}`, `endgame/…` | all preprocessing | no — must be supplied |
| Fisheye calibration | `difftactile/output/fisheye_params.npz` | undistortion | yes, with your own checkerboard images |
| Marker-tracker trajectories | `difftactile/output/marker_tracker/domain-adaptation-vascular-markers/traj_{0..3}_out.pkl` | **`script_main` construction** | from raw video via marker tracking |

### The fix: restore the published data bundle

Everything in that table is supplied by the Zenodo archive, so none of it has to be
regenerated:

```bash
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
> ⚠️ **Do not use `difftactile/scripts/script_all.py`.** It imports all three modules at the top,
> so `constants.py` loads `system-params.json` *before* `apply_scaling` rewrites it, and the
> simulation then runs against pre-scaling constants. Use `run_all.sh` (three separate
> processes) instead.

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

`script_cfl_and_contact_params_estimation` (CFL timestep + Hertzian contact-stiffness estimates)
is **diagnostics only and currently destructive**: it writes contact parameters back to
`system-params.json` as scalars, whereas `main.py` expects 3-element lists (one per contact
pair), so running it breaks the next `script_main`. This is why it is commented out in
`run_all.sh`.

### 2. Processing real sensor data

Requires the recorded videos.

```bash
python -m difftactile.scripts.script_iros_preprocess_data    # meat experiment: raw/ -> clean/
python -m difftactile.scripts.script_endgame                 # silicone phantom cleaning pipeline
python -m difftactile.scripts.script_marker_tracker          # marker tracking + tkinter labelling GUI
python -m difftactile.scripts.script_domain_adaptation       # sim-vs-real marker comparison
```

**Meat (`script_iros_preprocess_data`)** — decimates raw 1920×1080 AVIs by 15×, interpolates
robot poses onto frames, detects and Hungarian-reorders markers, then projects the straw geometry
through the fisheye camera model to derive a binary label per marker. Trial geometry is parsed
from `iros_experiment_spec.md`. Produces, per trial, `clean/<trial_id>/marker_positions.npz`
`(T, 127, 2)` and `marker_labels.npz` `(T, 127)`, plus an overlay video for eyeballing labels.

**Silicone (`script_endgame`)** — a chain of directory-to-directory stages: interpolate/trim →
dilate → extract markers → reorder → annotate (a manual cv2 click GUI) → line points → merge into
the simulation `.npz` format → add dense labels, ending at the `_dense` directory that
`exp_data_endgame` points to. Each stage is a method call in `endgame.main()`; on `iros` most are
commented out, on `sim-to-meat-test` the full chain is enabled. **Uncomment the stages you need,
in order.**

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
published checkpoint — no source editing and **no branch switching**:

```bash
# Train each of the three models from scratch
python -m difftactile.scripts.script_iros_gnn A-to-B --train   # train on sim,  test on silicone
python -m difftactile.scripts.script_iros_gnn C-to-B --train   # train on meat, test on silicone
python -m difftactile.scripts.script_iros_gnn A-to-C --train   # train on sim,  test on meat

# ...or reproduce the published numbers without retraining
python -m difftactile.scripts.script_iros_gnn A-to-B --eval    # evaluate on silicone + ROC
python -m difftactile.scripts.script_iros_gnn A-to-C --eval    # cross-domain, no retraining
```

| Config | Train set | Test set | `--train` | `--eval` | Default |
|---|---|---|---|---|---|
| `A-to-B` | simulation | silicone | Trains the large (ICRA) model on sim, tests on silicone. | Loads the published sim-trained checkpoint, evaluates on silicone, writes the ROC curve. | `--eval` |
| `C-to-B` | meat | silicone | Trains the small (IROS) model on the real meat trials, tests the best checkpoint on silicone. | Same as `--train` — the published run ends by testing on silicone. | `--train` |
| `A-to-C` | simulation | meat | Trains the large (ICRA) model on sim, tests on meat with every trial in the test split. | Loads the published sim-trained checkpoint and tests it on meat, no retraining. | `--eval` |

Omitting the mode uses the default in the last column (evaluation wherever a published
checkpoint makes it the cheaper path). The configuration can also be given as
`DIFFTACTILE_SCENARIO` and the mode as `DIFFTACTILE_MODE`.

> The **older scenario names still work** as aliases: `sim-to-silicone` → `A-to-B`,
> `sim-to-meat` → `C-to-B`, `silicone-to-meat` → `A-to-C`. Note that `silicone-to-meat` was a
> misnomer: it loads `final_segmentation_model_gnn_icra.pt`, which is the **simulation**-trained
> checkpoint, so the configuration it actually runs is sim → meat (A→C), as the paper describes.

Two architectures exist, and a checkpoint only loads into the one it was trained with —
`GNN(arch="iros")` is the small model (`latent_dim` 64) and `GNN(arch="icra")` the large one
(`latent_dim` 256), whose sizes come from the `*_icra` keys of the `gnn` config block. The
dispatcher picks the right one per configuration: the two sim-trained models (A→B, A→C) use
the large architecture, the meat-trained model (C→B) the small one.

Training writes `*_retrained` artifacts rather than overwriting the published checkpoints —
see the note in [Reproduce the published results](#reproduce-the-published-results).

The legacy `python -m difftactile.scripts.script_gnn` entrypoint (ICRA model) still exists.

> ⚠️ **A CUDA GPU is required to even construct the model**, not just to train quickly:
> `difftactile/cnn/iros_gnn.py` allocates accumulators with a hardcoded `device='cuda:0'`
> and no CPU fallback, so evaluation fails on a CPU-only machine too.

Hyperparameters come from the `gnn` block of `system-params.json`. Outputs:

| Artifact | Path |
|---|---|
| Per-epoch metrics (CSVLogger) | `logs/my_experiment/run_<timestamp>/metrics.csv` |
| Best checkpoint (monitors `val_iou/1`) | `lightning_logs/version_N/checkpoints/best-model-iros.ckpt` |
| Final weights | `saved_models_iros/final_segmentation_model_gnn_iros.pt` |
| Pickled test set + normalisation stats | `difftactile/output/test_loader_gnn_iros.pickle` |
| ROC curve | `difftactile/output/roc_curve_iros.pdf` |

`A-to-B --eval` needs **both** the `.pt` weights and the `.pickle` (it recovers the
normalisation statistics from the latter), plus the silicone dataset at
`SYSTEM_PARAMS.files.exp_data_endgame` — all three ship in the Zenodo bundle. The ROC PDF is
always written to disk; matplotlib falls back to a non-interactive backend when there is no
display, or with `DIFFTACTILE_HEADLESS=1`, so plotting never blocks a container or SSH run.

### 5. Visualisation and results

```bash
python -m difftactile.scripts.script_visualise        # predictions overlaid on sensor frames
python -m difftactile.scripts.script_visualise_mesh   # simulation mesh
python -m difftactile.scripts.script_predict_exp      # run a trained model on experimental data
```

### Running without a GPU

Set `RUN_ON_LAB_MACHINE = False` at `difftactile/main/main.py:30` to switch Taichi to the CPU
backend for the **simulation**. Expect a large slowdown.

The **GNN has no working CPU path**: several call sites do check `torch.cuda.is_available()`, but
`difftactile/cnn/iros_gnn.py:570` allocates its metric accumulators on a hardcoded `cuda:0`, so
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
│   ├── iros_gnn.py       IROS model
│   ├── gnn.py            ICRA model
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

## Known issues

This is a research snapshot rather than a packaged tool, and it is published as-is. The
following are known and are *not* worth reporting as bugs:

- **Several entrypoints are disabled or broken as shipped.** `script_all.py` has the import-order
  bug described above; `script_benchmark_dataset` and `script_hungarian_exp` reference JSON keys
  (`dataset_root_reordered`, `experiment_straight_markers_npz_reordered`) that do not exist in
  `system-params.json`; `script_train` (the legacy U-Net CNN baseline) calls `MyDataset` with an
  outdated signature and needs `monai`, which is not in the requirements;
  `difftactile/cnn/threshold_gnn.py` is dead code that no longer matches the `GNN` API.
- **`difftactile/data_analysis/experiment/roc_curve.py` plots a synthetic curve** (`tpr = fpr**0.5`)
  — it is a figure-styling template. The real ROC is `iros_gnn.evaluate_and_plot_roc()`.
- **Configuration is partly "edit the source".** Enabling a pipeline stage, switching train vs.
  evaluate, or choosing the Taichi backend all mean editing Python, not passing a flag.
- **Absolute paths** to the original machine remain in a handful of files (listed above).
- **Interactive by default.** Gmsh, Taichi GGUI, the annotation and marker-tracking GUIs, and
  `plt.show()` all assume a display.

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

Piotr Blaszyk — for questions about this fork, or to request the experimental datasets and
trained weights that are not included here, please open an issue on this repository.
