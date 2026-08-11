# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repository is

A fork of [DiffTactile](https://difftactile.github.io/) (ICLR 2024), heavily modified for a
masters project on **sim-to-real subsurface feature localisation with an optical tactile sensor**.

The research question: can a GNN trained *purely in simulation* localise features hidden
*beneath* a soft surface (veins in a silicone phantom, plastic straws under layers of steak)
from the deformation of markers on a ViTacTip sensor?

The upstream DiffTactile manipulation tasks (`box_open`, `cable_straightening`, `grasp_*`,
`object_repose`, `surface_follow`) have been **deleted** from the working branches. Only the
differentiable Taichi FEM core was kept and repurposed.

Pipeline in one line:
`Taichi FEM sim → synthetic marker displacements → GNN → evaluate on real sensor video`.

## Repository layout

| Path | Purpose |
|---|---|
| `difftactile/main/` | Simulation core. `main.py` (~2650 lines) is the Taichi contact/FEM sim + training-data collection loop. `pre_main.py` precomputes trajectories/geometry; `apply_scaling.py` converts physical units into sim units. |
| `difftactile/sensor_model/` | `vitactip.py` (~1150 lines) FEM model of the ViTacTip sensor; `fisheye_model_{taichi,no_taichi}.py` projects 3D nodes to camera pixels. |
| `difftactile/object_model/` | `phantom.py`, `vein.py`, `obj_loader.py` — the soft phantom and the subsurface feature. |
| `difftactile/cnn/` | ML despite the name — mostly GNNs. `gnn.py` (ICRA/silicone), `iros_gnn.py` (IROS), `dataset.py` (~1900 lines), `train.py`, `visualise.py`. |
| `difftactile/data_analysis/experiment/` | Real-sensor data: `endgame.py` (the main preprocessing pipeline), `marker_tracker.py`, `iros_preprocess_data.py`, camera calibration, annotation, ROC. |
| `difftactile/data_analysis/sim/` | Simulated-data postprocessing and dataset benchmarking. |
| `difftactile/scripts/` | Thin `script_*.py` entrypoint wrappers — each imports a `main()` and calls it. This is the intended way to run anything. |
| `difftactile/system_params/` | JSON configuration (see below). |
| `difftactile/meshes/`, `difftactile/manual_or_experimental_data/` | STL meshes; reference photos, calibration images, annotations, experiment specs. |

## How to run things

Always run **from the repository root** as a module:

```bash
python -m difftactile.scripts.script_main
```

This matters: `difftactile/main/constants.py` loads
`ConstantsFromJson("difftactile/system_params/system-params.json")` with a **relative** path,
so anything launched from another working directory fails immediately.

The simulation pipeline order is fixed by `difftactile/scripts/run_all.sh`:
`script_apply_scaling` → `script_pre_main` → `script_main`.

## Configuration model — read this before changing behaviour

There is **no argparse and no CLI flags anywhere** in the project. Behaviour is controlled by:

1. **`difftactile/system_params/system-params.json`** — the single large source of truth
   (geometry, material params, trajectory, `gnn` hyperparameters, and a big `files` section of
   paths). Reached in code as `SYSTEM_PARAMS.gnn.batch_size`, `SYSTEM_PARAMS.files.dataset_root`
   etc. via the attribute-wrapper in `constants_common.py`.
   `system-params-computed.json` is **generated** by `apply_scaling.py` — do not hand-edit it.
2. **Module-level constants**, notably `RUN_ON_LAB_MACHINE = True` at `difftactile/main/main.py:30`,
   which selects `arch=ti.cuda, device_memory_GB=9` vs `arch=ti.cpu`.
3. **Commented-out call lists.** Several `main()` functions are effectively a menu where the
   user comments/uncomments steps. `data_analysis/experiment/endgame.py` `main()` is the clearest
   example, and `scripts/script_iros_gnn.py` toggles between `main()` (train) and
   `evaluate_and_plot_roc()` (evaluate). On the `iros` branch, `iros_gnn.main()` starts with a
   bare `return`, so training is a no-op there by design.

When asked to "change a parameter", prefer editing the JSON over hardcoding in Python — but note
the JSON hierarchy: `apply_scaling.py` **regenerates parts of `system-params.json` in place** from
`system-params-distances.json` (SI metres) and `system-params-youngs-modulus.json`, and
`pre_main.py` generates `system-params-computed.json` wholesale. Edit the `-distances` /
`-youngs-modulus` sources for lengths and stiffnesses, never the generated files.
`system-params-units.json` and `system-params-literature-values.json` are documentation — no code
reads them.

## Known-broken things — do not "fix" these unasked

Several entrypoints do not work as shipped. This is a research snapshot; before debugging one,
check whether it is already known:

- `scripts/script_all.py` — imports all three modules at the top, so `constants.py` reads
  `system-params.json` *before* `apply_scaling` rewrites it. Use `run_all.sh` instead.
- `script_cfl_and_contact_params_estimation` — writes scalar contact params where `main.py:635-638`
  expects 3-element lists, breaking the next `script_main`. Deliberately commented out of `run_all.sh`.
- `script_benchmark_dataset`, `script_hungarian_exp` — reference JSON keys that do not exist
  (`dataset_root_reordered`, `experiment_straight_markers_npz_reordered`).
- `script_train` — legacy U-Net CNN; outdated `MyDataset` signature, and needs `monai` which is
  not in `requirements/`.
- `cnn/threshold_gnn.py` — dead code, no longer matches the `GNN` API.
- `data_analysis/experiment/roc_curve.py` — plots a **synthetic** curve (`tpr = fpr**0.5`); it is a
  styling template, not a result. Real ROC is `iros_gnn.evaluate_and_plot_roc()`.
- `cnn/iros_gnn.py:570` — hardcoded `device='cuda:0'` with no CPU fallback, so constructing `GNN()`
  requires a GPU even for evaluation.
- `generate_*_mesh_gmsh.py` — `os.makedirs("output")` creates `./output`, not `./difftactile/output`.

Interactivity is pervasive and intentional: Taichi GGUI windows (opened unconditionally in
`main.main()`), Gmsh FLTK windows, the cv2 annotation GUI, the tkinter marker-labelling GUI, and
`plt.show()`. None of this runs headless without patching.

## Data availability — the main gotcha

**None of the datasets or trained checkpoints are in this repository.** `.gitignore` excludes
`*.npz`, `*.pkl`, `*.pt`, `*.mp4`, `*.mkv`, `*.csv`, `output/`, `saved_models/`, `logs/`.
The paths below do not exist in a fresh clone:

- `difftactile/output/` — every intermediate artifact the sim writes
- `saved_models_iros/`, `saved_models_icra/` — trained GNN weights
- `difftactile/manual_or_experimental_data/iros_training_data/clean/` — real meat-experiment trials
- `difftactile/manual_or_experimental_data/endgame/20250901-131547_dense` — silicone dataset

So most ML entrypoints **cannot run in a fresh clone** — they will raise `FileNotFoundError`.
Do not "fix" such a failure by inventing data. The simulation half (`script_pre_main`,
`script_main`, mesh generation) *does* run from a clean clone because it generates its own inputs.

A few **absolute paths hardcoded to the original machine** remain and must be edited by anyone
else running the code:
- `difftactile/cnn/dataset.py:21` `IROS_CLEAN_DATA_DIR = "/home/psb120/Documents/diff-tactile-fork/..."`
- `difftactile/system_params/system-params.json:311` `root_dir`
- `difftactile/data_analysis/testing/prettify_confusion_image.py:5-6`
- `difftactile/data_analysis/training/print_metrics_csv.py:3`
- `difftactile/data_analysis/experiment/calibrate_camera.py:20`

## Branches

`main` tracks the latest work (identical to `iros`). Three parallel branches matter:

- **`iros`** — IROS submission state. Sim-to-real onto a **silicone** vascular phantom. GNN
  config is the small model (`latent_dim` 64, 30 epochs).
- **`sim-to-silicone`** — currently **the same commit as `iros`** (zero diff); the silicone
  experiment as submitted.
- **`sim-to-meat-test`** — 4 commits ahead of `iros`. Transfers the model to **real meat** with
  plastic straws as pseudo-vessels. Differs in: a much larger GNN (`latent_dim` 256,
  `small_input_dim` 248, `skip_dim` 128, 1 epoch, batch 16), `dataset.py::create_splits_iros`
  routes all trials to the **test** split, `endgame.main()` re-enables the full cleaning
  pipeline, and `script_iros_gnn.py` calls `main()` instead of `evaluate_and_plot_roc()`.

Tag `upstream-difftactile` preserves the pristine upstream DiffTactile state that `main`
formerly pointed at.

When editing, keep the four branches' READMEs consistent unless the change is branch-specific.

## Environment

Python 3.9–3.12, CUDA GPU strongly recommended (developed on an RTX 3080, and
`device_memory_GB=9` in `main.py` assumes ~10 GB). Dependencies are installed by the shell
scripts in `requirements/` (`install_dependencies_difftactile.sh` for the simulator,
`install_dependencies_ml.sh` for the GNN stack). Both are `#!/bin/zsh` and use `uv pip install`.
Note `requirements/requirements_ml.in` pins wheels against `torch-2.8.0+cu126`.
There is no pinned lockfile and no `requirements.txt` (it was deleted from upstream).

## Conventions

- British spelling in identifiers (`visualise`, `normalised`, `optimisation`).
- Entrypoints go in `difftactile/scripts/script_<name>.py` as a 3-line wrapper; the logic lives
  in the corresponding module.
- Taichi kernels live in `main.py` / `vitactip.py`; be careful editing them — Taichi's autodiff
  imposes constraints (no dynamic indexing patterns, careful with mutable state across kernels).
- Add comments/docstrings explaining intent at a high level.
