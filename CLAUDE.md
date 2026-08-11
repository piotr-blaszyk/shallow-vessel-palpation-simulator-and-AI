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

**Docker is the officially supported path** (`docker/` — see the README quickstart):

```bash
./docker/docker-build.sh && ./docker/docker-run.sh
docker exec -it difftactile ./docker/run_pipeline.sh check
docker exec -it difftactile ./docker/run_pipeline.sh sim-to-silicone
```

Directly, as a module:

```bash
python -m difftactile.scripts.script_main
```

Paths are resolved against the **repository root** by `difftactile/main/paths.py`
(`repo_path()` / `data_path()`), derived from that file's own location and overridable with
`DIFFTACTILE_ROOT`. Scripts therefore run from any working directory — the old
"must cd to the repo root first" constraint is gone.

The simulation pipeline order is fixed by `difftactile/scripts/run_all.sh`:
`script_apply_scaling` → `script_pre_main` → `script_main`. The simulator additionally needs
the Gmsh meshes (`script_generate_vitactip_mesh_gmsh`, `script_generate_vein_mesh_gmsh`) and
the sensor-geometry artifacts shipped in the Zenodo bundle.

### The three scenarios (single branch)

Selected **by name**, not by editing source:

```bash
python -m difftactile.scripts.script_iros_gnn sim-to-silicone   # evaluate on silicone + ROC
python -m difftactile.scripts.script_iros_gnn sim-to-meat       # train on meat, test on silicone
python -m difftactile.scripts.script_iros_gnn silicone-to-meat  # cross-domain, no retraining
```

`run_scenario()` in `cnn/iros_gnn.py` dispatches these. Note `GNN(arch=...)`: `"iros"` is the
small model (`latent_dim` 64), `"icra"` the large one (`latent_dim` 256) read from the
`*_icra` config keys — the ICRA checkpoint only loads into the latter.

### Environment overrides

| Variable | Effect |
|---|---|
| `DIFFTACTILE_ROOT` | Repository root used for all path resolution. |
| `DIFFTACTILE_DATA_ROOT` | Keep the large data bundle outside the repo. |
| `DIFFTACTILE_NUM_LOOPS` | Simulator loop count. Each loop = 2 substeps × 4 trajectories = 8 trials. Default 100 (800 trials, ~3 h at ~14 s/trial on an RTX 3080); `1` gives a ~3 min smoke test. |
| `DIFFTACTILE_HEADLESS=1` | Skip Taichi GGUI / Gmsh FLTK windows and blocking `plt.show()`. Required for SSH/CI/container runs with no display. |
| `DIFFTACTILE_SCENARIO` | Scenario name, if not passed as an argument. |

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

**Fixed since** (do not re-report these as bugs):
- `generate_*_mesh_gmsh.py` — `os.makedirs("output")` now uses `repo_path("difftactile/output")`,
  and the blocking `gmsh.fltk.run()` viewer is skipped when headless.
- `iros_gnn.main()` no longer starts with a bare `return`; scenario selection replaced it.
- `evaluate_and_plot_roc()` no longer hardcodes its output path, and `plt.show()` is guarded.

Interactivity is otherwise pervasive and intentional: the cv2 annotation GUI and the tkinter
marker-labelling GUI still block by design. The **simulator and all three GNN scenarios now run
headless** under `DIFFTACTILE_HEADLESS=1`; the Docker image also passes X through, so GUI windows
work when a display is available.

## Data availability — the main gotcha

**None of the datasets or trained checkpoints are in this repository.** `.gitignore` excludes
`*.npz`, `*.pkl`, `*.pt`, `*.mp4`, `*.mkv`, `*.csv`, `output/`, `saved_models/`, `logs/`.
The paths below do not exist in a fresh clone:

- `difftactile/output/` — every intermediate artifact the sim writes
- `saved_models_iros/`, `saved_models_icra/` — trained GNN weights
- `difftactile/manual_or_experimental_data/iros_training_data/clean/` — real meat-experiment trials
- `difftactile/manual_or_experimental_data/endgame/20250901-131547_dense` — silicone dataset

So most ML entrypoints **cannot run in a fresh clone** — they will raise `FileNotFoundError`.
Do not "fix" such a failure by inventing data. Restore the published bundle instead:

```bash
./data/restore_data.sh difftactile-data.tar.gz   # ~190 MB from Zenodo
./data/restore_data.sh --verify                  # check what is present
```

`data/MANIFEST.md` documents what the bundle contains and — more usefully — what is
deliberately excluded (raw videos, intermediate preprocessing stages, training logs), which
is what takes it from 4.5 GB to ~190 MB. `data/make_data_bundle.sh` rebuilds it (author-side).
`data/ZENODO_UPLOAD.md` covers publishing it from the command line.

The **hardcoded absolute paths are gone** — everything now resolves through
`difftactile/main/paths.py`. Note that the author's own checkout wires some data paths up as
**symlinks** into an external directory; those resolve on the host but *not* inside the
container, which is why `restore_data.sh` replaces them with real files.

## Branches

**All three scenarios now live on one branch** and are selected by name (see above). The
per-experiment branches below are a historical record; changes should target the unified
branch rather than reviving them.

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
