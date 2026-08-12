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
| `difftactile/cnn/` | ML despite the name — mostly GNNs. `gnn.py` (large/silicone model), `segmentation_gnn.py` (the three paper configurations), `dataset.py` (~1900 lines), `train.py`, `visualise.py`. |
| `difftactile/data_analysis/experiment/` | Real-sensor data: `preprocess_silicone_data.py` (the main preprocessing pipeline), `marker_tracker.py`, `preprocess_meat_data.py`, camera calibration, annotation, ROC. |
| `difftactile/data_analysis/sim/` | Simulated-data postprocessing and dataset benchmarking. |
| `difftactile/scripts/` | Thin `script_*.py` entrypoint wrappers — each imports a `main()` and calls it. This is the intended way to run anything. |
| `difftactile/system_params/` | JSON configuration (see below). |
| `difftactile/meshes/`, `difftactile/manual_or_experimental_data/` | STL meshes; reference photos, calibration images, annotations, experiment specs. |

## How to run things

**Docker is the officially supported path** (`docker/` — see the README quickstart):

```bash
./docker/docker-build.sh && ./docker/docker-run.sh
docker exec -it difftactile ./docker/run_pipeline.sh check
docker exec -it difftactile ./docker/run_pipeline.sh A-to-B
```

**Claude Code: test through the Docker container.** It is the only environment with the full
stack (Taichi included). Fall back to the bare-metal `/home/psb120/micromamba/envs/claude`
env only if the container fails or is too much hassle — note it has **no Taichi**, so the
simulator cannot run there and `run_pipeline.sh check` will fail partway through.

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

### The three paper configurations (single branch)

The paper uses three datasets — **A** simulated, **B** real silicone, **C** real meat — and
reports three models, one per (train → test) configuration. All are selected **by name**, not
by editing source, and each runs in either `--train` or `--eval` mode:

```bash
python -m difftactile.scripts.script_segmentation_gnn A-to-B --train  # train on sim,  test on silicone
python -m difftactile.scripts.script_segmentation_gnn C-to-B --train  # train on meat, test on silicone
python -m difftactile.scripts.script_segmentation_gnn A-to-C --train  # train on sim,  test on meat

python -m difftactile.scripts.script_segmentation_gnn A-to-B --eval   # published ckpt + ROC
python -m difftactile.scripts.script_segmentation_gnn A-to-C --eval   # cross-domain, no retraining
```

Omitting the mode uses `DEFAULT_MODES` (eval for A-to-B and A-to-C, train for C-to-B).
`run_scenario()` in `cnn/segmentation_gnn.py` dispatches on `CONFIG_ACTIONS`; the older names
(`sim-to-silicone`, `sim-to-meat`, `silicone-to-meat`) are still accepted via
`SCENARIO_ALIASES`. Beware that `silicone-to-meat` is a **misnomer** — it loads the
*simulation*-trained checkpoint, so it is really A→C.

Note `GNN(arch=...)`: `"compact"` is the small model (`latent_dim` 64), `"large"` the large one
(`latent_dim` 256) read from the `*_large` config keys — the simulation-trained checkpoint only
loads into the latter. The two sim-trained configurations (A→B, A→C) use `"large"`, C→B uses
`"compact"`.

Training never overwrites the published checkpoints: `_retrained_path()` inserts a
`_retrained_<config>` suffix (e.g. `final_segmentation_model_gnn_sim_retrained_A-to-B.pt`)
unless `DIFFTACTILE_OVERWRITE_PUBLISHED=1`. The configuration name is part of the suffix
because `train_on_sim()` backs **both** A→B and A→C and writes the same `*_sim` artifacts —
without it, training the two in sequence would leave only the second one's checkpoint.
Re-running the *same* configuration still overwrites in place.

### Environment overrides

| Variable | Effect |
|---|---|
| `DIFFTACTILE_ROOT` | Repository root used for all path resolution. |
| `DIFFTACTILE_DATA_ROOT` | Keep the large data bundle outside the repo. |
| `DIFFTACTILE_NUM_LOOPS` | Simulator loop count. Each loop = 2 substeps × 4 trajectories = 8 trials. Default 100 (800 trials, measured 2 h 45 m on an RTX 3080); `1` gives a ~3 min smoke test. |
| `DIFFTACTILE_HEADLESS=1` | Skip Taichi GGUI / Gmsh FLTK windows and blocking `plt.show()`. Required for SSH/CI/container runs with no display. |
| `DIFFTACTILE_TRAJECTORIES` | Comma-separated trajectory types to collect (0 press, 1 slide-vein, 2 twist-y, 3 twist-z). Default all four. **The published dataset is entirely type 3** — use `3` to reproduce it. |
| `DIFFTACTILE_VEIN_PAIR=1` | Enable the sensor↔vein contact pair on the first of each loop's two substeps, so a trajectory runs once **with** a subsurface vein and once **without**. The vein half is hard-disabled in the committed default (`if False and j < 1` in `main()`), so every substep otherwise runs vein-free. |
| `DIFFTACTILE_SCENARIO` | Configuration name (`A-to-B`, `C-to-B`, `A-to-C`, or a legacy alias), if not passed as an argument. |
| `DIFFTACTILE_MODE` | `train` or `eval`, if not passed as `--train` / `--eval`. |
| `DIFFTACTILE_OVERWRITE_PUBLISHED` | `1` lets a training run overwrite the published checkpoints instead of writing `*_retrained` copies. |

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
   user comments/uncomments steps. `data_analysis/experiment/preprocess_silicone_data.py` `main()` is the clearest
   example, and `scripts/script_segmentation_gnn.py` toggles between `main()` (train) and
   `evaluate_and_plot_roc()` (evaluate). On the historical `iros` branch (a frozen snapshot —
   the branch name predates the rename), `segmentation_gnn.main()` starts with a bare `return`,
   so training is a no-op there by design.

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

- `cnn/segmentation_gnn.py:570` — hardcoded `device='cuda:0'` with no CPU fallback, so constructing `GNN()`
  requires a GPU even for evaluation.
- `script_main` segfaults at interpreter exit (code 139) when a Taichi GGUI window is open —
  i.e. whenever `DISPLAY` is set, since `run_pipeline.sh` only forces headless when it is unset.
  The crash is in CUDA/GGUI teardown *after* `main()` prints `all done`, so the collected
  trajectories are complete and valid. Verified identical on the commit before the dead-code
  sweep, so it is long-standing and unrelated. Use `DIFFTACTILE_HEADLESS=1` (also ~35% faster).

**Deleted** (they were broken or superseded; see the dead-code sweep in git history — do not
recreate them): `scripts/script_all.py` (import-order bug — use `run_all.sh`),
`script_cfl_and_contact_params_estimation.py`, `script_benchmark_dataset.py` +
`data_analysis/sim/benchmark_dataset.py`, `script_hungarian_exp.py` (the *wrapper* only —
`hungarian_exp.py` itself is live, imported by `predict_exp.py`), `script_train.py` +
`cnn/train.py` + `cnn/lit_module_unet_cnn.py` (legacy U-Net path), `cnn/threshold_gnn.py`,
`data_analysis/experiment/roc_curve.py` (synthetic curve; the real ROC is
`segmentation_gnn.evaluate_and_plot_roc()`), the `sandbox/` and `ml_training_old/` folders, and assorted
one-off analysis scripts.

Note `main/cfl_and_contact_params_estimation.py` **stays** — `main.py:21` imports it. Only its
entrypoint wrapper was removed, since running it writes scalar contact params where `main.py`
expects 3-element lists and so breaks the next `script_main`.

**Fixed since** (do not re-report these as bugs):
- `generate_*_mesh_gmsh.py` — `os.makedirs("output")` now uses `repo_path("difftactile/output")`,
  and the blocking `gmsh.fltk.run()` viewer is skipped when headless.
- `segmentation_gnn.main()` no longer starts with a bare `return`; scenario selection replaced it.
- `evaluate_and_plot_roc()` no longer hardcodes its output path, and `plt.show()` is guarded.
- `evaluate_and_plot_roc()` loaded the **meat**-trained checkpoint (`*_gnn_meat`) with a
  default-architecture `GNN()`, so `A-to-B --eval` actually computed C→B and duplicated the
  C-to-B configuration. It now loads `*_gnn_sim` with `arch="large"` and the matching sim
  test-loader stats, which is a genuine A→B. The reported AUC moves 0.6786 → **0.7314**; the
  ROC PDF is `roc_curve_A-to-B.pdf` (`cnn/gnn.py` already owns `roc_curve_sim.pdf`).
- `_retrained_path()` now tags artifacts with the configuration name, so training A→B then
  A→C no longer leaves only the latter's checkpoint (both route through `train_on_sim()`).
- `--train` died mid-epoch with `RuntimeError: received 0 items of ancdata`. Docker's default
  soft `nofile` limit is 1024, and torch's `file_descriptor` sharing strategy passes one fd per
  shared tensor, so 16 DataLoader workers exhaust it. `docker-run.sh` now passes
  `--ulimit nofile=65535:524288`. **A container started before that change keeps the old
  limit** — `docker stop difftactile && ./docker/docker-run.sh` to pick it up. Check with
  `docker exec difftactile bash -lc 'ulimit -Sn'` (expect 65535, not 1024).
- Training on the **simulated** dataset was disabled by a bare `return` at the top of
  `cnn/gnn.py::main()` on *every* branch, so the sim-trained models (A→B, A→C) could not be
  reproduced. `segmentation_gnn.train_on_sim()` now implements this, dispatched by `--train`.
- `evaluate_and_plot_roc()` raised `TclError` under `DIFFTACTILE_HEADLESS=1`: `_show_plots()`
  guarded `plt.show()`, but `plt.figure()` had already tried to open a Tk window. `segmentation_gnn.py`
  now selects the `Agg` backend before importing pyplot when there is no display.

Interactivity is otherwise pervasive and intentional: the cv2 annotation GUI and the tkinter
marker-labelling GUI still block by design. The **simulator and all three GNN scenarios now run
headless** under `DIFFTACTILE_HEADLESS=1`; the Docker image also passes X through, so GUI windows
work when a display is available.

## Data availability — the main gotcha

**None of the datasets or trained checkpoints are in this repository.** `.gitignore` excludes
`*.npz`, `*.pkl`, `*.pt`, `*.mp4`, `*.mkv`, `*.csv`, `output/`, `saved_models/`, `logs/`.
The paths below do not exist in a fresh clone:

- `difftactile/output/` — every intermediate artifact the sim writes
- `saved_models_meat/`, `saved_models_sim/` — trained GNN weights
- `difftactile/manual_or_experimental_data/meat_training_data/clean/` — real meat-experiment trials
- `difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense` — silicone dataset

So most ML entrypoints **cannot run in a fresh clone** — they will raise `FileNotFoundError`.
Do not "fix" such a failure by inventing data. Restore the published bundle instead:

```bash
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz   # ~190 MB from Zenodo
./data/restore_data.sh --verify                  # check what is present
```

> **The bundle must be the post-rename one.** Its paths — and the names stored *inside* its
> pickles and checkpoints — match the current code exactly. The fallbacks that let a
> pre-rename archive (`endgame/`, `iros_training_data/`, `saved_models_{iros,icra}/`) restore
> into this layout have been removed now that the rebuilt bundle is the published artifact;
> they are in the git history if ever needed. A restore that reports those old paths means an
> outdated archive — re-download from the DOI rather than renaming by hand, since the old
> bundle also carries stale names inside its pickles that a path rename cannot reach.

The published bundle is the Zenodo record **shallow-vessel-palpation-dataset**, DOI
**10.5281/zenodo.21900934**. The file attached to that record is named
`shallow-vessel-palpation-data.tar.gz` — the same name as the local copy and as
`make_data_bundle.sh`'s default output — so the download URL is:

```
https://zenodo.org/records/21900934/files/shallow-vessel-palpation-data.tar.gz
```

Note the record *title* and the *filename* differ by one word (`-dataset` vs `-data`); the URL
uses the **filename**. If the attached file is ever renamed, the `wget` lines in `README.md`
and `docker/run_pipeline.sh` must be updated to match.

### Restoring without downloading from Zenodo (local/offline route)

`restore_data.sh` takes **any local path** — it never contacts Zenodo itself, so the download
step in the README is only a convenience. Both of these are supported and are the route to use
for local testing (e.g. while the Zenodo record is still an unpublished draft):

```bash
./data/restore_data.sh /path/to/shallow-vessel-palpation-data.tar.gz   # local tarball
./data/restore_data.sh /path/to/shallow-vessel-palpation-data          # already-unpacked dir
```

On the author's machine the exact bundle uploaded to Zenodo is kept at:

```
/home/psb120/Documents/phd/data/masters/zenodo-bundle/shallow-vessel-palpation-data.tar.gz
```

with a `.sha256` sidecar next to it (`sha256sum -c *.sha256` to check integrity). Use that path
to test the full restore → train → eval flow without touching Zenodo.

Deliberately **not documented in the README** — end users should get the bundle from the DOI so
the published artefact stays the single source of truth.

If no bundle exists on disk, rebuild one from the author's frozen submission-state tree with
`./data/make_data_bundle.sh [SOURCE_DIR] [OUTPUT_TAR]` (defaults are in the script header).
Note that gzip output is not byte-reproducible, so a rebuilt tarball will not match the
published SHA256 even when its contents are identical.

`data/MANIFEST.md` documents what the bundle contains and — more usefully — what is
deliberately excluded (raw videos, intermediate preprocessing stages, training logs), which
is what takes it from 4.5 GB to ~190 MB. `data/make_data_bundle.sh` rebuilds it (author-side).
`data/ZENODO_UPLOAD.md` covers publishing it from the command line.

The **hardcoded absolute paths are gone** — everything now resolves through
`difftactile/main/paths.py`. Note that the author's own checkout wires some data paths up as
**symlinks** into an external directory; those resolve on the host but *not* inside the
container, which is why `restore_data.sh` replaces them with real files.

## Branches

### `main` is the only supported branch — work there

**Only `main` is used by end users and only `main` is maintained.** Unless the user explicitly
names another branch:

- **Make every change on `main`.** Do not create per-experiment branches, and do not revive,
  update or "fix" the historical ones — they are frozen snapshots.
- **Only `main`'s documentation is kept current.** Do not sync READMEs across branches.
- **Do not merge any historical branch into `main`.** They all predate it.

All three paper models train and evaluate from `main` (see above), so there is never a reason
to switch branches to reach a different model.

**Do not merge `sim-to-meat-test` into `main`** in particular. It predates `main` (no
`paths.py`, no Docker setup, no data-bundle scripts), so the merge deletes that infrastructure,
and `main` already carries its useful content — including a normalisation fix that raises the
reported cross-domain vein IoU from 0.034 to 0.198.

The historical branches, for reference only. Their **names are frozen snapshots** and are
deliberately left as-is by the rename — do not rename or rewrite them:

- **`iros`** — first submission state. Sim-to-real onto a **silicone** vascular phantom. GNN
  config is the small model (`latent_dim` 64, 30 epochs). Currently the same commit as `main`.
- **`sim-to-silicone`** — the same commit as `iros`; the silicone experiment as submitted.
- **`sim-to-meat-test`** — obsolete. Transfers the model to **real meat** with plastic straws
  as pseudo-vessels. Differs in: a much larger GNN (`latent_dim` 256, `small_input_dim` 248,
  `skip_dim` 128, 1 epoch, batch 16), `dataset.py::create_splits_meat` routes all trials to the
  **test** split, `preprocess_silicone_data.main()` re-enables the full cleaning pipeline, and
  `script_segmentation_gnn.py` calls `main()` instead of `evaluate_and_plot_roc()`. Superseded by the
  `A-to-C` configuration.

Tag `upstream-difftactile` preserves the pristine upstream DiffTactile state that `main`
formerly pointed at.

Only `main`'s README is maintained. The historical branches keep whatever README they were
frozen with — do not update them to match.

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
- **Name things after the data or the model, never after a venue or a project phase.** Artifacts
  are named for the dataset they belong to — `sim` (A), `silicone` (B), `meat` (C) — and models
  for their size, `compact` (`latent_dim` 64) vs `large` (`latent_dim` 256), with the large
  variant's hyperparameters under the `*_large` keys of the `gnn` config block. Earlier revisions
  used conference names (`iros`, `icra`) and a timeline word (`endgame`) for these; those are
  gone from the code. The only survivors are load-bearing and deliberate: the historical git
  branch names, the `git clone --branch iros` line in the verbatim transcript in
  `REPRODUCTION_TEST.md`, and the external snapshot path in `make_data_bundle.sh` (a real
  directory outside this repository). Do not "clean up" any of those.
