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
docker exec -it vessel-palpation ./docker/run_pipeline.sh check
docker exec -it vessel-palpation ./docker/run_pipeline.sh A-to-B
```

**Claude Code: test through the Docker container.** It is the only environment with the full
stack (Taichi included). Fall back to the bare-metal `/home/psb120/micromamba/envs/claude`
env only if the container fails or is too much hassle — note it has **no Taichi**, so the
simulator cannot run there and `run_pipeline.sh check` will fail partway through. That env
installs `opencv-python-headless` on purpose; leave it that way.

**The one exception is `docker/annotate_data_bare_metal.sh`, which runs on bare metal by design.** It
drives the two interactive annotation viewers, which are hand-operated frame by frame, and
X-socket forwarding out of the container makes them choppy enough to be useless as a debugging
tool. `docker/annotate_data_docker.sh` is its **in-container twin** — same viewers, same
modules, same PySide6/PyAV versions, so the two can be run side by side and any difference
attributed to the container rather than the code. It is a comparison and debugging aid; bare
metal remains the normal way to annotate. It defaults to a **native Wayland** window (no
Xwayland), which is what makes the comparison fair; `--x11` forces the xcb path instead. See
"Both display transports in the container" below. `annotate_data_bare_metal.sh` itself is unchanged, and
the docker twin `exec`s into it for everything that is not display-related, so the viewer logic
lives in exactly one place.

`annotate_data_bare_metal.sh` uses its own dedicated micromamba env, `vessel-palpation-annotator`, defined by
`requirements/annotator-env.yml` and created with
`micromamba env create -f requirements/annotator-env.yml`. That env is deliberately minimal —
numpy, scipy, tqdm, **PySide6**, **av** and headless OpenCV, with no torch, no torch-geometric
and no Taichi — which is why `predict_exp` (the module that drags in the whole GNN stack) is
imported *lazily* inside the two methods that need it in `preprocess_{silicone,meat}_data.py`
rather than at module scope. Do not hoist those imports back to the top: it would put torch
back on the annotator's critical path and break the small env. `qt_viewer` and `video_decode`
are imported lazily inside `annotate()` / `browse_annotations()` for the mirror-image reason —
so the batch preprocessing in those same modules stays importable without PySide6 or PyAV.

**Three viewers are Qt; everything else uses OpenCV.** The two annotation viewers, plus the
prediction viewer behind `docker/view_predictions.sh` (`cnn/visualise.py::visualise_gnn`). They
were moved to PySide6 because the `opencv-python` wheel ships only the `xcb` Qt platform
plugin, so every cv2 window on a Wayland desktop is an Xwayland client. PySide6 bundles the
Wayland plugins, so these run natively. Consequences worth knowing:

- The env now installs **`opencv-python-headless`**, and that is correct — cv2 is used only for
  image operations here, Qt owns the windows. This reverses the old "must be GUI-capable
  opencv-python" rule *for this env only*; the rest of the project is unchanged.
- `is_headless()` accepts `WAYLAND_DISPLAY` as well as `DISPLAY`, because a Wayland session
  with no Xwayland has `DISPLAY` unset while windows open fine.
- `display.py`'s `run_frame_browser()`, `fit_to_view()` and `view_width()` were **deleted**:
  they existed only for these two viewers. Qt fits the view to the window itself
  (`fitInView`), and the scene stays in full video resolution so clicks map back through
  `mapToScene()` with no manual rescaling. `DIFFTACTILE_VIEW_WIDTH` and
  `DIFFTACTILE_DOUBLE_PRESENT` are gone with them.
- Silicone annotation points are `QGraphicsEllipseItem`s, not pixels drawn into the frame, so
  clicking one selects it and `Delete` removes that specific point.

### Both display transports in the container

`docker-run.sh` mounts **both** the X11 socket and the host compositor's Wayland socket, and
each application picks its own. Backend selection is per **process** (`DISPLAY` /
`WAYLAND_DISPLAY` / `QT_QPA_PLATFORM`), never per container, so there is no conflict and
nothing had to be traded off:

- **X11 stays the container-wide default.** `DISPLAY` is still set on `docker run`, because
  Taichi GGUI, Gmsh's FLTK viewer, the cv2 windows and matplotlib are X clients with no Wayland
  backend. On a Wayland desktop they reach the session's own Xwayland via `/tmp/.X11-unix`.
- **Wayland is opt-in**, taken by the two Qt entrypoints — `annotate_data_docker.sh` and
  `view_predictions.sh`. Each `unset`s `DISPLAY` in its own process and sets
  `QT_QPA_PLATFORM=wayland` with **no `;xcb` fallback**, so a broken setup fails loudly rather
  than silently degrading onto Xwayland — which would quietly invalidate the whole
  bare-metal-vs-container comparison. A stray inherited `DISPLAY` is the usual cause of that
  silent downgrade, hence the explicit unset. Both take `--x11` to force the other path.

Measured outcome on Ubuntu 24.04 / GNOME Wayland: the containerised viewer under **Wayland is
indistinguishable from bare metal**, while **`--x11` is visibly choppy** (an extra copy per
repaint through Xwayland). That is a known, accepted limitation — do not try to "fix" the X11
path; it is a fallback and a debugging switch, not the route anyone should annotate through.

### The prediction viewer (`view_predictions.sh`)

Structured like `annotate_data_docker.sh` — launched from the host, `exec`s into the container,
native Wayland by default, `--x11` and `--shell` available. Unlike the annotators there is **no
bare-metal twin**: it runs GNN inference, so it needs torch and CUDA.

**`--central` is the default; `--all` is the opt-in debugging view.** The model takes a
`clip_len`-frame window (`clip_len` is **5** since 2026-08-15; the ablation found 5 > 7) and predicts a label for *every* frame in it, but only
the **central** frame is ever reported: `dataset.py::get_mask()` marks exactly `clip_len // 2`,
and `segmentation_gnn.shared_step()` applies that mask in the `val`/`test` stages but **not** in
`train` (so training gets signal from all 7 frames), as does `evaluate_and_plot_roc()`. The
default is the reported view on purpose — if the two ever disagree about how good the model
looks, that is the one to see first. Passing **both** flags is still an error (in the shell
script and in `visualise.main()`): asking for both is contradictory, so neither silently wins.

- `--all` → `_MeatNavigator`, three levels (`i`/`o` trial, `j`/`k` clip, `n`/`m` frame), clips
  tiled **sequentially** so playback walks each trial once.
- `--central` → `_CentralFrameNavigator`, two levels (`i`/`o` trial, `j`/`k` central frame),
  clips cut as a **sliding** window so consecutive windows have consecutive centres and `j`/`k`
  is a real per-frame axis. Sequential clips here would give one prediction every 7 frames.
  The trial's first and last `clip_len // 2` frames are never a window's centre, so they are
  skipped — inherent to central-frame reporting, and accepted. Measured on the meat dataset:
  10 trials × ~20 central frames, video frames 3..22 of 26.

`visualise_gnn()` was reworked in three ways, all in `cnn/visualise.py`:

- **One Qt window, not five OpenCV ones.** The five panels (Ground Truth, Hard Prediction,
  Confusion Matrix, Soft Prediction, Metadata) are composited by `_compose_panels()` and shown
  through the same `qt_viewer.FrameBrowser` the annotators use. The old code placed five
  separate cv2 windows with `cv2.moveWindow()`, which **does nothing under Wayland** — a
  Wayland client cannot position itself, by design — so they piled up wherever the compositor
  put them. Doing the layout ourselves makes it backend-independent.
- **Stepped, not auto-played**, over the three levels the data actually has. `_MeatNavigator`
  owns the index arithmetic; one key pair per level, no key doing two jobs: `i`/`o` trial,
  `j`/`k` clip within trial, `n`/`m` frame within clip, `q` quit. Changing trial or clip lands
  on that unit's first frame; all moves clamp rather than wrap. (`g`/`G` are gone — they were a
  flat first/last on a flat list, which the nested structure made meaningless.)
- **All frames kept.** Collection used to discard everything but each clip's centre frame
  (`clip_len // 2`), which is why only clip-level stepping was possible; the per-frame axis
  needed them retained. The old `filter_left()` decimation is gone too — it kept 10 of every 20
  clips on the simulated dataset's "5 trajectories × 2 directions × 10 frames" assumption, and
  on meat that silently hid half the clips of every trial.
- **The Metadata panel is drawn at display time**, not baked in during collection, because its
  text depends on where you have navigated to. It reports trial `x/y`, the trial's description,
  clip `x/y`, the frames the clip covers as a **closed** interval `[first, last]`, and frame
  `x/y` within the clip. `_metadata_panel()` draws onto a `.copy()` — the underlying array is a
  slice of a per-clip stack that is revisited on every navigation, so drawing in place would
  stack text on text. The status bar deliberately carries **no** frame counter: it duplicated
  this and the two disagreed about what "frame" meant.
- **Dead code removed.** Everything after the `return` in the old display block — a second,
  abandoned stepping loop — is gone. It referenced `is_interactive()`, which was never
  imported, so it would have raised `NameError` had it ever run.

No PyAV here, deliberately: this path decodes no video. It renders from precomputed `.npz`
stacks and pickles, so there is nothing for a decoder to do. (The one `cv2.VideoCapture` left in
the file is in `visualize_experiment()`, a separate method not reachable from this entrypoint —
it also references the deleted `SegmentationModel`.)

**Import-order trap:** `run_browser` is imported *lazily inside the method*, and must stay
there. On Python 3.10 importing **PySide6 before torch** breaks torch — PySide6 ships a
`typing_extensions` that shadows the one torch needs, and `import torch._dynamo` then dies with
`TypeError: Plain typing.Self is not valid as type argument`. Importing at module scope would
put PySide6 first. The same lazy-import rule already applies to the annotation viewers, for the
separate reason that their small env has no torch.

### Metrics: AUROC and average precision (`curve_plots.py`)

`cnn/roc_plot.py` is now **`cnn/curve_plots.py`**, because it owns both curve types. It exports
`plot_roc()` and `plot_pr()`, styled as twins (same threshold colourmap, same marked operating
points) so a manuscript can place them side by side. The PR figure also draws its **chance
baseline** at the positive rate — unlike ROC's fixed diagonal that line moves with the dataset,
and a PR curve is unreadable without it.

**Every evaluation path reports both metrics.** `segmentation_gnn.score_ranking_metrics()` is
the single implementation: it collects probabilities over the same central-frame mask the
val/test stages use, computes AUROC + AP, and writes both figures. All three configurations call
it — `evaluate_and_plot_roc()` (A→B) delegates to it entirely, and `silicone_to_meat()` (A→C)
and `main()` (C→B) call it after their `trainer.test()`, which previously reported IoU at a
fixed threshold and nothing else. `auroc_all_scenarios.py` has its own copy of the collection
loop (it must, since it loads checkpoints itself) and reports the same pair.

**IoU is reported too, by the same function.** `marker_iou()` computes foreground (class 1,
vessel present) and background (class 0, vessel absent) IoU over the pooled scored marker nodes,
and `score_ranking_metrics()` returns them, so all three configurations print both. Previously
A→B `--eval` reported **no IoU at all**, and the other two emitted it only inside a Lightning
table as `test_iou/0` / `test_iou/1`, which never said which class was which.

Verified against that Lightning table on A→C: `test_iou/1` = 0.19748 = **foreground**,
`test_iou/0` = 0.80767 = **background** — so the 0.198 cross-domain figure quoted elsewhere in
this file is the foreground IoU. Keep `marker_iou()` and `compute_ious_acc()` in agreement; they
are independent implementations of the same quantity and their agreement is the check.

Note "foreground" is the **class**, not a side of the comparison: it is `|pred AND true| / |pred
OR true|` over the vessel-present class, needing both. Background IoU is always the flattering
number (the negative class is ~89% of nodes on silicone, ~93% on meat). And unlike AUROC/AP it
is **threshold-dependent** — it uses `DECISION_THRESHOLD`, not `MAP_DECISION_THRESHOLD`, since
it is a reported metric rather than a figure.

**A→A is a first-class configuration**, not a special case of A→B: `run_pipeline.sh A-to-A`,
`score_all_scenarios.sh A-to-A`, `view_predictions.sh A-to-A`, and sweepable. `--eval` goes
through `evaluate_on_sim()`, `--train` through `train_on_sim(test_on="sim")`. It scores the same
published checkpoint as A→B/A→C; only the test set differs. The gap A→A → A→B is the sim-to-real
transfer cost: AUROC 0.958 → 0.779 (5-seed means at clip_len 5, sweep 20260815-130143; the
clip_len-7 pair was 0.9563 → 0.7719, the pre-regeneration pair 0.9369 → 0.7314).

**Best-of-five convention (2026-08-15).** All tables quote mean ± std over the five seeds of
`files.published_sweep` (`saved_models_sweeps/20260815-130143`). Wherever ONE model must be shown
— the prediction viewer, the bird's-eye maps — the best-of-five instance by AP is used
(`cnn/model_selection.py::best_model`; `sweep.json` records `best_seed`: A-to-A 1, A-to-B 2,
A-to-C 2, C-to-B 4). The published paths `saved_models_sim/` + `saved_models_meat/` (and their
pickles) hold the A-to-B and C-to-B best instances. **Legacy models** (`saved_models_legacy/`,
pre-2026-08-15, clip_len 7) exist only because they made the accepted version's Fig. 8/Table 4;
`--model legacy` / `--legacy` load them and export `DIFFTACTILE_CLIP_LEN=7`.

The simulated test split is taken from the test-loader **pickle**, never re-derived — the pickle
stores the actual `MyDataset(mode='test')` the checkpoint was held out from, so rebuilding it
would risk scoring on trajectories the model trained on.

`train_on_sim()` still splits A 70/15/15 and, for A→B and A→C, reports that in-domain reference
before the cross-domain result (`in_domain_` prefixed keys).

**The A split is mechanical, not stratified**: filenames sorted, cut at 70%/85%. This is fine
because **every trajectory has exactly one vein** (`vein_polyline` is `(frames, 1, 50, 2)` in all
500), so there is no vein-count variation to stratify over. Vein-present frame fraction is
0.5019/0.5029/0.5017 across train/val/test. Splitting is by *trajectory*, so overlapping sliding
windows cannot leak between splits.

Deliberately the **test** split and not `val`: val drives `EarlyStopping` and
`ModelCheckpoint(save_top_k=1)`, so a number read off it is optimistic by construction. Only the
sim-trained configurations have this — C→B trains on meat, and `sweep()` omits the section
entirely rather than emitting an empty table. The metrics are carried in the same flat dict
under an `in_domain_` prefix, which is what lets the sweep summarise both side by side.

**Why both, and why not a tuned threshold.** Both are ranking-based: they read only the *order*
of the probabilities, never their scale. In a sim-to-real project the scale is the first thing to
shift across a domain gap, so a single-threshold score confounds "doesn't know where the vessels
are" with "knows, but is miscalibrated here". AUROC normalises false positives by the huge
negative total, so it can look reassuring where precision is poor; AP ignores true negatives, so
it cannot be flattered that way, but its baseline is the positive rate rather than 0.5 — hence
the `Chance` and `Lift` columns in `AUROC_RESULTS.md`. Measured on the published checkpoints the
two **disagree in rank**: C→B is worse than A→B on AUROC (0.679 vs 0.731) but better on AP
(0.287 vs 0.255). That disagreement is the reason for reporting both — do not drop either.

`docker/score_all_scenarios.sh` wraps `script_auroc_all_scenarios` so users need not type the
module path. ROC PDFs go to `difftactile/output/roc_curves/`, PR PDFs to `pr_curves/`.

**The decision thresholds are two named constants in `curve_plots.py`** — there are no
hardcoded probability cuts left anywhere in the project:

- **`DECISION_THRESHOLD = 0.5`** — the conventional, deliberately *untuned* cut. Used by the IoU
  logged in `shared_step()` (both `segmentation_gnn.py` and `gnn.py`) and by the two dead U-Net
  paths in `visualise.py`. Nothing reported depends on it, because AUROC and AP never see a
  threshold.
- **`MAP_DECISION_THRESHOLD = 0.58`** — now used ONLY by the viewer's Hard Prediction /
  Confusion Matrix panels (`visualise.py`). Overridable with `DIFFTACTILE_MAP_THRESHOLD`. It
  is an empirical "looked best on the old silicone map" pick, confined to that display.

**The bird's-eye vessel map uses NO fixed threshold** (since 2026-08-15). `vessel_map.py`
chooses each run's operating point: the threshold with the highest pixel-level recall among
those with precision ≥ 0.9, where a predicted pixel counts as correct within 3 mm of a true
pixel (`PRECISION_TARGET`, `PRECISION_TOLERANCE_MM`). At 0 mm the target is met only by 3–7
pixels (silicone) or 1 (meat) because the reprojected truth is sparse marker points, hence the
tolerance. If the target is unreachable with ≥ 20 predicted pixels the run falls back to the
F1-optimal threshold and flags it in `report.md` / `run.json` (measured: Sim→Meat). Override
with `--threshold` / `DIFFTACTILE_VESSEL_MAP_THRESHOLD`.

### Training is seeded and reproducible (`main/seeding.py`)

`seed_everything()` is called at the top of both training entrypoints
(`segmentation_gnn.main()` for C→B and `train_on_sim()` for A→B / A→C), **before** any dataset
or model is constructed — the weights are drawn at construction time, so seeding later would
not reach them. Default seed 42; override with `DIFFTACTILE_SEED`.

**The two SIMULATOR entrypoints are seeded too** — `main.py::main()` (training-data collection)
and `main.py::domain_adaptation_main()` (BO calibration). Both were unseeded until recently:
`NP_RNG` is `np.random.default_rng()` with no seed, and `seed_everything()` was called only
from the training path, so neither a collected dataset nor a DA run could be regenerated. Each
now calls it first thing, with `deterministic_torch=False` — there is no torch in the simulator
path, so the deterministic-kernel switches would only cost startup work and set
`CUBLAS_WORKSPACE_CONFIG` for nothing. Both print the seed they used.

What that reaches: `generate_trajectories()` / `generate_random_state_dicts()` (sensor poses,
press depths, slide directions), `randomise_contact_params()` (the per-trial sensor↔vein
contact pair), the `NP_RNG.permutation()` over `collision_ixs`, and in DA both `my_suggest_random`
and the per-iteration `tangential_stiffness` fraction. `NP_RNG` is the **only** Generator
constructed anywhere in the project and nothing draws from bare `np.random.*`, so reseeding it
in place covers the lot; there is no `ti.random` either, so Taichi adds no unseeded kernel
randomness. The GP was already deterministic via `BayesianOptimization(random_state=1)`.

**The currently published simulated dataset (`pickle_20260814_191137_reordered_dense`) was
collected seeded** (default seed 42, with `DIFFTACTILE_TRAJECTORIES=3 DIFFTACTILE_VEIN_PAIR=1
DIFFTACTILE_NUM_LOOPS=250`), so the same invocation should closely reproduce it — though
bit-wise identity across GPUs is unverified (Taichi CUDA kernels may not be deterministic).
The *previous* published dataset (`pickle_20250901_220921_reordered_dense`) predates seeding
and cannot be regenerated by any seed; it survives only in pre-2026-08-15 bundle archives.

Getting this right needed four things, not one. Seeding alone did **not** make training
reproducible, and a run that fixes only the obvious RNG will still drift:

1. **`torch` / `random` / `np.random`** — model init and misc.
2. **`NP_RNG`** (`main/constants.py`) — augmentation, dataset shuffles, per-epoch subset choice.
   Reseeded **in place** via `bit_generator.state`, *not* by rebinding `constants.NP_RNG`:
   `dataset.py` does `from ...constants import *`, so it holds its own reference and a
   rebinding would leave it drawing from the original unseeded stream. Do not "simplify" this.
3. **DataLoader workers** — `worker_init_fn=seed_worker` on every train/val loader. Forked
   workers inherit a *copy* of `NP_RNG`; without this they would all draw identical
   augmentations (16-fold repetition within a batch), which is worse than nondeterminism.
4. **Deterministic CUDA kernels** — the one that actually closed the gap. `GINEConv` and
   `global_add_pool` scatter with CUDA atomics, and atomic float addition is not associative,
   so identical inputs gave different gradients. Needs `cudnn.deterministic`,
   `use_deterministic_algorithms(True, warn_only=True)` **and** `CUBLAS_WORKSPACE_CONFIG`
   (cuBLAS raises at the first matmul without it). `warn_only` because some torch-geometric
   scatter paths have no deterministic CUDA implementation — raising there would make training
   impossible rather than reproducible.

Verified: two consecutive `run_pipeline.sh C-to-B` runs are **bit-identical** (IoU to 16 s.f.,
AUROC 0.6043, AP 0.2425), and A→B `--train` likewise. Different seeds still differ, so the
seeding is live rather than frozen.

**Measured seed sensitivity is large — quote it before quoting a single run.** Seven seeds on
C→B gave AUROC **0.604–0.779** and AP **0.239–0.342**, a spread wider than any gap the paper
claims between configurations. The meat training set is 139 clips, so which subset a run favours
dominates. A single-seed number is not a result here; report a mean and spread, and never select
the best-scoring seed.

`cnn/seed_sweep.py` automates that: `./docker/score_all_scenarios.sh --seeds N` trains each
configuration once per seed and appends a **Seed sweep** section to `AUROC_RESULTS.md` (mean ±
std, range, and every per-seed value). Design points worth keeping:

- **Each seed runs in a subprocess.** In one process the second run would inherit CUDA context,
  cuDNN autotuning state and torch's global RNG from the first, so "seed *k*" would not mean the
  same thing mid-sweep as standalone — which would defeat the purpose. Verified: sweep values
  match a manual `DIFFTACTILE_SEED=k` run exactly.
- **The child clears `sys.argv`.** `run_scenario()` parses argv itself and rejects unknown
  flags, so the worker's own `--child <config> <seed>` would abort the run. Both are passed as
  explicit arguments instead.
- **Metrics cross the process boundary as JSON behind `METRICS_MARKER`**, not by parsing the
  last line — Lightning's output is verbose and its format is not ours to depend on. This is why
  `main()`, `train_on_sim()` and `silicone_to_meat()` all now **return** their metrics dict.
- **`write_markdown()` replaces only the `## Seed sweep` section**, so the scenario table from
  `auroc_all_scenarios` survives and re-runs do not accumulate copies.
- **Every seed's weights are kept**, under `saved_models_sweeps/<timestamp>/<config>_seed<NN>/`.
  `DIFFTACTILE_ARTIFACT_DIR` (read by `_retrained_path()`) redirects a training run's artifacts
  there; the **basename is preserved** so everything that reads these files still recognises the
  `*_sim` / `*_meat` naming. The checkpoint and its test-loader pickle are always written and
  read as a pair — the pickle carries the normalisation stats, and mismatching them is a silent
  wrong answer, not an error. New timestamp per sweep, so sweeps accumulate; nothing prunes them.
- `sweep.json` beside the weights repeats every metric plus each checkpoint's path, so a sweep
  directory is self-contained.

The sweep summarises **AUROC, AP and both IoU values** as mean ± std, range and per-seed rows.

Variance is very uneven across configurations, which is itself a reportable finding: over three
seeds A→C is ±0.0008 (large simulated training set) while C→B is ±0.0208 — 25× more. **IoU is
much less stable than the ranking metrics**, as expected of a threshold-dependent metric: C→B's
background IoU has std ±0.1184 over three seeds, an eighth of the metric's whole range, so a
single-seed IoU there is close to meaningless.

**The prediction viewer shows ONE model, never a mean over seeds — do not "improve" this into an
ensemble view.** `view_predictions.sh --sweep TS [--seed N]` selects a specific seed's model
(checkpoint and stats pickle from the same per-seed directory). Averaging N models' predictions
would display an *ensemble*, which is a different model whose AUROC/AP are not the numbers any
table in this project reports — the figures would then contradict the results. Ensembling is a
research direction needing its own entrypoint and its own reported row, not a display toggle.
The viewer's job is "what does *this* model see on *this* frame"; the seed spread is already
quantified numerically in `AUROC_RESULTS.md`, which answers the variance question better than a
heat-map would.

### Manuscript artifacts: which script makes which figure

Recorded so a future change knows what it might break. The README quickstart carries the same
table for end users.

| Script | Manuscript artifact |
|---|---|
| `alignment_figures.sh` | **Fig. 5** — sim (red) vs real (green) marker alignment, four interactions (50 % opacity, per-trajectory MAE in caption) |
| `score_all_scenarios.sh --seeds 5` | **Table 3** (IoU, mean ± std) and **Fig. 6** (mean PR curves ± 1 std band; the ROC twins are generated but not in the manuscript) |
| `ablation_clip_len.sh` | the temporal-window ablation table (`tab:clip-len`) |
| `vessel_map_all.sh` | **Fig. 8** (Sim→Sim, Sim→Meat, Sim→Silicone, Meat→Silicone) and the map-space rows of **Table 4** (per-pixel TP/FP/FN/TN, MCC, F1, P, R, FG/BG IoU, AP, mean L2 at 0 mm growth) |
| `script_frame_space_metrics` (`cnn/frame_space_metrics.py`) | the video-frame-space rows of **Table 4**: the same statistics per marker, pooled once over all frames, best-of-five instances (`FRAME_SPACE_METRICS.md`) |

The former **Fig. 7** (per-frame prediction grid) was removed from the manuscript;
`view_predictions.sh` remains an interactive tool. The **accepted version's** Fig. 8 / Table 4
were made with the LEGACY models (`saved_models_legacy/`, see below), Fig. 8(b) with yellow
shapes added by hand; the current Fig. 8 / Table 4 come entirely from `vessel_map_all.sh`.

`domain_adaptation.sh` is new: `Contact.domain_adaptation()` had always existed but was
reachable only by editing `main()`, so it had no entrypoint. `main.py::domain_adaptation_main()`
now provides one. Note it calls `generate_trajectories()`, which REPLACES the four training
trajectory types with the four DA interactions (press / twist_z / twist_x / slide) — the two
sets are different and share the `trajectory_names` attribute.

### Reading the 3D scene: which colour is which

Getting this backwards makes every screenshot say the opposite of what it shows, so:

| Colour | What it is | Field / draw site (`main.py::visualisation_update_gui`) |
|---|---|---|
| **Green** `(0,1,0)` | **the ViTacTip SENSOR** | `sensor_points`, from `vitactip.vertices_deformed_A` |
| **Blue** `(0,0,1)` | **the PHANTOM** (healthy tissue) | `healthy_tissue_points`, from `phantom.particles_A` |
| **Yellow** `(1,1,0)` | the subsurface **vein** | `vein.particles_A`, drawn only when `2 in collision_ixs` |
| Magenta / small yellow | clock-arm keypoints | `clock_arm_points_3d`, `per_vertex_color` |

The sensor is the deformable body being *measured*, so it is the one that fills most of the
frame; the camera tracks it, which is why the PHANTOM appears to sweep across the view during
`slide` while the sensor stays roughly put. Do not confuse this with the **marker overlay**
convention used by Fig. 5 (`da_overlay_*.png`), where red = simulated markers and green = real
markers — that is a 2D image-space figure, unrelated to these 3D particle colours.

Because the vein's draw is gated on the same `collision_ixs` that gates its physics, the absence
of yellow in a frame is direct evidence that the vein was disabled for that run, not merely
hidden.

### The BO search space is deliberately constrained (`bo_gp.py`)

Two properties are imposed by restricting the SPACE rather than by trusting the search, and
both are admitted hacks in the service of reproducibility over robustness. The full reasoning
and the measured numbers live in `BoGp.__init__`; the summary:

**1. Log-transform then min-max for scale-like parameters.** `BoGp.LOG_SCALED` names the four
parameters that span decades (Young's modulus, the two contact stiffnesses, contact damping);
each maps to the unit cube through its logarithm, so every decade gets an equal share of the
range the GP searches. Poisson's ratio and the friction coefficient are bounded ratios on a
natural additive scale and stay linear. A log needs a strictly positive lower bound, so the
three contact coefficients are floored at `5e-2` instead of `0` (zero contact stiffness is
degenerate anyway); `_validate_bounds()` raises if a log-scaled bound is ever set non-positive.
Draw the initial design in NORMALISED coordinates — sampling raw values and taking the log
afterwards reintroduces exactly the linear spacing this removes.

**2. The sensor is stiff by construction.** `vitactip_youngs_modulus` ∈ [1.0e6, 2.0e6] (7×–14×
the nominal 139300) and `vitactip_poissons_ratio` ∈ [0.42, 0.45]. The BO otherwise converges on
very soft sensors, which reproduce marker positions by draping and dragging — the wrong
mechanism, since the real sensor/phantom interface is well lubricated and the deformation is
dominated by the press-like normal component.

**Why ν moved AWAY from incompressible, which looks wrong but is the point.** The visible
symptom was the sensor TIP lagging the BASE during `slide` with no contact involved: the base
is a hard kinematic constraint (`update_external_forces()` overwrites `is_fixed_layer`
velocities with `vertex_control_velocities`) while the tip only catches up as elastic waves
travel down the mesh. That lag is a SHEAR response, governed by `c_s = √(μ/ρ)`. As ν → 0.5,
`c_p` diverges (and `c_p` sets the CFL limit) while `c_s` barely moves — so
near-incompressibility paid the whole timestep cost and bought nothing against the lag. Backing
ν off frees the headroom to raise E 4×, and E is what raises `c_s`:

| configuration | `c_s` | tip catch-up |
|---|---|---|
| E=1.39e5, ν=0.497 (nominal) | 6.5 m/s | ~460 timesteps |
| E=2.0e6, ν=0.45 (current) | 25.0 m/s | ~120 timesteps |

against a `slide` of ~398 timesteps. **Do not "restore" ν towards 0.5** on the grounds that
silicone is nearly incompressible — that is what caused the lag, and it would cut the
affordable E by ~4× at this `dt`.

**The upper bounds are set by the timestep, not by taste.** `contact.dt_override` is fixed at
1e-5 s and CFL needs `dt ≤ dx/c_p`, so the stiffest corner is the binding case. Measured
C = 0.357 at E=2.0e6/ν=0.45, *below* the C = 0.363 the nominal configuration already runs at —
so nothing slows down. **CFL is necessary, not sufficient**: the corner is also verified
empirically (all four trajectories complete, every sensor vertex finite, with all three contact
coefficients at 5e2 and friction at 1.0), and the minimal tip lag was confirmed by watching a
full `slide` in the GGUI window. If a crash ever appears here the sensor is too stiff for the
fixed `dt` — halve the offending side of each stiffness bound (for ν, halve its *distance to
0.5*, since 0.49 → 0.495 is stiffer, the wrong way) and re-test the corner.

`contact.dt` was **deleted** — it was dead, read nowhere; `dt_override` is the live key at all
four read sites (`main.py`, `vitactip.py`, `phantom.py`, `set_dt()`). The two documentation
JSONs were renamed to match.

**Domain adaptation is Bayesian optimisation, not differentiable simulation.** An earlier design
backpropagated through Taichi to fit the contact/material parameters; that was abandoned, and
**all** the machinery supporting it has been deleted — `clear_grad`, `update_grad`, `clamp_grid`,
`backward_pass_common_part`, `compute_marker_loss_1..5`, the phantom's `single_svd_grad` /
`svd_of_trial_deformation_gradient_grad`, every `needs_grad=True` field, the loss/batch-loss
fields, and the `meta.enable_grad` config key. Do not reintroduce any of it: BO treats the
simulator as a black box and needs only forward simulation.

`domain_adaptation()` is now a clean BO loop — propose parameters, replay the four interactions,
score by apex MAE, register with the GP. Two fixes were needed to make it work at all:

1. **`copy_frame()` in the forward loop.** `update()` advances
   `vertices_undeformed_A[frame] -> [frame+1]` across the sub-frames of one timestep;
   `copy_frame()` copies the result back to frame 0, where the next timestep reads from.
   `collect_training_data()` always called it, the DA loop never did (it relied on
   `memory_to_cache()`, which is a no-op — bare `return`). Without it every timestep restarted
   from the initial pose and the PID error froze at 0.0982.
2. **`visualisation_prepare_tactile_readout_data_fp()` in `compute_da_loss()`.** Despite the
   name this is the *projection* step that fills `sim_markers_deformed`; the tactile-readout
   rendering merely reuses it. Headless runs skipped it, so the MAE scored against an all-zero
   array — a constant **1169 px** that no parameter change could move, which silently made BO
   meaningless. Calling it explicitly decouples the measurement from rendering.

Other things worth knowing:

- **Diverged parameter sets are scored, not fatal.** An extreme set makes the FEM solve blow up
  and markers return NaN (`linear_sum_assignment` then raises "matrix contains invalid numeric
  entries"). That is caught, penalised with the top of the target range so the GP learns to
  avoid the region, and the search continues.
- **Each run is timestamped** under `difftactile/output/domain_adaptation/<stamp>/`, holding the
  BO history, convergence figure, Fig. 5 overlays for the best config, and
  `trajectories/iterNNN_<name>.npz` with the raw simulated/real markers per iteration.
- **Nothing writes results back into `system-params.json`.** Adopting a configuration is manual
  and should stay that way.
- `DIFFTACTILE_BO_ITERATIONS` / `DIFFTACTILE_BO_RANDOM` control the search; ~35 s per iteration.

Measured: 5 iterations improved aggregated MAE 13.88 -> 11.40 px, best found by the acquisition
function (manuscript: 14 -> 13.5 px over a longer run). No regression from the grad removal —
`sim-short` collection completes, A->B eval is unchanged (0.7314 / 0.2553 / 0.2059), C->B
training reproduces 0.6043 / 0.1313.

### The vessel map (`vessel_map.sh`, `vessel_map_all.sh`) and the confusion colour scheme

Rewritten 2026-08-15 into `data_analysis/experiment/vessel_map.py` (`predict_exp.py` keeps
only the video→npz preprocessing helpers). Read that module's docstring first. Summary:

- **Every configuration**: `A-to-A` (one simulated slide with recorded poses, see below),
  `A-to-B` / `C-to-B` (the ten silicone sweeps on the published 180 × 100 mm grid, byte-for-byte
  the old geometry), `A-to-C` (one map per meat trial, all on one fixed grid; slide along robot
  −y, sensor assumed undeformed → plane 19 mm from the lens). 1 px = 1 mm everywhere.
- **Ground truth** `--ground-truth video|photo`: reprojected per-marker labels (default; called
  `simulator` for A-to-A) or the silicone top-view photo restricted to the swept region.
- **Model**: `best` (default; best-of-5 by AP from `files.published_sweep`, via
  `cnn/model_selection.py`) or `legacy` (needs clip_len 7 — the shell exports it).
- **Output** is versioned: `difftactile/output/vessel_maps/<train>-to-<test>_gt-<src>/<TS>/`,
  never overwritten. Per map: prediction/ground-truth masks, `confusion_rNN.png/.pdf` for the
  truth grown by an L2 disc of NN = 0, 1, 2 mm, `metrics_by_radius.md` (TP FP FN TN MCC F1
  precision recall accuracy + L2 median/mean/deciles), `l2_distances_rNN.png`; run-level
  `report.md`, `run.json`, `threshold_selection.png`. Recall at r > 0 is against the GROWN
  region (so it falls with r) — literal to the spec, stated in the manuscript.
- **Sim→Sim needs poses**, which the published dataset lacks: `main.py::vessel_map_trajectory_main`
  (`docker/vessel_map_sim_trajectory.sh`, seed 2026) simulates one vein-present slide recording
  `T_BA` per frame + the vein centreline; the reprojection chain (pixel → plane 16 mm → B via
  T_EB → A via T_BA → mm at ×200) was verified by reprojecting the recorded vein pixels back onto
  `vein_centreline_A` exactly. The recorded pixel convention needs NO y-flip before the fisheye
  inverse. Simulator length scale is **×5** (`meta.distance_scaling_factor`), not ×10.
- Video-vs-photo ground-truth IoU on silicone is now **0.2934** (clip_len 5 → more central
  frames; was 0.2640 at clip_len 7).

**One colour scheme, defined once** in `Visualisation.CONFUSION_COLOURS_RGB`:
green = both say vessel, **red = reference says vessel and the other does not (a miss)**,
**blue = the other says vessel and the reference does not (a false alarm)**, black = neither.
Red-for-misses is deliberate (a missed vessel is the dangerous error in palpation) and is the
*opposite* of the old scheme, which also used white for TP — do not "restore" it.

**`create_confusion_matrix_overlay()` returns float RGB**, for `plt.imshow()`. Anything writing
it with `cv2.imwrite` must go through **`confusion_overlay_bgr()`**, which does the flip. Handing
RGB straight to `cv2.imwrite` swaps red and blue, which on this scheme silently turns every miss
into a false alarm — that was a real bug in the old code, where the PNG and the PDF panel
disagreed. `confusion_legend_handles()` builds the legend from the same dict so it cannot drift.

Why containerised Wayland is fast: the entire client/compositor interface is that one Unix
socket, so a bind-mount puts the container on the identical IPC path a host-native client uses
— no proxy, no relay, no nested compositor. Buffers pass as file descriptors over `SCM_RIGHTS`,
which crosses namespaces intact, so frames are shared zero-copy; input arrives on the same
socket via `wl_seat`, needing no `/dev/input` access.

To verify which transport a viewer actually got, run `xlsclients` on the **host** while it is
up: in Wayland mode it must *not* be listed. The script also prints Qt's resolved platform at
startup.

Two smaller things `docker-run.sh` now does, both for the X11 side: it passes
`--hostname "$(hostname)"` so X clients' `WM_CLIENT_MACHINE` matches and the compositor can tie
a window to a local PID, and it finds Mutter's randomly-named
`$XDG_RUNTIME_DIR/.mutter-Xwaylandauth.*` cookie when `XAUTHORITY` is unset (normal on GNOME
Wayland), mounting it at `/tmp/.Xauthority` where the non-root container user can read it.

**A container keeps the mounts it was created with**, so one started before this change has no
Wayland socket. `docker stop vessel-palpation && ./docker/docker-run.sh` to pick it up; the
script detects the missing socket and says exactly this rather than letting Qt fail obscurely.

The image gained PySide6 and PyAV (Dockerfile section 9) purely for this script, pinned to the
same versions as `requirements/annotator-env.yml`. It also gained the `libxcb-*` packages
behind Qt 6's classic "could not load the Qt platform plugin xcb" error — the Wayland
libraries were already there. Note the image keeps the **GUI-capable** `opencv-python`, unlike
the annotator env's headless wheel, because the rest of its pipeline still opens cv2 windows.

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
| `DIFFTACTILE_HEADLESS=1` | Skip creating Taichi GGUI / Gmsh FLTK / cv2 windows entirely. Implied when `DISPLAY` is unset. |
| `DIFFTACTILE_INTERACTIVE=1` | Opt back in to **blocking** GUI windows (`plt.show()`, `cv2.waitKey(0)`, Gmsh FLTK, the tkinter labeller). Off by default: no script waits on user input, so unattended runs always terminate. See `difftactile/main/display.py`. |
| `DIFFTACTILE_MAX_FRAMES` | Frames a non-interactive viewer loop steps through before returning (per-loop defaults apply otherwise). |
| `QT_QPA_PLATFORM` | Force the Qt annotation viewers onto a platform plugin (`xcb` for X11). Unset, Qt picks `wayland` natively. `annotate_data_docker.sh` sets it itself (`wayland`, or `xcb` under `--x11`), so pass `--x11` there rather than this. |
| `DIFFTACTILE_ANNOTATOR_PYTHON` | Interpreter for `docker/annotate_data_bare_metal.sh`, instead of the `vessel-palpation-annotator` env. |
| `DIFFTACTILE_TRAJECTORIES` | Comma-separated trajectory types to collect (0 press, 1 slide-vein, 2 twist-y, 3 twist-z). Default all four. **The published dataset is entirely type 3** — use `3` to reproduce it. |
| `DIFFTACTILE_VEIN_PAIR=1` | Enable the sensor↔vein contact pair on the first of each loop's two substeps, so a trajectory runs once **with** a subsurface vein and once **without**. The vein half is hard-disabled in the committed default (`if False and j < 1` in `main()`), so every substep otherwise runs vein-free. |
| `DIFFTACTILE_SCENARIO` | Configuration name (`A-to-B`, `C-to-B`, `A-to-C`, or a legacy alias), if not passed as an argument. |
| `DIFFTACTILE_MODE` | `train` or `eval`, if not passed as `--train` / `--eval`. |
| `DIFFTACTILE_OVERWRITE_PUBLISHED` | `1` lets a training run overwrite the published checkpoints instead of writing `*_retrained` copies. |
| `DIFFTACTILE_SEED` | Seed for every RNG a training run touches (default 42). See `main/seeding.py`; seed sensitivity on C→B is large, so sweep rather than trusting one run. |
| `DIFFTACTILE_MAP_THRESHOLD` | Decision threshold for the viewer's hard-prediction panels only (default 0.58). Affects no reported metric. |
| `DIFFTACTILE_VESSEL_MAP_THRESHOLD` | Overrides the vessel map's chosen operating point (`vessel_map.sh --threshold`). |
| `DIFFTACTILE_MAP_CONFIG` / `_GT` / `_MODEL` / `_SEED` | What `vessel_map.sh` sets for `script_vessel_map` (configuration, ground-truth source, `best`/`legacy`, sweep seed). |
| `DIFFTACTILE_CLIP_LEN` | Override `gnn.clip_len` (the temporal window) for this process; positive odd integers only. Used by the clip-length ablation (`docker/ablation_clip_len.sh`), which trains A-to-B at lengths {1,3,5,7} and ranks them by foreground IoU. |
| `DIFFTACTILE_SIM_RAW_DIR` | Raw `pickle_<timestamp>` collection directory for `script_pre_process_sim_data` to Hungarian-reorder; writes `<input>_reordered_dense` beside it (the layout `files.sim_data` points at). |

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
  limit** — `docker stop vessel-palpation && ./docker/docker-run.sh` to pick it up. Check with
  `docker exec vessel-palpation bash -lc 'ulimit -Sn'` (expect 65535, not 1024).
- Training on the **simulated** dataset was disabled by a bare `return` at the top of
  `cnn/gnn.py::main()` on *every* branch, so the sim-trained models (A→B, A→C) could not be
  reproduced. `segmentation_gnn.train_on_sim()` now implements this, dispatched by `--train`.
- `evaluate_and_plot_roc()` raised `TclError` under `DIFFTACTILE_HEADLESS=1`: `_show_plots()`
  guarded `plt.show()`, but `plt.figure()` had already tried to open a Tk window. `segmentation_gnn.py`
  now selects the `Agg` backend before importing pyplot when there is no display.

### Nothing blocks on user input

**No script waits for a window to be closed.** `difftactile/main/display.py` is the single
policy for this; every blocking call in the project routes through it:

| Helper | Replaces |
|---|---|
| `wait_key(cv2, delay)` | `cv2.waitKey(...)` — a `0` ("wait forever") becomes a 1 ms poll, and long delays are capped at 30 ms |
| `imshow(cv2, ...)` / `destroy_windows(cv2)` | `cv2.imshow` / `cv2.destroyAllWindows` |
| `finish_plot(plt, path, **kw)` | `savefig` + `show` + `close` |
| `show_plots()` | the guard around a blocking `plt.show()` |
| `show_plotter(plotter, png)` | PyVista `plotter.show()` — renders a screenshot off-screen instead |
| `prompt(msg)` | `input()` — returns `""` rather than reading stdin |
| `iteration_limit(var, default)` | bounds a `while True:` frame browser that used to exit only on `q` |

Consequences worth knowing when editing:

- Figures are **always** written to disk; the window was only ever a convenience. Add new
  plots with `finish_plot()`, not a bare `plt.show()`.
- Viewer loops (`visualise.py`, `preprocess_silicone_data.py`, `fisheye_model_no_taichi.py`,
  `hungarian_exp.py`, `base_graph_connectivity.py`) advance on the "no key pressed" branch, so
  the old `elif key == ord('k')` next-frame case is now the `else`. Keep that shape — an
  unconditional `else` before an `elif` is a syntax error, and dropping it makes the loop
  redraw frame 0 forever.
- The two **manual-input** tools — `preprocess_silicone_data.py::annotate()` (mouse clicks)
  and `marker_tracker.py::VideoPlayer.run()` (tkinter) — return immediately with a printed
  note unless `DIFFTACTILE_INTERACTIVE=1`, since they produce nothing without a user. Existing
  annotations on disk are left untouched.
- `cv2` GUI calls are wrapped so an `opencv-python-headless` build (no GTK) warns once and
  continues rather than raising `cv2.error`.
- `DIFFTACTILE_INTERACTIVE=1` restores every blocking window, and `main.py`'s `HEADLESS`
  constant is now `display.is_headless()`, so it also covers an unset `DISPLAY`.

The Docker image passes X through, so the interactive opt-in works there when a display is
actually reachable.

## The meat dataset: 10 trials, descriptively named

The experiment recorded **23** runs; the dataset ships **10**. The other 13 were repeats of the
same condition at different sensor heights that no split ever referenced — `populate_clips_meat`
already filtered to `MEAT_TRAIN_TRIALS | MEAT_VALIDATION_TRIALS` — so they were removed rather
than shipped for an end user to puzzle over. `meat_experiment_spec.md` still lists all 23 as the
record of what was recorded, marking the ten that ship.

Trial directories are **`<description>-<timestamp>`**, e.g.
`2-metal-straws-beneath-2-steaks-20260228-235749`. Rules that matter when touching this:

- **The timestamp is the identity; the prefix is only a label.** The raw recordings are named by
  bare timestamp and `meat_experiment_spec.md` is keyed by it, so anything matching a directory
  against either keys on the timestamp — `_trial_timestamp()` /
  `dataset.py::meat_trial_timestamp()`. Never concatenate `output_dir / trial_id`; use
  `MeatPreprocessData._trial_out_dir()`, which resolves by timestamp and so also finds a
  directory named before the prefixes existed.
- **Reprocessing rebuilds the same names.** `_trial_dir_name()` derives the prefix from the
  spec's description, so a rebuild from raw reproduces the shipped layout instead of reverting
  to bare timestamps. The spec writes a bare "straw" for one metal straw and only ever says
  "silicone" explicitly, so the slug inserts both the count and "metal" to make them visible.
- **`meat_trial_description()` is duplicated on purpose.** `cnn/dataset.py` has it for the
  prediction viewer and `preprocess_meat_data.py` has its own copy (`_trial_description()`),
  because importing the former would drag torch onto the annotator's critical path and break the
  small `vessel-palpation-annotator` env. Four lines of string handling is cheaper than that
  coupling — do not "deduplicate" them.

## Data availability — the main gotcha

**None of the datasets or trained checkpoints are in this repository.** `.gitignore` excludes
`*.npz`, `*.pkl`, `*.pt`, `*.mp4`, `*.mkv`, `*.csv`, `output/`, `saved_models/`, `logs/`.
The paths below do not exist in a fresh clone:

- `difftactile/output/` — every intermediate artifact the sim writes
- `saved_models_meat/`, `saved_models_sim/` — trained GNN weights (best-of-five instances)
- `saved_models_sweeps/20260815-130143/` — the published five-seed sweep (all 20 checkpoints +
  pickles + `sweep.json`); `saved_models_legacy/` — the pre-2026-08-15 models
- `difftactile/output/vessel_map_sim/` — the one simulated slide with poses behind the Sim→Sim map
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

If no bundle exists on disk — or the data has changed and the published one is stale — rebuild
with `./data/make_data_bundle.sh [SOURCE_DIR] [OUTPUT_TAR]`. Run it **on bare metal** from the
repository root: it is plain bash and tar, so Docker adds nothing, and the container cannot see
the paths anyway.

**`SOURCE_DIR` defaults to this repository**, which is the authoritative copy of the data: the
meat trial rename and trim happened here, as did the checkpoint/pickle path renames. The
author's frozen submission-state snapshot is now an explicit opt-in rather than the default,
because it **predates all of that** — its meat data is under `iros_training_data/` as 23
bare-timestamp trials, and `saved_models_{sim,meat}/` and `test_loader_gnn_*.pickle` do not
exist under those names. Bundling from it used to succeed while silently omitting the meat
data, both checkpoints, both pickles and the silicone dataset.

That silent-omission failure mode is gone: `copy_tree` / `copy_file` now **collect** every
missing path and the script **refuses to write the tarball**, listing them all in one go. Each
bundled path is either unregenerable or hours of GPU time, so a gap is a broken artifact, not a
warning. `DIFFTACTILE_BUNDLE_ALLOW_MISSING=1` overrides for a deliberately partial build.

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
  `REPRODUCTION_TEST.md`, and the external snapshot path documented in `make_data_bundle.sh`
  (a real directory outside this repository — no longer that script's default, but still the
  literal path to pass when you want it). Do not "clean up" any of those.
