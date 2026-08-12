# Dead code / dead file analysis

**Status: ACTIONED — categories A, B and C were deleted on 2026-08-12.**

Removal was at file/folder granularity only; no partial edits were made to files that stay.
Two entries below were **overruled during verification** and kept, because reading the code
showed them to be live:

- `difftactile/main/cfl_and_contact_params_estimation.py` — `main.py:21` imports it. Only its
  `script_*` wrapper was deleted (the wrapper is what corrupts `system-params.json`).
- `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/` — listed under
  Category C, but it is `files.da_dir`, and `files.flat_sensor_default_state` names a `.jpg`
  **inside it**. That image is the sensor's default-state photo, loaded by `main.py:2279` and
  `vitactip.py:188,592`. Deleting the folder would break the simulator. Same class of mistake
  as the `downward-press-vascular-phantom` entry already caught below.

`.vscode/` needed no action — it was already untracked and gitignored.

After removal, all 70 remaining tracked `.py` files byte-compile, and the full Docker
quickstart (`check`, all three paper configurations, and `sim-short`) was re-run end to end
in a blank-slate clone — see `REPRODUCTION_TEST.md`.

The original review text follows unchanged.

*Updated after a second, independent pass; several entries below were verified
by reading the files rather than inferred from the import graph. Two items that
first looked dead turned out to be live and are now called out explicitly.*

Produced by building an AST-level import graph over all 106 tracked `.py` files
and cross-referencing every `script_*.py` wrapper, `run_all.sh`, the README and
CLAUDE.md. Analysis is at **file and folder level only**: files that are merely
*partly* unused (an unused method here and there) are deliberately not listed.

Judgement is anchored to the published pipeline:
`Taichi FEM sim → synthetic marker displacements → GNN → evaluation on real
silicone / meat data → ROC curves`.

---

## Category A — near-certain dead code

No importers, no entrypoint wrapper, and superseded by a named replacement.

| File | Why it looks dead |
|---|---|
| `difftactile/cnn/threshold_gnn.py` | Documented in CLAUDE.md as dead; no longer matches the `GNN` API. |
| `difftactile/data_analysis/experiment/roc_curve.py` | Plots a **synthetic** curve (`tpr = fpr**0.5`); the real ROC comes from `iros_gnn.evaluate_and_plot_roc()`. |
| `difftactile/data_analysis/experiment/calibrate_camera_2.py` | Second copy of the calibration script; neither imported nor wrapped. |
| `difftactile/data_analysis/experiment/real_data_hungarian.py` | Superseded by the marker reordering inside `endgame.py`. |
| `difftactile/data_analysis/experiment/traj_1_temporal_synch.py` | One-off synchronisation check for a single trajectory. |
| `difftactile/data_analysis/experiment/camera_temp_synch.py` | One-off camera/timestamp synchronisation experiment. |
| `difftactile/data_analysis/experiment/og_experiment_frames_poses_data.py` | Operates on the "og" experiment superseded by the endgame chain. |
| `difftactile/data_analysis/experiment/press_vein.py` | Early single-press probe, superseded by the endgame chain. |
| `difftactile/data_analysis/experiment/process_video.py` | Standalone video pass; `marker_tracker.py` / `endgame.py` do this now. |
| `difftactile/data_analysis/sim/rigid_mpm.py` | Experiment with a rigid-MPM object model that is not in the paper. |
| `difftactile/data_analysis/sim/merge_datasets.py` | Dataset-merging utility for a multi-dataset scheme not used in the published runs. |
| `difftactile/data_analysis/sim/domain_adaptation_line_chart.py` | Chart for the domain-adaptation side experiment, not a paper figure. |
| `difftactile/data_analysis/sim/gnn_input_visualisation.py` | Debug visualisation of GNN inputs. |
| `difftactile/data_analysis/sim/vitactip_photo_default_state_draw_circle.py` | One-off image annotation helper. |
| `difftactile/test/np_random_singleton.py` | Not imported; not a pytest test (no `test_` functions collected). |
| `difftactile/data_analysis/sim/3d_to_2d.py` | **Verified: the entire file is the single statement `pass`** (6 bytes). |
| `difftactile/sandbox/prompt.py` | **Verified: 0 bytes.** |
| `sitecustomize.py` (root) | **Verified: 0 bytes — and actively harmful to keep.** Python auto-imports any module named `sitecustomize` found on the path at interpreter start-up. |
| `difftactile/data_analysis/sim/sandbox.py` | Scratch on a hardcoded 3-element array; the file ends with `foo = 7`. |

## Category B — probably dead / legacy (lower confidence)

| File / folder | Why |
|---|---|
| `difftactile/ml_training_old/` (4 files) | Name says "old"; predates the current `cnn/` GNN work. **Verified un-importable:** three of the four do `from exploratory_data_analysis import *`, and that module does not exist anywhere in the repository — so they cannot even be imported, let alone run. |
| `difftactile/sandbox/` (WHOLE FOLDER) | Scratch area: the 4 `.py` above plus notes `.md`, a `.tex` and a small image. Last commits were "generate diagrams for the report". |
| `difftactile/cnn/train.py` + `scripts/script_train.py` | Legacy U-Net CNN path; CLAUDE.md records it as broken (outdated `MyDataset` signature, needs `monai`, which is not in `requirements/`). |
| `difftactile/cnn/lit_module_unet_cnn.py` | The U-Net Lightning module for the same legacy CNN path. |
| `difftactile/data_analysis/sim/benchmark_dataset.py` + `script_benchmark_dataset.py` | Broken: references a JSON key `dataset_root_reordered` that does not exist. |
| `difftactile/data_analysis/experiment/hungarian_exp.py` + `script_hungarian_exp.py` | Broken: references `experiment_straight_markers_npz_reordered`, which does not exist. |
| `difftactile/scripts/script_all.py` | Broken by import order (CLAUDE.md); `run_all.sh` is the working equivalent. |
| `difftactile/main/cfl_and_contact_params_estimation.py` + its script | Deliberately commented out of `run_all.sh`; writes scalars where `main.py` expects 3-element lists. |
| `difftactile/data_analysis/experiment/ml_exploratory_data_analysis.py` | Exploratory analysis, not part of the published pipeline. |
| `difftactile/data_analysis/experiment/calibrate_camera_rotation.py` | Calibration validation; a branch of the same name suggests it was exploratory. |
| `min_values.py`, `test_iou.py` (root) | Root-level scratch/util files. (`sitecustomize.py` is listed in Category A.) |
| `profile.out` (root, 117 KB) | cProfile output from a past profiling run. |
| `vitactip_cadquery.stl` (root, 101 KB) | Loose STL at the repo root; the meshes used by the sim live in `difftactile/meshes/`. |

## Category C — non-code files not needed to reproduce the paper

| Path | Size | Why |
|---|---:|---|
| `docs/box.gif`, `docs/pbd.gif`, `docs/repose.gif`, `docs/surface.gif` | **12 MB** | Upstream DiffTactile demo animations for the `box_open` / `object_repose` / `surface_follow` manipulation tasks — all of which were **deleted** from this fork. Referenced nowhere in the README or code. Largest easy win in the repository. |
| `difftactile/meshes/cylinder.stl`, `suturing-phantom.stl`, `thin-long-indenter.stl`, `vascular-phantom.stl`, `vascular-phantom-fragment-long.stl`, `vascular-phantom-fragment-shallow.stl` | ~30 KB | Only `vascular-phantom-fragment.stl` is referenced (by `files.phantom` in the config). The rest are alternative geometries from earlier experiments. |
| `difftactile/manual_or_experimental_data/mpv-shot0001.jpg`, `my_shape.png` | 280 KB | Loose screenshots, unreferenced. |
| `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/` (5 images) | 1.3 MB | Inputs to the domain-adaptation side experiment (`script_domain_adaptation`), not a paper result. |
| `.vscode/` | small | Personal IDE config with machine-specific interpreter paths. **Already untracked** as part of this work. |

## Corrections found while verifying — these are NOT dead

| File | Why it must stay |
|---|---|
| `difftactile/data_analysis/experiment/annotate.py` | Looked dead (no importer, no wrapper) but is the **sole producer** of `phantom_ground_truth_segmentation_mask`, which `predict_exp.py:72` reads. This was a genuine reachability bug: no `script_*` wrapper existed, so the documented `python -m difftactile.scripts.*` route could not run it. **Fixed** — `script_annotate.py` added, and its blocking `plt.show()` guarded so it works headless. |
| `difftactile/manual_or_experimental_data/downward-press-vascular-phantom/` (7.7 MB) | Initially a size-reduction candidate. It is `files.fisheye_model_image_dir` — the input `script_fisheye_model` uses to regenerate `init-marker-positions.npz`, a prerequisite for every dataset/train/eval run. **Keep.** |
| `difftactile/data_analysis/experiment/hungarian_exp.py` | The **module** is live (imported by `predict_exp.py`); only its `script_hungarian_exp.py` wrapper is broken. Delete the wrapper, not the module. |
| `difftactile/main/cfl_and_contact_params_estimation.py` | Diagnostics-only, but `main.py` imports it — removing it requires editing that import first. |
| `system-params-units.json`, `system-params-literature-values.json` | No code reads them, but they are the provenance for the material parameters. **Keep** — precisely what a reproducer wants. |

## Category D — keep (actively used)

Simulation core: `main/main.py`, `main/pre_main.py`, `main/apply_scaling.py`,
`main/constants*.py`, `main/paths.py`, `main/common.py`,
`main/synthetic_image_generator.py`, `main/generate_*_mesh_gmsh.py`,
`main/constants_bo_gp.py`.

Sensor / object models: `sensor_model/vitactip.py`,
`sensor_model/fisheye_model_{taichi,no_taichi}.py`, `object_model/phantom.py`,
`object_model/vein.py`, `object_model/obj_loader.py`, `object_model/common.py`.

ML: `cnn/iros_gnn.py`, `cnn/gnn.py`, `cnn/dataset.py`, `cnn/common.py`,
`cnn/visualise.py`.

Preprocessing / analysis: `data_analysis/experiment/endgame.py`,
`marker_tracker.py`, `iros_preprocess_data.py`, `adjacency.py`, `annotate.py`,
`base_graph_connectivity.py`, `bo_gp.py`, `domain_adaptation.py`,
`predict_exp.py`, `undistort.py`, `calibrate_camera.py`,
`data_analysis/sim/generate_test_loader_icra.py`, `pre_process_sim_data.py`,
`visualise_mesh.py`, `data_analysis/testing/prettify_confusion_image.py`,
`data_analysis/training/print_metrics_csv.py`.

Plus all `difftactile/scripts/script_*.py` wrappers for the above, `setup.py`,
and `difftactile/test/test_constants.py` (+ its wrapper).

---

## Suggested next step

Reply with which categories/files you accept, and I will delete exactly those.
If you want a fast, low-risk cut, **Category C's `docs/` alone removes 12 MB**
and is the clearest case: those animations belong to upstream tasks this fork
no longer contains.
