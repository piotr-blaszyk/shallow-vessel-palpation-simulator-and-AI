# Zenodo data bundle — contents and rationale

This document describes the archive published on Zenodo alongside this
repository, and — just as importantly — what is **deliberately left out**.

## Guiding principle

**Anything the repository's own scripts can regenerate is excluded.** The bundle
carries only artifacts that a user cannot produce themselves, either because
they are recordings of physical experiments or because regenerating them takes
impractically long.

Two kinds of exception are made:

1. **Generable but expensive.** The simulated training dataset (`script_main`,
   ~5 h 30 m of GPU time for the shipped 500-trajectory configuration), the
   published five-seed sweep of trained models, and the domain-adaptation BO
   run are all reproducible from the repository, but a user who only wants to
   verify the published results should not have to pay hours of GPU time.
   They are treated as non-generable and shipped.
2. **The manuscript's own figures and tables** (`manuscript_artifacts/`). These
   are cheap to regenerate and are the *only* things shipped purely for
   convenience: they are the exact figures and result tables the manuscript
   presents, so a reader can check any number against the archive without
   running anything. They are staged under their own directory rather than at
   their working-tree paths, so a restore never overwrites what a user's own
   runs have produced.

## Bundle paths vs. the currently published archive

The bundle must be the **post-rename** one: its paths, and the names stored
*inside* its pickles and checkpoints, match the current code exactly. The
compatibility shims that let a pre-rename archive restore into this layout were
removed once the rebuilt bundle became the published artifact; see the repository
history if you ever need to read a pre-rename bundle again.

> If a restore reports paths under `endgame/`, `iros_training_data/` or
> `saved_models_{iros,icra}/`, you have an outdated archive. Re-download it from
> the DOI rather than renaming files by hand — the old bundle also carries stale
> names inside its pickles, which a path rename cannot reach.

## What IS in the bundle

| Path (after restore) | Size | Why it cannot be regenerated |
|---|---:|---|
| `difftactile/output/training_data/pickle_20260814_191137_reordered_dense/` | ~250 MB | 500 simulated trajectories (250 with / 250 without the vessel), collected at the post-Bayesian-optimisation simulator parameters. Generable via `script_main` (~5 h 30 m GPU); shipped for convenience. |
| `difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense/` | ~684 KB | Real **silicone** phantom trials, fully preprocessed. Physical recording. |
| `difftactile/manual_or_experimental_data/meat_training_data/clean/` | ~480 KB | Real **meat** phantom trials, fully preprocessed (10 trials × 2 `.npz`). Physical recording. Directories are named `<description>-<timestamp>`, e.g. `2-metal-straws-beneath-2-steaks-20260228-235749`, so the trial's condition is readable without cross-referencing `meat_experiment_spec.md`. |
| `difftactile/manual_or_experimental_data/meat_training_data/clean/*/frames.mp4` | ~22 MB | The 26 camera frames per meat trial that preprocessing kept, H.264 CRF 26 (~2.3 MB each). Frame *i* corresponds 1:1 with row *i* of `marker_labels.npz`, so the annotation viewer can draw the ground-truth labels over the real meat images (paper Fig. `annotation-line`(d)). Cut down from the 1.6 GB raw recordings by `script_make_meat_clean_videos`; the raw videos themselves are a physical recording and are not shipped. |
| `saved_models_sim/final_segmentation_model_gnn_sim.pt` | ~4.2 MB | The simulation-trained (large) GNN checkpoint behind Sim→Sim, Sim→Silicone and Sim→Meat. It is the **best-of-five** seed instance (highest AP on Silicone) of the published sweep below. |
| `saved_models_meat/final_segmentation_model_gnn_meat.pt` | ~300 KB | The meat-trained (compact) GNN checkpoint behind Meat→Silicone; the best-of-five instance of the sweep. |
| `saved_models_sweeps/20260815-194045/` | ~70 MB | **The published five-seed sweep**: for each of the four configurations, five seeds (0–4), each with its checkpoint, its test-loader pickle (normalisation statistics — the two must be used together) and its per-seed score arrays; `sweep.json` records every metric and which seed is best-of-five. Every mean ± std in the manuscript, and the mean ROC/PR curves, come from here. Trained with a 5-frame temporal window (`gnn.clip_len` 5) on the regenerated simulated dataset. Retrainable in ~20 min of GPU time (`./docker/score_all_scenarios.sh --seeds 5`), shipped for convenience and because it is the artifact the results rest on. |
| `saved_models_legacy/{sim,meat}/` | ~4.6 MB | **Legacy models** (pre-2026-08-15): the original checkpoints + pickles from the accepted version. Kept only because they produced the accepted manuscript's bird's-eye map figure and localisation table; **not used for any main result**. Cannot be retrained (previous, unseeded simulated dataset; 7-frame window). See `saved_models_legacy/README.md`. |
| `difftactile/output/vessel_map_sim/{raw,raw_reordered_dense}/` | ~1.5 MB | The one simulated vein-present slide, with per-frame sensor poses (`T_BA`) and the vein's world centreline, from which the Sim→Sim bird's-eye map is drawn. Regenerable in ~2 min with a GPU (`docker/vessel_map_sim_trajectory.sh`, seed 2026); shipped so the map needs no simulator. |
| `difftactile/output/vessel_map_sim/test_trajectories/` | ~4 MB | The ten held-out simulated trajectories shown in the project page's Sim→Sim prediction video (7 vessel-present + 3 vessel-absent, fixed-seed draw, interleaved), each the **published** markers/labels plus the sensor poses re-simulated by replaying the dataset's seed (`docker/vessel_map_sim_test_trajectories.sh`, ~15 min GPU, verified against the published files); `selection.json` fixes the order. Behind the page's ten Sim→Sim bird's-eye maps (`docker/website_vessel_maps.sh`). |
| `difftactile/output/test_loader_gnn_sim.pickle` | ~60 KB | Normalisation statistics that the simulation-trained checkpoint was trained with; needed to evaluate it correctly. |
| `difftactile/output/test_loader_gnn_meat.pickle` | ~4 KB | As above, for the meat-trained checkpoint. |
| `difftactile/output/` sensor-geometry set (16 files) | ~5.5 MB | Sensor mesh, marker layout and graph connectivity: `base-graph-connectivity.npz`, `marker_locations_ordered.npz`, `init-marker-positions.{npz,pkl}`, `gmsh_mesh_{vitactip,vein}.pkl`, `vitactip_mesh.npz`, `edge_lengths.pkl`, `tactile_sensor.f2v.pkl`, `phantom_points.npz`, `vein_points.npz`, `is_fixed_layer.npz`, `grid_node_v0_mask.npz`, `initial_vertex_positions_undeformed.pkl`, `vitactip_points_E.pkl`. Regenerable (Gmsh + marker detection) but tiny, fixed, and required before the simulator will start. |
| `difftactile/output/marker_tracker/domain-adaptation-vascular-markers/` | ~1.3 MB | System-identification marker tracks (`traj_0..3_out.pkl`), loaded at simulator start-up to align the contact model against the real sensor. |
| `manuscript_artifacts/` → restored to `difftactile/output/manuscript_artifacts/` | ~8 MB | **The manuscript's figures and tables, as published — the one "generable but included" exception.** `figures/`: the Fig. 4 alignment panels (`alignment_{press,twist_z,twist_x,slide,all}.png`) and the Fig. 6 mean PR curves plus their ROC twins (`mean_{pr,roc}_curve_<config>.pdf`, five-seed mean ± 1 std) and the one shared legend and x-axis label set once for the row (`curve_legend.pdf`, `xlabel_recall.pdf`). `tables/`: `AUROC_RESULTS.md` (per-scenario metrics of the published checkpoints), `sweep.json` (every per-seed metric behind Fig. 6 and the best-of-five choice), `CLIP_LEN_ABLATION.md` (Table 3), `FRAME_SPACE_METRICS.md` (Table 4, upper half), `alignment_validation.json` (the MAEs beside Fig. 4). `vessel_maps/<train>-to-<test>_gt-<source>/`: the latest `vessel_map_all.sh` run of each of the six map configurations (Fig. 7, Table 4 lower half) with `confusion_rNN.{png,pdf}`, `metrics_by_radius.md`, `report.md`, `run.json`. `manuscript_artifacts/README.md` maps every file to its manuscript figure/table and to the repository path and script it came from. |
| `difftactile/output/domain_adaptation_published/` | ~12 MB | The published domain-adaptation BO run behind the adopted simulator parameters. `joint_bo/` is the 10-iteration joint search over sensor Young's modulus and sensor↔vein contact stiffness: `bo_joint_results.json` (every iteration, both reward terms and the best configuration), `iteration_log.csv` (the same, written live one line per iteration), `final_joint_validation.json` (all four trajectories at the winning configuration), the `da_overlay_*.png` alignment figures and the per-iteration `vein_iterNNN_overlay_slide.png`. Also holds the manuscript's white-background alignment panels (`alignment_{press,twist_z,twist_x,slide,all}.png`, drawn by `docker/alignment_figures.sh` at the adopted parameters), the `markers_*.npz` marker-position caches behind them (so the figures can be restyled without re-simulating) and `alignment_validation.json` with the per-trajectory MAEs. |

**Total: roughly 375 MB uncompressed / ~275 MB as a `.tar.gz`**, dominated by the
simulated dataset, the five-seed sweep and the meat camera frames.

Note the BO runs' `snapshots/` subdirectories are **excluded** — ~16 MB of rendered
PNGs per run, regenerable from the recorded parameters, and a debugging aid rather
than an artifact. The JSON histories and the overlay figures are what the reported
results rest on.

## What is NOT in the bundle (and why)

| Excluded | Size in the raw archive | Reason |
|---|---:|---|
| `meat_training_data/raw/` | 1.6 GB | Full-rate `.avi` recordings (429 frames each) + pose `.npz`. Only needed to re-run preprocessing from scratch; `clean/` is the product, and the 26 frames per trial that preprocessing actually keeps now ship as `clean/*/frames.mp4` at 3% of the size. |
| `meat_training_data/intermediate/` | 250 MB | Intermediate preprocessing stages; reproducible from `raw/`. |
| `meat_training_data/clean/*/marker_labels.avi` | 81 MB | Pre-rendered label overlays. Superseded by `frames.mp4` + `marker_labels.npz`, which the viewer composites on the fly for a fraction of the size. |
| `silicone_training_data/20250901-131547{,_interpolated_trimmed,_markers,_dilated,...}` | ~1.05 GB | Intermediate silicone preprocessing stages; only the final `_dense` directory is consumed. |
| `system-id-screws-*.mkv`, `vein_slide_across.avi`, `domain-adaptation-vascular-videos/` | ~960 MB | Videos used for system identification and exploratory analysis, not for the published results. |
| `difftactile/output/*` (meshes, voronoi, marker_tracker, debug images) | ~60 MB | All regenerated by `script_pre_main` / `script_generate_*_mesh_gmsh` / `script_marker_tracker`. |
| `difftactile/output/vessel_maps/` (all but the latest run per configuration), `da_recordings/`, `videos_raw/`, `pr_curves/`, `roc_curves/`, `alignment_figures/`, `da_snapshots*/`, `domain_adaptation/<other runs>/` and their `snapshots/` | ~1 GB | Superseded / intermediate runs and renders. Every run of `vessel_map.sh`, `record_da_trajectories.sh` and `domain_adaptation.sh` gets a fresh timestamped directory, so these accumulate; only the published run of each is shipped (see `manuscript_artifacts/` and `domain_adaptation_published/`). The demonstration videos are in the git repository (`videos/`), not the bundle. |
| `saved_models_sweeps/<other timestamps>/`, `saved_models_*/*_retrained_*` , `stale_artifacts_pre_new_dataset_20260815/` | ~200 MB | Earlier sweeps (clip_len 7, the pre-regeneration dataset), local `--train` outputs and pre-2026-08-15 artifacts, all superseded by the published sweep `20260815-194045`. |
| `difftactile/output/exp_probs.npz` | 424 KB | Cached per-marker probabilities for the bird's-eye vessel map. Pure model output: `./docker/vessel_map.sh` recomputes it from the shipped silicone dataset + meat checkpoint in **under a second**, bit-identically. Only `--cached` reuses it, and that flag is an optimisation, not a requirement. |
| `difftactile/output/phantom_ground_truth_segmentation_mask.jpg` and the other `*ground_truth*` images | ~150 KB | Regenerated bit-identically by `script_annotate` from two **git-tracked** inputs (`phantom-labels-vgg.json`, `phantom-uncropped-compressed-undistorted.jpg`), so they are in the repository already, not missing. The remaining `ground_truth_*` images are written by `predict_exp` itself during the same run. |
| `lightning_logs/`, `logs/` | 68 MB | Training logs from the original runs; recreated on any new training run. |

Excluding these takes the bundle from **4.5 GB to roughly 375 MB (~275 MB compressed)** without
affecting the reproducibility of any published result.

## Restoring

From the repository root:

```bash
./data/restore_data.sh path/to/shallow-vessel-palpation-data.tar.gz
```

The script unpacks into the correct locations and verifies each expected path
afterwards. See `data/restore_data.sh --help` for options.
