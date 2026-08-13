# Zenodo data bundle — contents and rationale

This document describes the archive published on Zenodo alongside this
repository, and — just as importantly — what is **deliberately left out**.

## Guiding principle

**Anything the repository's own scripts can regenerate is excluded.** The bundle
carries only artifacts that a user cannot produce themselves, either because
they are recordings of physical experiments or because regenerating them takes
impractically long.

The single exception is the **simulated training dataset**. It *is* generable
(`script_main`), but doing so takes about 2 h 45 m of GPU time, and a user who
only wants to verify the published results should not have to pay that cost. It
is therefore treated as non-generable and shipped.

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
| `difftactile/output/training_data/pickle_20250901_220921_reordered_dense/` | ~245 MB | 500 simulated trajectories. Generable via `script_main` but takes ~2 h 45 m GPU; shipped for convenience. |
| `difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense/` | ~684 KB | Real **silicone** phantom trials, fully preprocessed. Physical recording. |
| `difftactile/manual_or_experimental_data/meat_training_data/clean/` | ~480 KB | Real **meat** phantom trials, fully preprocessed (10 trials × 2 `.npz`). Physical recording. Directories are named `<description>-<timestamp>`, e.g. `2-metal-straws-beneath-2-steaks-20260228-235749`, so the trial's condition is readable without cross-referencing `meat_experiment_spec.md`. |
| `difftactile/manual_or_experimental_data/meat_training_data/clean/*/frames.mp4` | ~22 MB | The 26 camera frames per meat trial that preprocessing kept, H.264 CRF 26 (~2.3 MB each). Frame *i* corresponds 1:1 with row *i* of `marker_labels.npz`, so the annotation viewer can draw the ground-truth labels over the real meat images (paper Fig. `annotation-line`(d)). Cut down from the 1.6 GB raw recordings by `script_make_meat_clean_videos`; the raw videos themselves are a physical recording and are not shipped. |
| `saved_models_sim/final_segmentation_model_gnn_sim.pt` | ~4.2 MB | Trained silicone GNN checkpoint. |
| `saved_models_meat/final_segmentation_model_gnn_meat.pt` | ~300 KB | Trained meat GNN checkpoint. |
| `difftactile/output/test_loader_gnn_sim.pickle` | ~60 KB | Normalisation statistics that the simulation-trained checkpoint was trained with; needed to evaluate it correctly. |
| `difftactile/output/test_loader_gnn_meat.pickle` | ~4 KB | As above, for the meat-trained checkpoint. |
| `difftactile/output/` sensor-geometry set (16 files) | ~5.5 MB | Sensor mesh, marker layout and graph connectivity: `base-graph-connectivity.npz`, `marker_locations_ordered.npz`, `init-marker-positions.{npz,pkl}`, `gmsh_mesh_{vitactip,vein}.pkl`, `vitactip_mesh.npz`, `edge_lengths.pkl`, `tactile_sensor.f2v.pkl`, `phantom_points.npz`, `vein_points.npz`, `is_fixed_layer.npz`, `grid_node_v0_mask.npz`, `initial_vertex_positions_undeformed.pkl`, `vitactip_points_E.pkl`. Regenerable (Gmsh + marker detection) but tiny, fixed, and required before the simulator will start. |
| `difftactile/output/marker_tracker/domain-adaptation-vascular-markers/` | ~1.3 MB | System-identification marker tracks (`traj_0..3_out.pkl`), loaded at simulator start-up to align the contact model against the real sensor. |

**Total: roughly 310 MB uncompressed / ~242 MB as a `.tar.gz`**, dominated by the
simulated dataset and the meat camera frames.

## What is NOT in the bundle (and why)

| Excluded | Size in the raw archive | Reason |
|---|---:|---|
| `meat_training_data/raw/` | 1.6 GB | Full-rate `.avi` recordings (429 frames each) + pose `.npz`. Only needed to re-run preprocessing from scratch; `clean/` is the product, and the 26 frames per trial that preprocessing actually keeps now ship as `clean/*/frames.mp4` at 3% of the size. |
| `meat_training_data/intermediate/` | 250 MB | Intermediate preprocessing stages; reproducible from `raw/`. |
| `meat_training_data/clean/*/marker_labels.avi` | 81 MB | Pre-rendered label overlays. Superseded by `frames.mp4` + `marker_labels.npz`, which the viewer composites on the fly for a fraction of the size. |
| `silicone_training_data/20250901-131547{,_interpolated_trimmed,_markers,_dilated,...}` | ~1.05 GB | Intermediate silicone preprocessing stages; only the final `_dense` directory is consumed. |
| `system-id-screws-*.mkv`, `vein_slide_across.avi`, `domain-adaptation-vascular-videos/` | ~960 MB | Videos used for system identification and exploratory analysis, not for the published results. |
| `difftactile/output/*` (meshes, voronoi, marker_tracker, debug images) | ~60 MB | All regenerated by `script_pre_main` / `script_generate_*_mesh_gmsh` / `script_marker_tracker`. |
| `difftactile/output/exp_probs.npz` | 424 KB | Cached per-marker probabilities for the bird's-eye vessel map. Pure model output: `./docker/vessel_map.sh` recomputes it from the shipped silicone dataset + meat checkpoint in **under a second**, bit-identically. Only `--cached` reuses it, and that flag is an optimisation, not a requirement. |
| `difftactile/output/phantom_ground_truth_segmentation_mask.jpg` and the other `*ground_truth*` images | ~150 KB | Regenerated bit-identically by `script_annotate` from two **git-tracked** inputs (`phantom-labels-vgg.json`, `phantom-uncropped-compressed-undistorted.jpg`), so they are in the repository already, not missing. The remaining `ground_truth_*` images are written by `predict_exp` itself during the same run. |
| `lightning_logs/`, `logs/` | 68 MB | Training logs from the original runs; recreated on any new training run. |

Excluding these takes the bundle from **4.5 GB to roughly 258 MB (~190 MB compressed)** without
affecting the reproducibility of any published result.

## Restoring

From the repository root:

```bash
./data/restore_data.sh path/to/shallow-vessel-palpation-data.tar.gz
```

The script unpacks into the correct locations and verifies each expected path
afterwards. See `data/restore_data.sh --help` for options.
