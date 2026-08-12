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

The paths in the tables below are the **current** ones. The archive presently
attached to the Zenodo record was built before the naming cleanup and still uses
the older directory names (`endgame/`, `iros_training_data/`, `saved_models_iros/`,
`saved_models_icra/`, and the `*_iros` / `*_icra` file suffixes).

That mismatch is handled automatically: `restore_data.sh` looks for each expected
path under its current name and falls back to the pre-rename name, restoring it
under the **new** name either way. A fresh clone therefore reproduces the published
results from the existing Zenodo archive with no manual steps. `make_data_bundle.sh`
applies the same fallback on the source side, so a rebuilt bundle contains only the
new names. The fallback tables in both scripts can be dropped once a re-built bundle
replaces the published one.

Old names also survive *inside* two kinds of binary artifact, which a path rename
alone cannot reach. `data/migrate_bundle_artifacts.py` rewrites both, and
`make_data_bundle.sh` runs it over the staged bundle so a rebuild is clean:

- **Test-loader pickles** carry an `'iros'` flag key and a pickled `MyDataset`
  whose attributes are `iros_data` / `dilation_iros` with `scheme='iros'`.
  `pickle.load` restores `__dict__` without calling `__init__`, so an un-migrated
  loader's `scheme` no longer matches the renamed `"meat"` branch and
  `len(dataset)` silently returns 0. The three paper configurations are unaffected
  (they read only `dataset_stats`), but `cnn/visualise.py` uses the dataset object.
- **`.pt` checkpoints** are zip archives whose internal entries are named after the
  file's name at save time, so the meat checkpoint embedded
  `final_segmentation_model_gnn_iros/`. Re-saving clears it; the tensors are
  bit-for-bit identical, though the round-trip can shift a reported metric by
  ~1 ULP (AUC `…4648` → `…4646`).

## What IS in the bundle

| Path (after restore) | Size | Why it cannot be regenerated |
|---|---:|---|
| `difftactile/output/training_data/pickle_20250901_220921_reordered_dense/` | ~245 MB | 500 simulated trajectories. Generable via `script_main` but takes ~2 h 45 m GPU; shipped for convenience. |
| `difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense/` | ~684 KB | Real **silicone** phantom trials, fully preprocessed. Physical recording. |
| `difftactile/manual_or_experimental_data/meat_training_data/clean/` | ~900 KB | Real **meat** phantom trials, fully preprocessed (23 trials × 2 `.npz`). Physical recording. |
| `saved_models_sim/final_segmentation_model_gnn_sim.pt` | ~4.2 MB | Trained silicone GNN checkpoint. |
| `saved_models_meat/final_segmentation_model_gnn_meat.pt` | ~300 KB | Trained meat GNN checkpoint. |
| `difftactile/output/test_loader_gnn_sim.pickle` | ~60 KB | Normalisation statistics that the simulation-trained checkpoint was trained with; needed to evaluate it correctly. |
| `difftactile/output/test_loader_gnn_meat.pickle` | ~4 KB | As above, for the meat-trained checkpoint. |
| `difftactile/output/` sensor-geometry set (16 files) | ~5.5 MB | Sensor mesh, marker layout and graph connectivity: `base-graph-connectivity.npz`, `marker_locations_ordered.npz`, `init-marker-positions.{npz,pkl}`, `gmsh_mesh_{vitactip,vein}.pkl`, `vitactip_mesh.npz`, `edge_lengths.pkl`, `tactile_sensor.f2v.pkl`, `phantom_points.npz`, `vein_points.npz`, `is_fixed_layer.npz`, `grid_node_v0_mask.npz`, `initial_vertex_positions_undeformed.pkl`, `vitactip_points_E.pkl`. Regenerable (Gmsh + marker detection) but tiny, fixed, and required before the simulator will start. |
| `difftactile/output/marker_tracker/domain-adaptation-vascular-markers/` | ~1.3 MB | System-identification marker tracks (`traj_0..3_out.pkl`), loaded at simulator start-up to align the contact model against the real sensor. |

**Total: roughly 258 MB uncompressed / ~190 MB as a `.tar.gz`**, dominated by the
simulated dataset.

## What is NOT in the bundle (and why)

| Excluded | Size in the raw archive | Reason |
|---|---:|---|
| `meat_training_data/raw/` | 1.6 GB | Raw `.avi` recordings + `.npz`. Only needed to re-run preprocessing from scratch; `clean/` is the product. |
| `meat_training_data/intermediate/` | 250 MB | Intermediate preprocessing stages; reproducible from `raw/`. |
| `meat_training_data/clean/*/marker_labels.avi` | 81 MB | Visualisation overlays only. Training reads solely `marker_positions.npz` and `marker_labels.npz`. |
| `silicone_training_data/20250901-131547{,_interpolated_trimmed,_markers,_dilated,...}` | ~1.05 GB | Intermediate silicone preprocessing stages; only the final `_dense` directory is consumed. |
| `system-id-screws-*.mkv`, `vein_slide_across.avi`, `domain-adaptation-vascular-videos/` | ~960 MB | Videos used for system identification and exploratory analysis, not for the published results. |
| `difftactile/output/*` (meshes, voronoi, marker_tracker, debug images) | ~60 MB | All regenerated by `script_pre_main` / `script_generate_*_mesh_gmsh` / `script_marker_tracker`. |
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
