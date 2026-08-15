# Legacy models (pre-2026-08-15)

The two checkpoints in this directory - `sim/` (the large simulation-trained
GNN, behind the Sim→Silicone and Sim→Meat models of the accepted manuscript)
and `meat/` (the compact meat-trained GNN, behind Meat→Silicone) - are the
**original** trained weights from the version of this project that was
accepted for publication. Each ships with the `test_loader_gnn_*.pickle`
carrying the normalisation statistics it was trained with; the two files of a
pair must always be used together.

They are kept for one reason: **they produced the accepted manuscript's
bird's-eye vessel map figure (`fig:vessel-map`, Fig. 8) and the localisation
table derived from it (`tab:localisation-map`, Table 4).** Panel (b) of that
figure was completed by hand (the yellow bins were added in a photo editor)
and Table 4 was read off that edited panel; redoing that manual step on new
models was not part of the 2026-08 update, so the figure and table stayed as
they were and the models that made them are preserved here so the artefact
remains traceable.

**They are not used for any of the main results.** Every other number and
figure in the manuscript - the IoU table, the ROC/PR curves, the seed sweep,
the clip-length ablation, the prediction viewer and the regenerated vessel
maps - comes from the current models: five seed instances per configuration
trained on the regenerated simulated dataset with a 5-frame temporal window,
under `saved_models_sweeps/<published sweep>/`, with the best-of-five instance
(by AP) at the published paths `saved_models_sim/` and `saved_models_meat/`.

## Why they cannot be retrained

They were trained on the *previous* simulated dataset
(`pickle_20250901_220921_reordered_dense`), which predates seeding of the
simulator and survives only in pre-2026-08-15 bundle archives, and with a
7-frame temporal window (`gnn.clip_len` was 7; it is 5 now). Their input layer
is sized to that window, so they only load with `DIFFTACTILE_CLIP_LEN=7`,
which the shell entrypoints set for `--model legacy`.

## How to use them

```bash
./docker/vessel_map.sh A-to-B --model legacy    # Sim→Silicone map with the legacy sim model
./docker/vessel_map.sh C-to-B --model legacy    # Meat→Silicone map with the legacy meat model
```

`cnn/model_selection.py::legacy_model()` is the only code path that loads
them, and it refuses any configuration other than A-to-B and C-to-B.
