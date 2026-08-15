# Clean-slate reproduction log (2026-08-15)

A fresh clone into an empty directory (`/tmp/vessel-clean-slate/`), the data bundle
downloaded from the published Zenodo record, and every entrypoint run in a **second,
freshly started container** (`VESSEL_PALPATION_CONTAINER=vessel-palpation-clean`) —
nothing reused from the development checkout. Repository state: commit `2e01dab`.

| Step | Result |
|---|---|
| `git clone` (SSH URL; the https URL needs the repository to be public) | OK |
| DOI `10.5281/zenodo.21900934` → `https://zenodo.org/records/21900934`; `wget .../files/shallow-vessel-palpation-data.tar.gz` | 288 023 322 bytes, sha256 `ca540cb3…1a34fe` (matches the local build) |
| `./data/restore_data.sh <tarball>` | all expected paths present |
| `docker-build.sh` (cached) → `docker-run.sh` → `run_pipeline.sh check` | GPU, torch, taichi, torch_geometric, config, data: OK |
| `run_pipeline.sh sim-short` | 8 trials collected, "all done"; exit 139 (known GGUI teardown segfault after writing) |
| `run_pipeline.sh A-to-A / A-to-B / A-to-C` (published best-of-five checkpoints) | AUROC 0.9585 / 0.7832 / 0.8405, AP 0.492 / 0.326 / 0.228, FG IoU 0.185 / 0.239 / 0.167 |
| `run_pipeline.sh C-to-B` (train, seed 42, 48 s) | AUROC 0.6755, AP 0.281, FG IoU 0.145 (a single seed — inside the published five-seed spread) |
| `score_all_scenarios.sh --pretrained` | table + 8 ROC/PR PDFs |
| `score_all_scenarios.sh --seeds 2` | 4 configurations × 2 seeds trained and summarised (354 s) |
| `ablation_clip_len.sh --seeds 1` | clip lengths {1,3,5,7} trained and ranked (152 s) |
| `vessel_map_all.sh` | first run failed on the Silicone maps: the photo ground-truth mask is not in the bundle → **fixed** (built on first use, `2e01dab`); re-run: all six maps, thresholds identical to the published runs (0.6283 / 0.5739 / 0.5990 / 0.6129 / 0.6183; Sim→Meat F1 fallback 0.5530), video-vs-photo truth IoU 0.2934 |
| `vessel_map_sim_trajectory.sh` | posed slide simulated and reordered (77 s; exit 139 after writing) |
| `alignment_figures.sh` | MAE 11.87 / 15.82 / 14.53 / 12.42 px (press / twist-z / twist-x / slide) |
| `script_frame_space_metrics` | `FRAME_SPACE_METRICS.md` identical to the published one |
| `domain_adaptation.sh` (joint BO, `DIFFTACTILE_BO_ITERATIONS=2 DIFFTACTILE_BO_RANDOM=1`) | ran to completion (493 s), wrote results + validation on the four interactions; exit 139 after writing |
| `score_params.sh` | vpn 0.905, van 0.075, objective +0.830 |
| `record_da_trajectories.sh` | four .mp4 recorded (exit 139 after writing) |
| `view_predictions.sh A-to-C --record`, `annotate_data_bare_metal.sh --meat --record`, `record_videos.sh --predictions` | recorded offscreen, compressed into `videos/` |

Notes: every exit code 139 is the documented Taichi GGUI teardown segfault that happens
*after* the run has printed its completion message and written its outputs. Running the
sweep, the ablation and DA in a clone rewrites `AUROC_RESULTS.md`, `CLIP_LEN_ABLATION.md`
and `bo-gp.json` (the BO hand-off file) in that clone, as designed.
