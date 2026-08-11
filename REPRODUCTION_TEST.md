# Blank-slate reproduction test

Record of an end-to-end verification run: a fresh clone into an empty directory,
data restored from the bundle, and the full Docker workflow exercised — the
sequence an external reader would follow.

**Date:** 2026-08-11 · **Host:** Ubuntu 24.04, NVIDIA RTX 3080 (10 GB), Docker 29.6.2

## Procedure

```bash
git clone --branch iros <repo> diff-tactile-fork && cd diff-tactile-fork
./data/restore_data.sh --verify                       # expect: 23 paths MISSING
./data/restore_data.sh difftactile-data.tar.gz        # ~190 MB bundle
./docker/docker-build.sh && ./docker/docker-run.sh
docker exec difftactile ./docker/run_pipeline.sh check
docker exec difftactile ./docker/run_pipeline.sh sim-to-silicone
docker exec difftactile ./docker/run_pipeline.sh silicone-to-meat
docker exec difftactile ./docker/run_pipeline.sh sim-to-meat
docker exec difftactile ./docker/run_pipeline.sh sim-short
```

## Results

| Stage | Outcome |
|---|---|
| Fresh clone | OK. `restore_data.sh --verify` correctly reported all **23** expected paths missing, with an actionable message. |
| Restore bundle | OK. All 23 paths present afterwards. |
| Docker build | OK, ~8 min (CUDA 12.6 base + torch 2.8.0+cu126 + PyG + Taichi). Image 8.79 GB. |
| `check` | OK. CUDA available, torch 2.8.0+cu126, Taichi 1.7.4, PyG 2.8.0, config loads, data verified. |
| **`sim-to-silicone`** | OK. **AUC 0.6785678641614648**, `roc_curve_iros.pdf` written. |
| **`silicone-to-meat`** | OK. 199 meat data points, **IoU 0.809 (bg) / 0.197 (vein)**. |
| **`sim-to-meat`** | OK. 30 epochs, tested on silicone, checkpoint saved. ~51 s. |
| `sim-short` | OK. Full 3-stage sim pipeline; 8 trials in 163 s. |

### Reproducibility against the original checkout

| Metric | Original | Fresh clone |
|---|---|---|
| sim-to-silicone AUC | 0.6785678641614646 | 0.6785678641614648 |
| silicone-to-meat IoU (bg / vein) | 0.8092 / 0.1983 | 0.8091 / 0.1975 |

Agreement to ~15 significant figures on AUC. The evaluation scenarios are
deterministic; `sim-to-meat` trains from a random initialisation and so varies
between runs by design.

## Notes for anyone repeating this

- **Simulated data shape.** In each 8-trial batch, the two `trajectory_type=0`
  files contain empty arrays. This is correct, not a failure: that trajectory is
  a short press that terminates in ~36 timesteps, below the `ts > 80` threshold at
  which recording begins. Types 1/2/3 yield ~73/17/317 frames, and the 317 matches
  the published dataset exactly.
- **Symlinks.** The author's own checkout wires several data paths to an external
  directory. Those resolve on the host but not inside the container;
  `restore_data.sh` replaces them with real files, which is why the containerised
  `check` may report paths "missing" on the author's machine until it is run.
- **File ownership.** The container runs as the invoking user, so artifacts written
  into the bind-mounted repository are owned by you, not root.
