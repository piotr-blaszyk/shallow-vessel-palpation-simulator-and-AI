# Blank-slate reproduction test

Record of an end-to-end verification run: a fresh clone into an empty directory,
data restored from the bundle, and the full Docker workflow exercised — the
sequence an external reader would follow.

**Date:** 2026-08-11 · **Host:** Ubuntu 24.04, NVIDIA RTX 3080 (10 GB), Docker 29.6.2

Run twice: once on the initial Docker/unification work, and again on a second
clean clone after an adversarial code review produced fixes (checkpoint
protection, cwd-independent paths). The numbers below are from the second run.

## Procedure

> This is the transcript of the run as it happened, kept verbatim. It cloned
> `iros` because that was the submission branch at the time. **Use `main`
> today** — it is the only supported branch, and `git clone <repo>` already
> lands there. The scenario names below are likewise the pre-paper ones, still
> accepted as aliases.

```bash
git clone --branch iros <repo> shallow-vessel-palpation-simulator-and-AI && cd shallow-vessel-palpation-simulator-and-AI
./data/restore_data.sh --verify                       # expect: 23 paths MISSING
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz        # ~190 MB bundle
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
| `all-scenarios` | OK. All three run in sequence, and the published checkpoint is **byte-identical** afterwards (training writes `*_retrained`). |
| Run from another cwd | OK. Every scenario and the simulator run with `-w /tmp`. |
| **`sim-full` (overnight)** | OK. 100 loops / **800 trials in 9905 s (2 h 45 m)**, exit 0, no errors, 0 corrupt files. Exactly 200 of each trajectory type; type 3 gives 316-317 frames, matching the published dataset. 132 MB. |
| **`DIFFTACTILE_TRAJECTORIES=3`** | OK. Produces only type-3 trials at 317 frames — the published dataset's format, now regenerable. |

### Reproducibility against the original checkout

| Metric | Original | Fresh clone |
|---|---|---|
| sim-to-silicone AUC | 0.6785678641614646 | 0.6785678641614648 |
| silicone-to-meat IoU (bg / vein) | 0.8092 / 0.1983 | 0.8091 / 0.1975 |

Agreement to ~15 significant figures on AUC. The evaluation scenarios are
deterministic; `sim-to-meat` trains from a random initialisation and so varies
between runs by design.

> The scenario names above are the pre-paper ones, kept here as the transcript of
> that run. They still work as aliases: `sim-to-silicone` → `A-to-B`,
> `sim-to-meat` → `C-to-B`, `silicone-to-meat` → `A-to-C`.

### All three paper configurations, train and evaluate (2026-08-12)

Added `--train` / `--eval` for each of the paper's three (train → test)
configurations, so no branch switching is needed to train a different model.
Verified on the development machine (RTX 3080):

| Check | Outcome |
|---|---|
| Dispatcher routing | OK. 15/15 cases: paper names, both modes, defaults, legacy aliases, env vars, and rejection of unknown configs/flags. |
| `A-to-B --train` | OK. New sim-training path runs end to end, tests on silicone, writes `*_retrained` artifacts. |
| `A-to-C --train` | OK. Same, evaluating against the meat trials. |
| `A-to-B --eval` | OK. **AUC 0.6785678641614646** — matches the original checkout exactly. |
| `A-to-C --eval` | OK. **IoU 0.809 (bg) / 0.198 (vein)** — matches the table above. |
| Published checkpoints | Untouched by training runs (`_retrained` suffix). |

Training runs above used a deliberately tiny config (1 epoch, 2 batches) purely
to exercise the code path; they are smoke tests, not published results.

Two pre-existing bugs were fixed to make this possible:

1. **Simulation training was disabled on every branch.** `cnn/gnn.py::main()`
   began with a bare `return`, so neither sim-trained model (A→B, A→C) could be
   reproduced. Reimplemented as `iros_gnn.train_on_sim()`.
2. **`evaluate_and_plot_roc()` crashed headless.** `_show_plots()` guarded
   `plt.show()`, but `plt.figure()` had already tried to open a Tk window, so
   `A-to-B --eval` raised `TclError` under `DIFFTACTILE_HEADLESS=1` (confirmed
   on unmodified `main`). `iros_gnn.py` now selects the `Agg` backend before
   importing pyplot when no display is present.

### Blank-slate re-run after the dead-code removal (2026-08-12)

Repeated the whole quickstart in a fresh clone of `main` at `5ecb7e5`, which deletes
DEAD_CODE_ANALYSIS categories A–C, to confirm nothing live was removed. Bundle restored
from a local copy rather than Zenodo (`restore_data.sh` accepts any path).

| Stage | Outcome |
|---|---|
| Fresh clone | OK. 23 paths reported MISSING; working tree 15 MB (was ~27 MB — the `docs/` GIFs). |
| Restore bundle | OK. All 23 present. |
| Docker build | OK. |
| `check` | OK. CUDA, torch 2.8.0+cu126, Taichi 1.7.4, PyG 2.8.0.post1, config loads, data verified. |
| **`A-to-B`** | OK. **AUC 0.6785678641614648** — bit-identical to the runs above. |
| **`A-to-C`** | OK. **IoU 0.8083 (bg) / 0.1968 (vein)** — matches 0.809 / 0.197. |
| **`C-to-B`** | OK. Trains, tests on silicone, writes `*_retrained`; published checkpoint untouched. |
| `sim-short` | OK. 8 trials, frame counts 0/73/17/317 per trajectory type, 0 corrupt. |
| Syntax | OK. All 70 remaining tracked `.py` files byte-compile. |

#### Pre-existing segfault surfaced by this run

`sim-short` exits **139 (SIGSEGV)** when a display is available. It is *not* caused by the
dead-code removal — a 2×2 over {commit before removal, after} × {GUI, headless} pins it to
the GUI alone:

| Commit | GUI | Exit | Runtime |
|---|---|---|---|
| `6ef899c` (before removal) | on | **139** | 148.3 s |
| `6ef899c` (before removal) | off | 0 | 108.8 s |
| `5ecb7e5` (after removal) | on | **139** | 149.7 s |
| `5ecb7e5` (after removal) | off | 0 | 107.4 s |

The fault is in CUDA/GGUI teardown *after* `main()` prints `all done`; the 8 trajectory files
are written and valid in every case, including the crashing ones. `run_pipeline.sh` forces
headless only when `DISPLAY` is unset, and `docker-run.sh` passes `DISPLAY` through, so the
containerised `sim-short` hits it by default. Documented in the README's known issues;
`DIFFTACTILE_HEADLESS=1` avoids it and is ~35% faster.

## ⚠️ One deliberate behaviour change — please review

`silicone-to-meat` unifies what used to live on the `sim-to-meat-test` branch,
but it is **not a byte-faithful port**, and the difference matters.

On that branch, `test_dataset.set_stats(stats)` sat inside a dead `if False:`
block. `set_stats()` is what clears `MyDataset.warmup`, and `normalise()` is
gated on `if not self.warmup` — so the ICRA checkpoint was being evaluated on
**unnormalised inputs**, despite having been trained on normalised ones. The
unified code calls `set_stats()`, applying the statistics the checkpoint expects.

Measured effect on the cross-domain result:

| | vein IoU | background IoU |
|---|---|---|
| Old path (unnormalised, `if False:`) | **0.034** | 0.888 |
| Unified code (normalised) | **0.198** | 0.809 |

The old path understated the cross-domain vein IoU roughly **six-fold**. The
normalised number is the defensible one — evaluating a model on a different
input distribution than it was trained on is a bug, not a design choice — but
**if 0.034 (or anything derived from it) appears in the paper, that figure needs
revisiting.** Flagging rather than silently changing the reported result.

## Notes for anyone repeating this

- **Regenerating the published dataset needs `DIFFTACTILE_TRAJECTORIES=3`.**
  All 500 trajectories in the shipped dataset are type 3 ("slide (vein)"), each
  316-317 frames. Git history shows it was collected when the collection loop
  read `range(3, 4)`; commit `0e7280a` later widened it to `range(0, 4)`. So a
  default run today produces a *different* mix — all four types, of which only
  type 3 matches. The loop is now driven by `DIFFTACTILE_TRAJECTORIES`, defaulting
  to all four (current committed behaviour) with `3` reproducing the published set.
  **Verified end-to-end:** a run with `DIFFTACTILE_TRAJECTORIES=3` yields only
  type-3 trials at 317 frames, matching the published format.
- **Simulated data shape.** With all four types enabled, the `trajectory_type=0`
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
