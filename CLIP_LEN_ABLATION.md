# Clip-length ablation

How many video frames should the GNN see at once? The model takes a
`clip_len`-frame window of marker positions and predicts the central frame;
this trains the A-to-B configuration (train on simulation, test on real
silicone) at each window length and scores it on the silicone test set.

Each row is 3 training run(s) (seeds 0, 1, 2),
reported as mean ± std - single runs on this project reflect their seed as much
as the model (see the seed sweep in AUROC_RESULTS.md).

**The deciding metric is foreground IoU** (the vessel-present class, at the
standard decision threshold); AUROC and AP are context.

| clip_len | Foreground IoU mean ± std | Foreground IoU range | AUROC mean ± std | AP mean ± std |
|---|---|---|---|---|
| 1 | **0.1142 ± 0.0018** | 0.1123–0.1157 | 0.5815 ± 0.0030 | 0.2094 ± 0.0054 |
| 3 | **0.1967 ± 0.0059** | 0.1900–0.2011 | 0.7403 ± 0.0024 | 0.2008 ± 0.0012 |
| 5 **(best)** | **0.2384 ± 0.0012** | 0.2371–0.2394 | 0.7818 ± 0.0018 | 0.3243 ± 0.0011 |
| 7 | **0.2139 ± 0.0294** | 0.1799–0.2310 | 0.7676 ± 0.0078 | 0.3105 ± 0.0061 |

Checkpoints and per-run artifacts: `saved_models_ablation/20260815-005709/clip_len_XX/`;
`ablation.json` there repeats these numbers in machine-readable form.
