# Ranking metrics across the six canonical scenarios

Per-node vessel-classification AUROC and average precision (AP) for each
(train -> test) configuration, scored twice: once from the published Zenodo
checkpoint and once from a checkpoint retrained locally with `--train`.

Datasets: **A** = simulation, **B** = real silicone phantom, **C** = real meat phantom.

Both metrics are **threshold-free and ranking-based**: they read only the
ordering of the predicted probabilities, never their absolute scale, so no
decision threshold is chosen anywhere in this table. That is deliberate - in a
sim-to-real setting the output *scale* is the first thing to shift between
domains, and a single-threshold score would confound that with what the model
actually learned.

They are reported together because each is blind to something the other sees:

- **AUROC** normalises false positives by the total negative count. On this
  heavily imbalanced problem a large absolute number of false alarms barely
  moves it. Its baseline is always 0.5, which makes it comparable across papers.
- **AP** ignores true negatives entirely, so the negative majority cannot
  flatter it. Its baseline is the **positive rate**, given in the `Chance`
  column, so AP must be read against that - hence the `Lift` column
  (AP / chance), which is how many times better than random ranking the model is.

### Intersection over union

Also reported, at decision threshold **0.5**, over the same
frame-by-frame marker predictions (**not** the reprojected bird's-eye phantom map,
whose separate IoU comes from `vessel_map.sh`):

- **Foreground IoU** — the *vessel present* class. Note this is the agreement between
  prediction and ground truth over that class, `|pred AND true| / |pred OR true|`, not
  "what the ground truth says" or "what the prediction says" on its own.
- **Background IoU** — the same for *vessel absent*. Always the flattering number: the
  negative class is ~89% of nodes on silicone and ~93% on meat, so even a poor model
  overlaps with it heavily. The foreground number is the informative one.

Unlike AUROC and AP, IoU **depends on the decision threshold**, so it carries every
caveat about that choice. It is reported because it is the intuitive quantity, not
because it is the most trustworthy one.

ROC curves: `difftactile/output/roc_curves/roc_curve_<config>_<weights>.pdf`
PR curves:  `difftactile/output/pr_curves/pr_curve_<config>_<weights>.pdf`

| Scenario | Train -> Test | Weights | AUROC | AP | IoU fg | IoU bg | Chance | Lift | Nodes | Positive |
|---|---|---|---|---|---|---|---|---|---|---|
| A-to-A [pretrained] | train on simulation (A), test on held-out simulation (A) | pretrained | **0.9371** | **0.6179** | **0.2644** | 0.7645 | 0.0828 | 7.46x | 66675 | 5524 (8.3%) |
| A-to-B [pretrained] | train on simulation (A), test on silicone (B) | pretrained | **0.7314** | **0.2553** | **0.2059** | 0.7562 | 0.1143 | 2.23x | 12700 | 1452 (11.4%) |
| C-to-B [pretrained] | train on meat (C), test on silicone (B) | pretrained | **0.6786** | **0.2871** | **0.1531** | 0.6066 | 0.1143 | 2.51x | 12700 | 1452 (11.4%) |
| A-to-C [pretrained] | train on simulation (A), test on meat (C) | pretrained | **0.8171** | **0.2208** | **0.1982** | 0.8119 | 0.0716 | 3.09x | 25273 | 1809 (7.2%) |

## Seed sweep

Each configuration trained from scratch once per seed, then scored. **This is the
number to report** - a single training run on this project reflects its seed as much
as the model, because the training sets are small enough that which subset a run
favours dominates the outcome.

Read the **spread**, not the best row. Selecting the highest-scoring seed would be
fitting to the test set exactly as tuning a decision threshold on it would be. When
comparing two configurations, compare their distributions - a gap smaller than the
seed spread is not evidence of anything.

### Threshold-free metrics

| Config | Seeds | AUROC mean ± std | AUROC range | AP mean ± std | AP range | Chance |
|---|---|---|---|---|---|---|
| A-to-A | 5 | **0.9581 ± 0.0007** | 0.9573–0.9592 | **0.4932 ± 0.0048** | 0.4873–0.4999 | 0.0421 |
| A-to-B | 5 | **0.7789 ± 0.0043** | 0.7732–0.7832 | **0.3213 ± 0.0043** | 0.3156–0.3255 | 0.1133 |
| C-to-B | 5 | **0.7165 ± 0.0543** | 0.6472–0.7874 | **0.3035 ± 0.0377** | 0.2602–0.3481 | 0.1133 |
| A-to-C | 5 | **0.8371 ± 0.0024** | 0.8338–0.8405 | **0.2243 ± 0.0036** | 0.2184–0.2279 | 0.0651 |

### Intersection over union

Over the frame-by-frame marker predictions, at the decision threshold — **this is the
table to quote in the manuscript**, with the ± rather than a single seed's value.
*Foreground* is the vessel-present class and *background* the vessel-absent one; each
is the agreement between prediction and ground truth over that class, so neither is
"what the ground truth says" alone. Background IoU is always the flattering number,
since the negative class dominates — read the foreground column as the real result.

| Config | Seeds | Foreground IoU mean ± std | Foreground range | Background IoU mean ± std | Background range |
|---|---|---|---|---|---|
| A-to-A | 5 | **0.1930 ± 0.0181** | 0.1794–0.2246 | 0.8182 ± 0.0224 | 0.8006–0.8573 |
| A-to-B | 5 | **0.2342 ± 0.0073** | 0.2218–0.2394 | 0.8113 ± 0.0265 | 0.7858–0.8554 |
| C-to-B | 5 | **0.1550 ± 0.0121** | 0.1341–0.1648 | 0.5367 ± 0.1236 | 0.3713–0.7099 |
| A-to-C | 5 | **0.1707 ± 0.0058** | 0.1658–0.1804 | 0.7304 ± 0.0191 | 0.7159–0.7628 |

### In-domain reference (simulation → simulation)

The same models scored on the **held-out 15% of the simulated dataset** — same
distribution as their training data, never seen during training. The gap between
this and the cross-domain table above **is the sim-to-real transfer cost**, which
is the quantity the project exists to measure, so the pair is worth reporting
together.

This is the simulated **test** split, not the validation split: validation drives
early stopping and checkpoint selection, so a number read off it is optimistic by
construction. Only the simulation-trained configurations (A→B, A→C) appear here —
C→B trains on meat and has no same-distribution split to report.

| Config | Seeds | AUROC mean ± std | AP mean ± std | Foreground IoU mean ± std | Background IoU mean ± std |
|---|---|---|---|---|---|
| A-to-B | 5 | **0.9583 ± 0.0007** | **0.4937 ± 0.0047** | **0.1929 ± 0.0179** | 0.8181 ± 0.0224 |
| A-to-C | 5 | **0.9583 ± 0.0007** | **0.4937 ± 0.0047** | **0.1929 ± 0.0179** | 0.8181 ± 0.0224 |

Weights for every seed above are preserved under `saved_models_sweeps/20260815-130143/`, one subdirectory per
(configuration, seed), each holding the checkpoint **and** the test-loader pickle
carrying the normalisation statistics it was trained with. `sweep.json` in that
directory repeats these metrics in machine-readable form.

### Per-seed values

Shown in full so the summary above can be checked, and so an outlier is visible
rather than averaged away.

**A-to-A** (best-of-5 by AP: seed 1, used wherever a single model instance is shown)

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.9584 | 0.4932 | 0.1883 | 0.8127 | 37.5 | `A-to-A_seed00/` |
| 1 | 0.9592 | 0.4999 | 0.1888 | 0.8132 | 37.5 | `A-to-A_seed01/` |
| 2 | 0.9580 | 0.4905 | 0.1841 | 0.8072 | 37.5 | `A-to-A_seed02/` |
| 3 | 0.9573 | 0.4873 | 0.1794 | 0.8006 | 37.4 | `A-to-A_seed03/` |
| 4 | 0.9577 | 0.4953 | 0.2246 | 0.8573 | 37.5 | `A-to-A_seed04/` |

**A-to-B** (best-of-5 by AP: seed 2, used wherever a single model instance is shown)

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.7797 | 0.3241 | 0.2371 | 0.8123 | 40.8 | `A-to-B_seed00/` |
| 1 | 0.7824 | 0.3233 | 0.2387 | 0.8046 | 41.0 | `A-to-B_seed01/` |
| 2 | 0.7832 | 0.3255 | 0.2394 | 0.7985 | 41.0 | `A-to-B_seed02/` |
| 3 | 0.7732 | 0.3156 | 0.2342 | 0.7858 | 40.7 | `A-to-B_seed03/` |
| 4 | 0.7758 | 0.3177 | 0.2218 | 0.8554 | 40.8 | `A-to-B_seed04/` |

**C-to-B** (best-of-5 by AP: seed 4, used wherever a single model instance is shown)

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.6472 | 0.2602 | 0.1341 | 0.5774 | 48.2 | `C-to-B_seed00/` |
| 1 | 0.6801 | 0.2732 | 0.1604 | 0.7099 | 47.8 | `C-to-B_seed01/` |
| 2 | 0.7292 | 0.3028 | 0.1648 | 0.5335 | 48.4 | `C-to-B_seed02/` |
| 3 | 0.7387 | 0.3333 | 0.1586 | 0.4913 | 48.1 | `C-to-B_seed03/` |
| 4 | 0.7874 | 0.3481 | 0.1572 | 0.3713 | 48.0 | `C-to-B_seed04/` |

**A-to-C** (best-of-5 by AP: seed 2, used wherever a single model instance is shown)

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.8367 | 0.2250 | 0.1706 | 0.7297 | 42.7 | `A-to-C_seed00/` |
| 1 | 0.8374 | 0.2258 | 0.1697 | 0.7267 | 42.8 | `A-to-C_seed01/` |
| 2 | 0.8405 | 0.2279 | 0.1669 | 0.7169 | 42.4 | `A-to-C_seed02/` |
| 3 | 0.8338 | 0.2184 | 0.1658 | 0.7159 | 42.6 | `A-to-C_seed03/` |
| 4 | 0.8370 | 0.2242 | 0.1804 | 0.7628 | 42.4 | `A-to-C_seed04/` |

> Each sweep writes a **new timestamped directory**, so sweeping repeatedly
> accumulates rather than overwriting. Nothing prunes them - delete them when done.
> The published checkpoints and the ordinary `*_retrained_<config>` artifacts are
> untouched by a sweep.
