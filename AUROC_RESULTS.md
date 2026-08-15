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
| A-to-A | 5 | **0.9557 ± 0.0024** | 0.9518–0.9573 | **0.5133 ± 0.0110** | 0.4983–0.5261 | 0.0472 |
| A-to-B | 5 | **0.7667 ± 0.0060** | 0.7587–0.7734 | **0.3107 ± 0.0069** | 0.3033–0.3186 | 0.1143 |
| C-to-B | 5 | **0.7049 ± 0.0469** | 0.6550–0.7787 | **0.3002 ± 0.0309** | 0.2676–0.3418 | 0.1143 |
| A-to-C | 5 | **0.8237 ± 0.0012** | 0.8222–0.8256 | **0.2274 ± 0.0023** | 0.2252–0.2309 | 0.0716 |

### Intersection over union

Over the frame-by-frame marker predictions, at the decision threshold — **this is the
table to quote in the manuscript**, with the ± rather than a single seed's value.
*Foreground* is the vessel-present class and *background* the vessel-absent one; each
is the agreement between prediction and ground truth over that class, so neither is
"what the ground truth says" alone. Background IoU is always the flattering number,
since the negative class dominates — read the foreground column as the real result.

| Config | Seeds | Foreground IoU mean ± std | Foreground range | Background IoU mean ± std | Background range |
|---|---|---|---|---|---|
| A-to-A | 5 | **0.2133 ± 0.0381** | 0.1686–0.2618 | 0.8183 ± 0.0486 | 0.7562–0.8771 |
| A-to-B | 5 | **0.2159 ± 0.0243** | 0.1799–0.2362 | 0.8246 ± 0.0422 | 0.7711–0.8704 |
| C-to-B | 5 | **0.1506 ± 0.0102** | 0.1386–0.1638 | 0.4712 ± 0.1372 | 0.2908–0.6307 |
| A-to-C | 5 | **0.1783 ± 0.0119** | 0.1631–0.1925 | 0.7202 ± 0.0391 | 0.6709–0.7662 |

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
| A-to-B | 5 | **0.9555 ± 0.0028** | **0.5121 ± 0.0110** | **0.2135 ± 0.0386** | 0.8184 ± 0.0485 |
| A-to-C | 5 | **0.9555 ± 0.0028** | **0.5121 ± 0.0110** | **0.2135 ± 0.0386** | 0.8184 ± 0.0485 |

Weights for every seed above are preserved under `saved_models_sweeps/20260815-004056/`, one subdirectory per
(configuration, seed), each holding the checkpoint **and** the test-loader pickle
carrying the normalisation statistics it was trained with. `sweep.json` in that
directory repeats these metrics in machine-readable form.

### Per-seed values

Shown in full so the summary above can be checked, and so an outlier is visible
rather than averaged away.

**A-to-A**

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.9570 | 0.5219 | 0.2618 | 0.8771 | 44.0 | `A-to-A_seed00/` |
| 1 | 0.9550 | 0.5102 | 0.1686 | 0.7562 | 43.8 | `A-to-A_seed01/` |
| 2 | 0.9518 | 0.4983 | 0.2052 | 0.8153 | 43.8 | `A-to-A_seed02/` |
| 3 | 0.9573 | 0.5261 | 0.2416 | 0.8539 | 44.0 | `A-to-A_seed03/` |
| 4 | 0.9573 | 0.5101 | 0.1890 | 0.7889 | 43.8 | `A-to-A_seed04/` |

**A-to-B**

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.7587 | 0.3038 | 0.1799 | 0.8704 | 47.7 | `A-to-B_seed00/` |
| 1 | 0.7706 | 0.3120 | 0.2310 | 0.7711 | 47.4 | `A-to-B_seed01/` |
| 2 | 0.7734 | 0.3157 | 0.2308 | 0.8286 | 47.7 | `A-to-B_seed02/` |
| 3 | 0.7627 | 0.3033 | 0.2018 | 0.8591 | 47.5 | `A-to-B_seed03/` |
| 4 | 0.7680 | 0.3186 | 0.2362 | 0.7939 | 47.4 | `A-to-B_seed04/` |

**C-to-B**

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.6550 | 0.2676 | 0.1386 | 0.4059 | 49.4 | `C-to-B_seed00/` |
| 1 | 0.6964 | 0.2888 | 0.1638 | 0.5825 | 49.0 | `C-to-B_seed01/` |
| 2 | 0.6787 | 0.2800 | 0.1583 | 0.6307 | 49.3 | `C-to-B_seed02/` |
| 3 | 0.7159 | 0.3225 | 0.1472 | 0.4459 | 48.7 | `C-to-B_seed03/` |
| 4 | 0.7787 | 0.3418 | 0.1452 | 0.2908 | 49.0 | `C-to-B_seed04/` |

**A-to-C**

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.8256 | 0.2309 | 0.1925 | 0.7662 | 49.4 | `A-to-C_seed00/` |
| 1 | 0.8222 | 0.2263 | 0.1631 | 0.6709 | 49.4 | `A-to-C_seed01/` |
| 2 | 0.8234 | 0.2252 | 0.1782 | 0.7189 | 49.3 | `A-to-C_seed02/` |
| 3 | 0.8233 | 0.2262 | 0.1869 | 0.7502 | 49.4 | `A-to-C_seed03/` |
| 4 | 0.8238 | 0.2285 | 0.1707 | 0.6946 | 49.5 | `A-to-C_seed04/` |

> Each sweep writes a **new timestamped directory**, so sweeping repeatedly
> accumulates rather than overwriting. Nothing prunes them - delete them when done.
> The published checkpoints and the ordinary `*_retrained_<config>` artifacts are
> untouched by a sweep.
