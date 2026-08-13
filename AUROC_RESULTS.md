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
| C-to-B | 3 | **0.6767 ± 0.0208** | 0.6550–0.6964 | **0.2788 ± 0.0106** | 0.2676–0.2888 | 0.1143 |

### Intersection over union

Over the frame-by-frame marker predictions, at the decision threshold — **this is the
table to quote in the manuscript**, with the ± rather than a single seed's value.
*Foreground* is the vessel-present class and *background* the vessel-absent one; each
is the agreement between prediction and ground truth over that class, so neither is
"what the ground truth says" alone. Background IoU is always the flattering number,
since the negative class dominates — read the foreground column as the real result.

| Config | Seeds | Foreground IoU mean ± std | Foreground range | Background IoU mean ± std | Background range |
|---|---|---|---|---|---|
| C-to-B | 3 | **0.1536 ± 0.0133** | 0.1386–0.1638 | 0.5397 ± 0.1184 | 0.4059–0.6307 |

Weights for every seed above are preserved under `saved_models_sweeps/20260813-173417/`, one subdirectory per
(configuration, seed), each holding the checkpoint **and** the test-loader pickle
carrying the normalisation statistics it was trained with. `sweep.json` in that
directory repeats these metrics in machine-readable form.

### Per-seed values

Shown in full so the summary above can be checked, and so an outlier is visible
rather than averaged away.

**C-to-B**

| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |
|---|---|---|---|---|---|---|
| 0 | 0.6550 | 0.2676 | 0.1386 | 0.4059 | 49.4 | `C-to-B_seed00/` |
| 1 | 0.6964 | 0.2888 | 0.1638 | 0.5825 | 49.3 | `C-to-B_seed01/` |
| 2 | 0.6787 | 0.2800 | 0.1583 | 0.6307 | 49.5 | `C-to-B_seed02/` |

> Each sweep writes a **new timestamped directory**, so sweeping repeatedly
> accumulates rather than overwriting. Nothing prunes them - delete them when done.
> The published checkpoints and the ordinary `*_retrained_<config>` artifacts are
> untouched by a sweep.
