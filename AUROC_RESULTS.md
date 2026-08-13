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

ROC curves: `difftactile/output/roc_curves/roc_curve_<config>_<weights>.pdf`
PR curves:  `difftactile/output/pr_curves/pr_curve_<config>_<weights>.pdf`

| Scenario | Train -> Test | Weights | AUROC | AP | Chance | Lift | Nodes | Positive |
|---|---|---|---|---|---|---|---|---|
| A-to-B [pretrained] | train on simulation (A), test on silicone (B) | pretrained | **0.7314** | **0.2553** | 0.1143 | 2.23x | 12700 | 1452 (11.4%) |

## Seed sweep

Each configuration trained from scratch once per seed, then scored. **This is the
number to report** - a single training run on this project reflects its seed as much
as the model, because the training sets are small enough that which subset a run
favours dominates the outcome.

Read the **spread**, not the best row. Selecting the highest-scoring seed would be
fitting to the test set exactly as tuning a decision threshold on it would be. When
comparing two configurations, compare their distributions - a gap smaller than the
seed spread is not evidence of anything.

| Config | Seeds | AUROC mean ± std | AUROC range | AP mean ± std | AP range | Chance |
|---|---|---|---|---|---|---|
| C-to-B | 2 | **0.6757 ± 0.0293** | 0.6550–0.6964 | **0.2782 ± 0.0150** | 0.2676–0.2888 | 0.1143 |

Weights for every seed above are preserved under `saved_models_sweeps/20260813-163829/`, one subdirectory per
(configuration, seed), each holding the checkpoint **and** the test-loader pickle
carrying the normalisation statistics it was trained with. `sweep.json` in that
directory repeats these metrics in machine-readable form.

### Per-seed values

Shown in full so the summary above can be checked, and so an outlier is visible
rather than averaged away.

**C-to-B**

| Seed | AUROC | AP | Seconds | Weights |
|---|---|---|---|---|
| 0 | 0.6550 | 0.2676 | 51.1 | `C-to-B_seed00/` |
| 1 | 0.6964 | 0.2888 | 50.5 | `C-to-B_seed01/` |

> Each sweep writes a **new timestamped directory**, so sweeping repeatedly
> accumulates rather than overwriting. Nothing prunes them - delete them when done.
> The published checkpoints and the ordinary `*_retrained_<config>` artifacts are
> untouched by a sweep.
