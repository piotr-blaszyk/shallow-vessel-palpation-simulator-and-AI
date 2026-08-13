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
| A-to-B [retrained] | train on simulation (A), test on silicone (B) | retrained | **0.7807** | **0.3259** | 0.1143 | 2.85x | 12700 | 1452 (11.4%) |
| C-to-B [pretrained] | train on meat (C), test on silicone (B) | pretrained | **0.6786** | **0.2871** | 0.1143 | 2.51x | 12700 | 1452 (11.4%) |
| C-to-B [retrained] | train on meat (C), test on silicone (B) | retrained | **0.7766** | **0.3513** | 0.1143 | 3.07x | 12700 | 1452 (11.4%) |
| A-to-C [pretrained] | train on simulation (A), test on meat (C) | pretrained | **0.8171** | **0.2208** | 0.0716 | 3.09x | 25273 | 1809 (7.2%) |
| A-to-C [retrained] | train on simulation (A), test on meat (C) | retrained | **0.8246** | **0.2237** | 0.0716 | 3.12x | 25273 | 1809 (7.2%) |
