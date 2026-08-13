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
| A-to-B [pretrained] | train on simulation (A), test on silicone (B) | pretrained | **0.7314** | **0.2553** | **0.2059** | 0.7562 | 0.1143 | 2.23x | 12700 | 1452 (11.4%) |
| C-to-B [pretrained] | train on meat (C), test on silicone (B) | pretrained | **0.6786** | **0.2871** | **0.1531** | 0.6066 | 0.1143 | 2.51x | 12700 | 1452 (11.4%) |
| A-to-C [pretrained] | train on simulation (A), test on meat (C) | pretrained | **0.8171** | **0.2208** | **0.1982** | 0.8119 | 0.0716 | 3.09x | 25273 | 1809 (7.2%) |
