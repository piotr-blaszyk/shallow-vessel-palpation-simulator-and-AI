# AUROC with the normalisation bug deliberately reintroduced

**This file records a diagnostic experiment, not a result to cite.** The
normalisation bug that the repository fixed was temporarily put *back* into the
evaluation harness, and all six canonical scenarios were re-scored, in order to
test the claim that the paper's low A→C AUROC is explained by that bug.

Reproduced by wrapping the dataset-side normalisation call in
`difftactile/cnn/auroc_all_scenarios.py::_build_test_dataset()` in a dead
`if False:` block:

```python
    # TEMPORARY BUG REINTRODUCTION - not a fix, do not keep.
    if False:
        dataset.set_stats(stats)
    dataset.eval()
```

This mirrors the historical `sim-to-meat-test` branch verbatim — see
`git show sim-to-meat-test:difftactile/cnn/iros_gnn.py`, line 707, where
`test_dataset.set_stats(stats)` sat inside exactly such a block.

Run with:

```bash
docker exec -e DIFFTACTILE_HEADLESS=1 vessel-palpation \
    python -m difftactile.scripts.script_auroc_all_scenarios
```

Hardware: RTX 3080, container `vessel-palpation:cuda12.6`.

Evaluation uses `shuffle=False`, and the two silicone-tested configurations
(A→B, C→B) are bit-exact on re-run. The meat-tested configuration (A→C) is
**not**: three repeat runs of the fixed path gave 0.8183 / 0.8171 / 0.8175
(pretrained) and 0.8253 / 0.8261 / 0.8256 (retrained), a spread of about
±0.001. The cause is `MyDataset.getitem_meat()` (`dataset.py:950`), which
applies a random rotation drawn from {0°, 60°, ..., 300°} — exploiting the
sensor's hexagonal symmetry — on *every* item fetch, including at evaluation
time. Quote A→C to two decimal places (**0.82**); the third decimal is not
stable.

## What the model is fed, before and after the fix

`MyDataset.set_stats()` (`difftactile/cnn/dataset.py:268`) does two things: it
copies the per-channel mean/std arrays onto the dataset, and — the load-bearing
part — it clears the `warmup` flag. `MyDataset.normalise()` (line 1381) is
gated on exactly that flag:

```python
def normalise(self, pos, regular_nodes, edge_attr_spatial, ...):
    if not self.warmup:
        ...   # z-score every node and edge feature
```

`warmup` is initialised to `True` (line 119). So with the call disabled, the
whole normalisation body is skipped silently — no error, no warning.

| | Node/edge features fed to the model | Model weights expect |
|---|---|---|
| **With the bug** (`if False:`) | **Raw, unnormalised** — marker positions and edge vectors in their native units, means far from 0 and standard deviations far from 1 | z-scored inputs |
| **After the fix** (call restored) | **z-scored** using the training-set statistics stored in the checkpoint's test-loader pickle, i.e. `(x - mean) / std` per channel | z-scored inputs |

The critical asymmetry: **the checkpoints were trained on normalised inputs in
both cases.** Only the evaluation path differed. Feeding a trained network
inputs from a different distribution than it was fitted on is a bug, not a
modelling choice.

One subtlety worth stating, because it is what made the bug survive review:
there are *two* different `set_stats()` methods, and the buggy path still
called one of them.

- `GNN.set_stats()` (`segmentation_gnn.py:581`) only assigns
  `self.focal_loss.alpha`. It affects the **training loss** and nothing else —
  at evaluation time it is a no-op. It prints a reassuring
  `focal_loss.alpha=...` line, which is visible in the run log below even with
  the bug active.
- `MyDataset.set_stats()` (`dataset.py:268`) is the one that actually enables
  input normalisation.

The historical path called the model's version and skipped the dataset's, so
the logs looked normal while the inputs were wrong.

## Results with the bug reintroduced

| Scenario | Train → Test | Weights | AUROC (bug) | AUROC (fixed) | Δ | Nodes | Positive |
|---|---|---|---|---|---|---|---|
| A-to-B `pretrained` | simulation → silicone | published ckpt | **0.5000** | 0.7314 | −0.2314 | 12700 | 1452 (11.4%) |
| A-to-B `retrained`  | simulation → silicone | trained here   | **0.5000** | 0.7807 | −0.2807 | 12700 | 1452 (11.4%) |
| C-to-B `pretrained` | meat → silicone       | published ckpt | **0.5422** | 0.6786 | −0.1364 | 12700 | 1452 (11.4%) |
| C-to-B `retrained`  | meat → silicone       | trained here   | **0.5285** | 0.6737 | −0.1452 | 12700 | 1452 (11.4%) |
| A-to-C `pretrained` | simulation → meat     | published ckpt | **0.5256** | 0.8183 | −0.2927 | 25273 | 1809 (7.2%) |
| A-to-C `retrained`  | simulation → meat     | trained here   | **0.5000** | 0.8253 | −0.3253 | 25273 | 1809 (7.2%) |

"AUROC (fixed)" is the current repository result, from `AUROC_RESULTS.md`.

ROC curves for the bugged run are preserved in
`difftactile/output/roc_curves_unnormalised/roc_curve_<config>_<weights>_unnormalised.pdf`.
(The sweep overwrites `AUROC_RESULTS.md` and `difftactile/output/roc_curves/`
in place; both were restored to their correct, fixed-path contents afterwards.)

## Why every scenario collapses to chance

The three AUROC values of exactly `0.5000` are the giveaway. Probing the raw
model outputs under the bug:

| Config | nodes | distinct probabilities | min | max | mean |
|---|---|---|---|---|---|
| A-to-B `pretrained` | 12700 | 7307 | 0.0 | 1.0 | 0.0479 |
| A-to-C `pretrained` | 25273 | 14154 | 0.0 | 1.0 | 0.0485 |

The outputs are not constant — thousands of distinct values survive — but the
**extremes are exactly 0.0 and exactly 1.0 in float32**, i.e. fully saturated.
Unnormalised features are orders of magnitude larger than the scale the
network's weights were fitted for, so the pre-sigmoid logits blow up and
`sigmoid()` saturates. Once a large fraction of nodes is pinned at exactly 0.0,
those nodes are all tied, and tied scores contribute 0.5 to the AUROC by
definition. Where the saturation is total, AUROC lands on exactly 0.5.

This matters for interpreting the paper's number: the bug does **not** produce
a mildly degraded ranking. It destroys the ranking almost entirely, in every
configuration, symmetrically.

## What this does and does not establish

**Establishes.** The fix is causally responsible for the A→C improvement, and
the direction and mechanism are exactly as documented. Restoring the call moves
A→C `pretrained` from 0.5256 to 0.82. The mechanism — `warmup` gating a
silent no-op — is confirmed directly rather than inferred.

**Complicates the simple story.** Under the bug, *all six* scenarios collapse
to near-chance, including A→B and C→B. But the paper reports A→B at 0.72 and
C→B at 0.68, and the fixed repository reproduces both to rounding (0.7314,
0.6786). So the paper's A→B and C→B numbers were plainly **not** produced by a
bugged path — only its A→C number was. That is consistent with the bug's
history: it lived in the sim→meat evaluation path specifically
(`iros_gnn.py::main()`, the A→C route), not in the shared A→B / C→B routes.

**Does not establish.** The bugged A→C here scores 0.5256, whereas the paper
reports ≈0.60. These do not match, so this run does not reproduce the paper's
A→C figure exactly — it reproduces the *failure mode*, not the precise value.
The residual gap is expected, since the historical path differed from today's
harness in more than the one line: it used the old `iros`/`icra` artifact
names, `target_difficulty = 0.0` rather than `1.0`, a different dataset split
routine, and `batch_size=16`. Pinning 0.60 exactly would mean re-running the
frozen `sim-to-meat-test` branch against the pre-rename bundle, which is a
larger exercise than this diagnostic.

## Scope of the A→C evaluation set

A→C is scored on **10 of the 23** trials in `meat_training_data/clean/` (199
clips × 127 markers = 25273 nodes). **This restriction is deliberate and
correct**: the 10 admitted trials are the clean recordings, and the remaining
13 are duplicate data that must not be scored — including them would inflate
the node count with repeated material and weight the metric toward whichever
trials happen to be duplicated.

The mechanism, since it is not obvious from a first reading:
`create_splits_meat(all_to_test=True)` routes every *loaded* trial to the test
split, but the pool was already filtered at load time by
`populate_clips_meat()` (`dataset.py:204`), which admits only
`MEAT_TRAIN_TRIALS | MEAT_VALIDATION_TRIALS` — 10 trial IDs. `all_to_test=True`
re-routes that filtered pool rather than widening it, so the docstring's "every
trial goes to the test split" means every *admitted* trial, not all 23.

The corrected 0.82 therefore describes this designated 10-trial evaluation set,
which is the intended comparison and is consistent across the pretrained and
retrained rows.

## Verdict on the corrected number

The corrected A→C AUROC of **0.82** (published checkpoint) is sound, and is
the number to use:

1. It comes from the same uniform harness that reproduces the paper's other two
   configurations to rounding, which is the strongest available evidence that
   the harness is not itself flawed.
2. It is stable on re-running: three repeats span 0.8171–0.8183, so the
   quotable figure is **0.82** to two decimal places.
3. The retrained checkpoint independently reaches ≈0.826 — a separate training
   run, agreeing closely, so the result is not an artifact of one set of
   weights.
4. The corrected direction is the physically sensible one: a model trained on
   normalised inputs should be evaluated on normalised inputs.

Use the `pretrained` row (**0.82**) when replacing the paper's 0.60, since
that is the published checkpoint and therefore the like-for-like comparison.
The corresponding IoU correction is vein 0.03 → **0.198**, background
0.88 → 0.809.

As `RESULTS.md` already notes, this weakens the manuscript's claim that
performance degrades under the larger domain shift to meat: A→C is no longer
the worst configuration by AUROC. A domain gap does remain in per-node IoU
(0.192 on meat vs 0.231 on silicone), so the narrower claim that fine
marker-level classification is harder under domain shift still holds.

## Current repository state

**The normalisation fix has been re-applied.**
`difftactile/cnn/auroc_all_scenarios.py` is byte-identical to its committed
state again (`git diff` is empty), and the full six-scenario sweep was re-run
to confirm the corrected numbers return.

`AUROC_RESULTS.md` and `difftactile/output/roc_curves/` hold the **correct,
fixed-path** numbers. Note that the sweep's `write_markdown()` emits only the
bare results table, so a plain re-run replaces the curated commentary in
`AUROC_RESULTS.md` (the "Paper" comparison column, the granularity note, the
A→C discussion). That file was restored from git after each run here; restore
it with `git checkout -- AUROC_RESULTS.md` if a future sweep flattens it.
