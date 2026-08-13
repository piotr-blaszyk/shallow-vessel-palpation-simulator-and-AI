"""Ranking-metric evaluation across all six (train -> test) x (weights) scenarios.

Reports **AUROC** and **average precision (AP)** for each, both threshold-free
and ranking-based, with one ROC and one PR figure per scenario. See
`cnn/curve_plots.py` for why both are reported rather than either alone.


The paper reports three transfer configurations:

    A-to-B   train on simulation (A),  test on silicone (B)
    C-to-B   train on meat (C),        test on silicone (B)
    A-to-C   train on simulation (A),  test on meat (C)

Each can be scored either from the *published* checkpoint shipped in the Zenodo
bundle, or from a checkpoint produced locally by a `--train` run (the
`*_retrained_<config>` artifacts written by `_retrained_path()`), giving six
scenarios in total.

Only A-to-B previously computed a ROC curve (`segmentation_gnn.evaluate_and_plot_roc`);
`C-to-B` and `A-to-C` ran `trainer.test()` and reported IoU only. This module
factors the probability-collection and ROC-plotting steps out so that the same
measurement is applied uniformly to every scenario, and writes one PDF per
scenario into a dedicated folder.

Usage:
    python -m difftactile.scripts.script_auroc_all_scenarios              # all six
    python -m difftactile.scripts.script_auroc_all_scenarios A-to-B       # one config, both weightings
    python -m difftactile.scripts.script_auroc_all_scenarios A-to-B --pretrained
"""

import os
import pickle
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import matplotlib
from difftactile.main.display import is_headless

if is_headless():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from difftactile.cnn.common import *
from difftactile.cnn.dataset import *
from difftactile.cnn.curve_plots import plot_pr, plot_roc
from difftactile.cnn.segmentation_gnn import GNN
from difftactile.main.paths import repo_path


# One PDF per scenario in each directory, named after the scenario. ROC and PR
# are kept apart so a directory listing is one curve type, at a glance.
ROC_DIR = "difftactile/output/roc_curves"
PR_DIR = "difftactile/output/pr_curves"

# The three paper configurations, described by which checkpoint they load and
# which dataset they are scored on.
#
#   arch          : GNN architecture the checkpoint was trained with
#   ckpt_key      : SYSTEM_PARAMS.files key of the published checkpoint
#   stats_key     : SYSTEM_PARAMS.files key of the test-loader pickle holding the
#                   normalisation statistics the checkpoint expects
#   test_dataset  : which real dataset the model is evaluated on
CONFIGS = {
    "A-to-B": {
        "description": "train on simulation (A), test on silicone (B)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "silicone",
    },
    "C-to-B": {
        "description": "train on meat (C), test on silicone (B)",
        "arch": "compact",
        "ckpt_key": "final_segmentation_model_gnn_meat",
        "stats_key": "test_loader_gnn_meat",
        "test_dataset": "silicone",
    },
    "A-to-C": {
        "description": "train on simulation (A), test on meat (C)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "meat",
    },
}

# The two weight sources. "pretrained" is the published Zenodo checkpoint;
# "retrained" is what a local `--train` run of the same configuration wrote.
WEIGHTS = ("pretrained", "retrained")


def _retrained_variant(rel, config):
    """Path of the `*_retrained_<config>` artifact matching a published one.

    Mirrors `segmentation_gnn._retrained_path()`, which is what a `--train` run
    uses to avoid clobbering the published checkpoints.
    """
    base, ext = os.path.splitext(rel)
    return f"{base}_retrained_{config}{ext}"


def _resolve_artifacts(config, weights):
    """Return (checkpoint_path, stats_pickle_path) for a scenario.

    For the retrained weighting both the checkpoint and the test-loader pickle
    come from the local training run, since a retrained model's normalisation
    statistics are the ones that run computed - not the published ones.
    """
    cfg = CONFIGS[config]
    ckpt_rel = getattr(SYSTEM_PARAMS.files, cfg["ckpt_key"])
    stats_rel = getattr(SYSTEM_PARAMS.files, cfg["stats_key"])
    if weights == "retrained":
        ckpt_rel = _retrained_variant(ckpt_rel, config)
        stats_rel = _retrained_variant(stats_rel, config)
    return repo_path(ckpt_rel), repo_path(stats_rel)


def _load_stats(stats_path):
    """Read the normalisation statistics a checkpoint was trained with.

    Simulation loaders key their stats by curriculum difficulty; meat loaders
    store a single flat dict. Returns (stats, difficulty_or_None).
    """
    with open(stats_path, "rb") as f:
        test_data = pickle.load(f)
    if has_flat_stats(test_data):
        return test_data["dataset_stats"], None
    all_stats = test_data["dataset_stats"]
    # train_on_sim() trains at difficulty 1.0 and stores {1.0: stats}; fall back
    # to whatever single entry exists if a loader was written differently.
    difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
    return all_stats[difficulty], difficulty


def _build_test_dataset(which, stats, difficulty):
    """Construct the dataset a configuration is evaluated on.

    `which` is "silicone" (dataset B) or "meat" (dataset C). In both cases every
    sample is used for evaluation - nothing is held back, since no fitting
    happens here.
    """
    if which == "silicone":
        full = MyDataset(
            scheme="single_dataset",
            sim_exp="exp",
            data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
            apply_augmentations=False,
            name="silicone",
        )
        # The whole silicone set is the evaluation set for these transfer
        # configurations, so it is requested as the "train" split of a 100/0/0
        # split rather than being subsetted.
        dataset, _, _ = full.create_splits(train_size=1.0, val_size=0.0, test_size=0.0)
        if difficulty is not None:
            dataset.set_difficulty_level(difficulty)
    elif which == "meat":
        full = MyDataset(
            scheme="meat",
            sim_exp="apple",
            data_dir="banana",
            apply_augmentations=False,
            name="meat",
        )
        # Pure transfer: every trial goes to the test split.
        _, _, dataset = full.create_splits(all_to_test=True)
    else:
        raise ValueError(f"unknown test dataset {which!r}")
    dataset.set_stats(stats)
    dataset.eval()
    return dataset


def _collect_probabilities(model, dataset, device, batch_size, num_workers):
    """Run the model over the dataset and return (probabilities, labels).

    Only the masked (valid) marker nodes contribute, matching how the published
    evaluation paths score predictions.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch, labels_images, poses, metadata, frame_ix in loader:
            batch = batch.to(device)
            x, x_mask, edge_index, _, edge_attr = model.my_prepare_data(
                batch, batch.num_graphs
            )
            out = model(x, edge_index, edge_attr, batch.batch)
            out = out.squeeze(-1)[x_mask]
            mask = batch.mask
            out = out[mask]
            all_probs.append(torch.sigmoid(out).cpu())
            all_labels.append(batch.y[mask].cpu())
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def evaluate_scenario(config, weights):
    """Score one (configuration, weight-source) scenario.

    Returns a dict with the AUROC, the AP and bookkeeping, or a dict carrying
    `error` when the scenario cannot run (typically a missing retrained
    checkpoint).
    """
    cfg = CONFIGS[config]
    label = f"{config} [{weights}]"
    print(f"\n=== {label}: {cfg['description']} ===")

    ckpt_path, stats_path = _resolve_artifacts(config, weights)
    for path, what in ((ckpt_path, "checkpoint"), (stats_path, "stats pickle")):
        if not os.path.exists(path):
            msg = f"missing {what}: {path}"
            print(f"SKIPPED - {msg}")
            return {"config": config, "weights": weights, "error": msg}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(arch=cfg["arch"])
    print(f"loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model = model.to(device)

    stats, difficulty = _load_stats(stats_path)
    model.set_stats(stats)
    dataset = _build_test_dataset(cfg["test_dataset"], stats, difficulty)

    batch_size = getattr(SYSTEM_PARAMS.gnn, "batch_size_large", SYSTEM_PARAMS.gnn.batch_size)
    num_workers = getattr(SYSTEM_PARAMS.gnn, "num_workers_large", SYSTEM_PARAMS.gnn.num_workers)

    start = time.perf_counter()
    all_probs, all_labels = _collect_probabilities(
        model, dataset, device, batch_size, num_workers
    )
    duration = time.perf_counter() - start

    auc = roc_auc_score(all_labels, all_probs)
    # `roc_thresholds` gives the decision threshold at each vertex of the curve,
    # which is what the curve's colour encodes.
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)

    out_path = repo_path(f"{ROC_DIR}/roc_curve_{config}_{weights}.pdf")
    plot_roc(plt, fpr, tpr, all_probs, all_labels, auc, out_path,
             thresholds_roc=roc_thresholds)

    # Average precision, reported alongside AUROC rather than instead of it.
    # Both are threshold-free and ranking-based, but AP ignores true negatives,
    # so on this ~5%-positive problem it cannot be flattered by the negative
    # majority the way AUROC can. `average_precision_score` is the step-wise
    # sum, not the trapezoidal area under the drawn curve - the standard
    # definition, and the one that does not interpolate optimistically.
    ap = average_precision_score(all_labels, all_probs)
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)

    pr_path = repo_path(f"{PR_DIR}/pr_curve_{config}_{weights}.pdf")
    plot_pr(plt, precision, recall, all_probs, all_labels, ap, pr_path,
            thresholds_pr=pr_thresholds)

    n_pos = int(all_labels.sum())
    baseline = n_pos / len(all_labels)
    # The lift over chance is what makes AP readable: its baseline is the
    # positive rate, not the fixed 0.5 that AUROC always has.
    lift = ap / baseline if baseline > 0 else float("nan")
    print(f"AUROC = {auc:.4f}   ({len(all_labels)} nodes, {n_pos} positive)")
    print(f"AP    = {ap:.4f}   (chance = {baseline:.4f}, i.e. {lift:.2f}x lift)")
    print(f"ROC curve written to: {out_path}")
    print(f"PR curve written to:  {pr_path}")
    return {
        "config": config,
        "weights": weights,
        "description": cfg["description"],
        "auroc": float(auc),
        "ap": float(ap),
        "ap_lift": float(lift),
        "checkpoint": ckpt_path,
        "num_nodes": int(len(all_labels)),
        "num_positive": n_pos,
        "positive_fraction": float(baseline),
        "roc_pdf": out_path,
        "pr_pdf": pr_path,
        "seconds": duration,
    }


def main(configs=None, weightings=None):
    """Evaluate the requested scenarios and write a Markdown summary."""
    configs = configs or list(CONFIGS)
    weightings = weightings or list(WEIGHTS)

    results = []
    for config in configs:
        for weights in weightings:
            try:
                results.append(evaluate_scenario(config, weights))
            except Exception as exc:  # keep going; one scenario must not sink the rest
                print(f"FAILED {config} [{weights}]: {exc}")
                results.append({"config": config, "weights": weights, "error": str(exc)})

    md_path = repo_path("AUROC_RESULTS.md")
    write_markdown(results, md_path)
    print(f"\nSummary written to: {md_path}")
    return results


def write_markdown(results, md_path):
    """Render the AUROC / AP table to Markdown."""
    lines = [
        "# Ranking metrics across the six canonical scenarios",
        "",
        "Per-node vessel-classification AUROC and average precision (AP) for each",
        "(train -> test) configuration, scored twice: once from the published Zenodo",
        "checkpoint and once from a checkpoint retrained locally with `--train`.",
        "",
        "Datasets: **A** = simulation, **B** = real silicone phantom, **C** = real meat phantom.",
        "",
        "Both metrics are **threshold-free and ranking-based**: they read only the",
        "ordering of the predicted probabilities, never their absolute scale, so no",
        "decision threshold is chosen anywhere in this table. That is deliberate - in a",
        "sim-to-real setting the output *scale* is the first thing to shift between",
        "domains, and a single-threshold score would confound that with what the model",
        "actually learned.",
        "",
        "They are reported together because each is blind to something the other sees:",
        "",
        "- **AUROC** normalises false positives by the total negative count. On this",
        "  heavily imbalanced problem a large absolute number of false alarms barely",
        "  moves it. Its baseline is always 0.5, which makes it comparable across papers.",
        "- **AP** ignores true negatives entirely, so the negative majority cannot",
        "  flatter it. Its baseline is the **positive rate**, given in the `Chance`",
        "  column, so AP must be read against that - hence the `Lift` column",
        "  (AP / chance), which is how many times better than random ranking the model is.",
        "",
        f"ROC curves: `{ROC_DIR}/roc_curve_<config>_<weights>.pdf`",
        f"PR curves:  `{PR_DIR}/pr_curve_<config>_<weights>.pdf`",
        "",
        "| Scenario | Train -> Test | Weights | AUROC | AP | Chance | Lift | Nodes | Positive |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        name = f"{r['config']} [{r['weights']}]"
        if "error" in r:
            lines.append(f"| {name} | - | {r['weights']} | not run | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {r['description']} | {r['weights']} | **{r['auroc']:.4f}** | "
            f"**{r['ap']:.4f}** | {r['positive_fraction']:.4f} | {r['ap_lift']:.2f}x | "
            f"{r['num_nodes']} | {r['num_positive']} ({r['positive_fraction']:.1%}) |"
        )
    errors = [r for r in results if "error" in r]
    if errors:
        lines += ["", "## Scenarios not run", ""]
        for r in errors:
            lines.append(f"- **{r['config']} [{r['weights']}]** — {r['error']}")
    lines.append("")
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def run_from_cli():
    """Entrypoint for scripts/script_auroc_all_scenarios.py."""
    argv = sys.argv[1:]
    flags = [a for a in argv if a.startswith("--")]
    positional = [a for a in argv if not a.startswith("--")]
    configs = positional or None
    if configs:
        for c in configs:
            if c not in CONFIGS:
                raise SystemExit(f"Unknown configuration {c!r}; expected one of {list(CONFIGS)}")
    weightings = None
    if "--pretrained" in flags:
        weightings = ["pretrained"]
    elif "--retrained" in flags:
        weightings = ["retrained"]
    return main(configs, weightings)
