"""Which trained model instance stands in when ONE model is needed.

Every reported number in this project is a mean +/- std over the five seeds of
the published sweep (see cnn/seed_sweep.py and AUROC_RESULTS.md). Some
artifacts cannot be averaged, though: the frame-by-frame prediction viewer
shows one model's output on one frame, and the bird's-eye vessel map is built
from one model's predictions. For those, the convention is:

    use the BEST of the five seed instances of that configuration,
    where "best" means the highest average precision (AP) on that
    configuration's own test set (AUROC breaks ties).

AP rather than AUROC because the map is thresholded to a high-precision
operating point (see vessel_map.py), and AP summarises the precision-recall
trade-off that operating point is chosen from. Selecting on the test set is a
deliberate, documented optimistic choice for these qualitative artifacts; the
seed spread is quantified separately in AUROC_RESULTS.md, and nothing in the
scored tables uses this selection.

The sweep that counts is pinned in system-params.json as `files.published_sweep`
(a `saved_models_sweeps/<timestamp>` directory holding `sweep.json` and one
subdirectory per (configuration, seed) with the checkpoint AND its test-loader
pickle - the two must travel together, since the pickle carries the
normalisation statistics the checkpoint was trained with).

The LEGACY models are the pre-2026-08-15 checkpoints (trained on the earlier
simulated dataset, temporal window 7). They are kept only because they produced
the accepted manuscript's Fig. 8 / Table 4; see saved_models_legacy/README.md.
Nothing else should load them.

This module imports neither torch nor the dataset code, so shell-level tooling
and the sweep parent (which only orchestrates subprocesses) can use it freely.
"""

import json
import os

from difftactile.main.constants import SYSTEM_PARAMS
from difftactile.main.paths import repo_path

# Configuration -> the arch of its checkpoint and the *_sim / *_meat family of
# its artifacts. Kept here (rather than imported from segmentation_gnn, which
# needs torch) so the mapping is readable without a GPU.
CONFIG_ARCH = {
    "A-to-A": "large",
    "A-to-B": "large",
    "A-to-C": "large",
    "C-to-B": "compact",
}
CONFIG_FAMILY = {
    "A-to-A": "sim",
    "A-to-B": "sim",
    "A-to-C": "sim",
    "C-to-B": "meat",
}

# Temporal window the legacy checkpoints were trained with. Their input layer is
# sized to it (node_dim = 2 + 3 + num_nodes + clip_len), so they only load when
# gnn.clip_len is 7 - the shell entrypoints export DIFFTACTILE_CLIP_LEN=7 for
# `--model legacy`, and `legacy_model()` refuses to proceed otherwise.
LEGACY_CLIP_LEN = 7
LEGACY_ROOT = "saved_models_legacy"


def published_sweep_dir():
    """Absolute path of the pinned sweep, from `files.published_sweep`."""
    path = getattr(SYSTEM_PARAMS.files, "published_sweep", None)
    if not path:
        raise KeyError(
            "system-params.json has no files.published_sweep entry; run a seed "
            "sweep (./docker/score_all_scenarios.sh --seeds 5) and pin its "
            "saved_models_sweeps/<timestamp> directory there."
        )
    return path if os.path.isabs(path) else repo_path(path)


def load_sweep(sweep_dir=None):
    """The parsed sweep.json of a sweep directory (default: the published one)."""
    sweep_dir = sweep_dir or published_sweep_dir()
    with open(os.path.join(sweep_dir, "sweep.json")) as f:
        return json.load(f)


def best_run(runs):
    """The best of a configuration's per-seed runs: highest AP, then AUROC."""
    runs = [r for r in runs if r.get("checkpoint")]
    if not runs:
        raise ValueError("no successful runs with a checkpoint to choose from")
    return max(runs, key=lambda r: (r["ap"], r["auroc"]))


def _artifacts_in(directory):
    """(checkpoint, pickle) inside a per-seed directory. Exactly one of each."""
    names = sorted(os.listdir(directory))
    pts = [n for n in names if n.endswith(".pt")]
    pkls = [n for n in names if n.endswith(".pickle")]
    if len(pts) != 1 or len(pkls) != 1:
        raise FileNotFoundError(
            f"expected one .pt and one .pickle in {directory}, found {pts} / {pkls}"
        )
    return os.path.join(directory, pts[0]), os.path.join(directory, pkls[0])


def sweep_seed_model(config, seed, sweep_dir=None):
    """One named seed's model from a sweep. Returns the same dict as best_model()."""
    sweep_dir = sweep_dir or published_sweep_dir()
    directory = os.path.join(sweep_dir, f"{config}_seed{seed:02d}")
    checkpoint, stats = _artifacts_in(directory)
    return {
        "config": config,
        "kind": "sweep-seed",
        "seed": seed,
        "arch": CONFIG_ARCH[config],
        "checkpoint": checkpoint,
        "stats": stats,
        "sweep_dir": sweep_dir,
        "description": f"{config} seed {seed} of sweep {os.path.basename(sweep_dir)}",
    }


def best_model(config, sweep_dir=None):
    """The best-of-N seed instance of `config` in the published (or given) sweep.

    Returns a dict with `checkpoint`, `stats` (its test-loader pickle), `arch`,
    `seed`, the selection metrics, and a one-line `description`.
    """
    sweep_dir = sweep_dir or published_sweep_dir()
    sweep = load_sweep(sweep_dir)
    summaries = {s["config"]: s for s in sweep["summaries"]}
    if config not in summaries:
        raise KeyError(f"{config} is not in sweep {sweep_dir} ({sorted(summaries)})")
    run = best_run(summaries[config]["runs"])
    # Resolve the per-seed directory relative to the sweep, not from the path
    # recorded in sweep.json: that path is absolute and was written wherever
    # the sweep ran (e.g. inside the container), so it does not survive a move
    # or a restore from the data bundle.
    directory = os.path.join(sweep_dir, os.path.basename(run["artifact_dir"]))
    checkpoint, stats = _artifacts_in(directory)
    return {
        "config": config,
        "kind": "best-of-sweep",
        "seed": run["seed"],
        "ap": run["ap"],
        "auroc": run["auroc"],
        "num_seeds": len(summaries[config]["runs"]),
        "arch": CONFIG_ARCH[config],
        "checkpoint": checkpoint,
        "stats": stats,
        "sweep_dir": sweep_dir,
        "description": (
            f"{config} best-of-{len(summaries[config]['runs'])} seeds "
            f"(seed {run['seed']}, AP {run['ap']:.4f}, AUROC {run['auroc']:.4f}) "
            f"from sweep {os.path.basename(sweep_dir)}"
        ),
    }


def legacy_model(config):
    """The pre-2026-08-15 checkpoint of `config`'s family (sim or meat).

    Only meaningful for the silicone-tested configurations the legacy models
    were used for (A-to-B, C-to-B); refuses others, and refuses to run unless
    the temporal window matches what these checkpoints were built for.
    """
    if config not in ("A-to-B", "C-to-B"):
        raise ValueError(
            f"the legacy models were only ever used for A-to-B and C-to-B "
            f"(the accepted manuscript's Fig. 8 / Table 4), not {config}"
        )
    if SYSTEM_PARAMS.gnn.clip_len != LEGACY_CLIP_LEN:
        raise RuntimeError(
            f"legacy checkpoints need gnn.clip_len == {LEGACY_CLIP_LEN} (they were "
            f"trained with a {LEGACY_CLIP_LEN}-frame window and their input layer "
            f"is sized to it), but clip_len is {SYSTEM_PARAMS.gnn.clip_len}. "
            f"Export DIFFTACTILE_CLIP_LEN={LEGACY_CLIP_LEN} - the shell "
            f"entrypoints do this for --model legacy."
        )
    family = CONFIG_FAMILY[config]
    root = repo_path(f"{LEGACY_ROOT}/{family}")
    checkpoint = os.path.join(root, f"final_segmentation_model_gnn_{family}.pt")
    stats = os.path.join(root, f"test_loader_gnn_{family}.pickle")
    for p in (checkpoint, stats):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"legacy artifact missing: {p}. Restore the data bundle "
                f"(./data/restore_data.sh) - the legacy models ship in it."
            )
    return {
        "config": config,
        "kind": "legacy",
        "seed": None,
        "arch": CONFIG_ARCH[config],
        "checkpoint": checkpoint,
        "stats": stats,
        "description": f"{config} LEGACY (pre-2026-08-15) {family} checkpoint",
    }


def resolve_model(config, choice="best", sweep_dir=None, seed=None):
    """Dispatch on `choice`: "best" (default), "legacy", or "sweep" (+seed)."""
    if choice == "best":
        return best_model(config, sweep_dir=sweep_dir)
    if choice == "legacy":
        return legacy_model(config)
    if choice == "sweep":
        if seed is None:
            raise ValueError("choice='sweep' needs a seed")
        return sweep_seed_model(config, seed, sweep_dir=sweep_dir)
    raise ValueError(f"unknown model choice {choice!r}; expected best, legacy or sweep")
