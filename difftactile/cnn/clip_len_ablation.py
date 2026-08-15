"""Ablation over the GNN's temporal window: which clip length works best?

THE QUESTION. The model takes a `clip_len`-frame window of marker positions and
reports a prediction for the window's central frame. Is that temporal context
actually earning its keep, and how much of it is needed? This trains the A-to-B
configuration (train on simulation, test on real silicone) once per clip length
in CLIP_LENS and compares the results.

THE METRIC. Foreground IoU (the vessel-present class) on the silicone test set,
at the standard decision threshold - the same quantity the manuscript's IoU
table reports. AUROC and AP are recorded alongside for context, but the ranking
in the summary table is by foreground IoU.

HOW IT RUNS. Each (clip length, seed) trains in a fresh subprocess, reusing the
seed sweep's child mechanism (cnn/seed_sweep.py) - same METRICS_MARKER JSON
handshake, same per-run artifact directory. The clip length reaches the child
through the DIFFTACTILE_CLIP_LEN environment variable, applied when the child
imports SYSTEM_PARAMS (see main/constants.py), so dataset windowing and model
input dimensions stay consistent within a run without touching
system-params.json.

Odd lengths only: the reported prediction is the central frame of the window
(dataset.get_mask marks clip_len // 2), which only names a unique centre for an
odd length. clip_len 1 means "no temporal context at all" - the graph then has
no temporal edges, which dataset.py handles explicitly.

SEEDS. Defaults to N_SEEDS_DEFAULT seeds per clip length, reported as
mean +/- std: a single training run on this project reflects its seed as much
as the model (see cnn/seed_sweep.py's docstring), and an ablation that compares
four single runs would mostly be comparing four seeds.

Artifacts land under saved_models_ablation/<timestamp>/clip_len_XX/, one
subdirectory per (clip length, seed), each holding that run's checkpoint and
test-loader pickle. The summary table is written to CLIP_LEN_ABLATION.md and
printed.

Entrypoint: ./docker/ablation_clip_len.sh  (inside the container), or
            python -m difftactile.scripts.script_clip_len_ablation
"""

import json
import os
import sys
import time

from difftactile.main.paths import repo_path
from difftactile.cnn import seed_sweep

# Temporal window lengths under study. 7 is the published configuration.
CLIP_LENS = (1, 3, 5, 7)

# The configuration the ablation is scored on: train on simulation, test on
# real silicone - the project's headline sim-to-real setting.
CONFIG = "A-to-B"

CLIP_LEN_ENV_VAR = "DIFFTACTILE_CLIP_LEN"

# Root for the per-ablation timestamped directories. Gitignored, like the seed
# sweep's saved_models_sweeps/.
ABLATION_ROOT = "saved_models_ablation"

# Seeds per clip length. Three is the smallest count that gives a usable
# mean +/- std; raise it via the CLI when more confidence is wanted.
N_SEEDS_DEFAULT = 3


def run_one_clip_len(clip_len, seeds, root_dir):
    """Train CONFIG once per seed at `clip_len`; return the per-seed metrics."""
    run_dir = os.path.join(root_dir, f"clip_len_{clip_len:02d}")
    os.makedirs(run_dir, exist_ok=True)
    # The child reads the override at import time (main/constants.py), so it
    # must travel in the environment rather than argv.
    env = dict(os.environ)
    env[CLIP_LEN_ENV_VAR] = str(clip_len)

    runs = []
    for seed in seeds:
        print(f"\n--- clip_len {clip_len}, seed {seed} ---", flush=True)
        result = seed_sweep.run_one_seed(CONFIG, seed, run_dir, env=env)
        if result is not None:
            result["clip_len"] = clip_len
            runs.append(result)
    return runs


def summarise_clip_len(clip_len, runs):
    """Summary row for one clip length: mean/std/range of each metric."""
    return {
        "clip_len": clip_len,
        "runs": runs,
        "iou_foreground": seed_sweep.summarise([r["iou_foreground"] for r in runs]),
        "iou_background": seed_sweep.summarise([r["iou_background"] for r in runs]),
        "auroc": seed_sweep.summarise([r["auroc"] for r in runs]),
        "ap": seed_sweep.summarise([r["ap"] for r in runs]),
    }


def format_markdown(summaries, seeds, root_dir):
    """The small results table the ablation exists to produce."""
    lines = [
        "# Clip-length ablation",
        "",
        "How many video frames should the GNN see at once? The model takes a",
        "`clip_len`-frame window of marker positions and predicts the central frame;",
        f"this trains the {CONFIG} configuration (train on simulation, test on real",
        "silicone) at each window length and scores it on the silicone test set.",
        "",
        f"Each row is {len(seeds)} training run(s) (seeds {', '.join(map(str, seeds))}),",
        "reported as mean ± std - single runs on this project reflect their seed as much",
        "as the model (see the seed sweep in AUROC_RESULTS.md).",
        "",
        "**The deciding metric is foreground IoU** (the vessel-present class, at the",
        "standard decision threshold); AUROC and AP are context.",
        "",
        "| clip_len | Foreground IoU mean ± std | Foreground IoU range | AUROC mean ± std | AP mean ± std |",
        "|---|---|---|---|---|",
    ]
    best = best_clip_len(summaries)
    for s in summaries:
        if not s["runs"]:
            lines.append(f"| {s['clip_len']} | (all runs failed) | - | - | - |")
            continue
        fg, a, p = s["iou_foreground"], s["auroc"], s["ap"]
        marker = " **(best)**" if s["clip_len"] == best else ""
        lines.append(
            f"| {s['clip_len']}{marker} | "
            f"**{fg['mean']:.4f} ± {fg['std']:.4f}** | "
            f"{fg['min']:.4f}–{fg['max']:.4f} | "
            f"{a['mean']:.4f} ± {a['std']:.4f} | "
            f"{p['mean']:.4f} ± {p['std']:.4f} |"
        )
    rel = os.path.relpath(root_dir, repo_path("."))
    lines += [
        "",
        f"Checkpoints and per-run artifacts: `{rel}/clip_len_XX/`;",
        "`ablation.json` there repeats these numbers in machine-readable form.",
        "",
    ]
    return "\n".join(lines)


def best_clip_len(summaries):
    """The clip length with the highest mean foreground IoU, or None."""
    scored = [s for s in summaries if s["runs"]]
    if not scored:
        return None
    return max(scored, key=lambda s: s["iou_foreground"]["mean"])["clip_len"]


def main(clip_lens=CLIP_LENS, num_seeds=N_SEEDS_DEFAULT):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root_dir = repo_path(f"{ABLATION_ROOT}/{timestamp}")
    os.makedirs(root_dir, exist_ok=True)
    seeds = list(range(num_seeds))
    print(f"Clip-length ablation over {list(clip_lens)}, seeds {seeds}")
    print(f"Artifacts: {root_dir}")

    summaries = []
    for clip_len in clip_lens:
        runs = run_one_clip_len(clip_len, seeds, root_dir)
        summaries.append(summarise_clip_len(clip_len, runs))

    # Machine-readable twin of the Markdown, kept beside the weights.
    with open(os.path.join(root_dir, "ablation.json"), "w") as f:
        json.dump(
            {"timestamp": timestamp, "config": CONFIG, "seeds": seeds,
             "summaries": summaries},
            f, indent=2,
        )

    md = format_markdown(summaries, seeds, root_dir)
    md_path = repo_path("CLIP_LEN_ABLATION.md")
    with open(md_path, "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Written to: {md_path}")
    return summaries


def run_from_cli():
    """Entrypoint for scripts/script_clip_len_ablation.py: `[--seeds N]`."""
    argv = sys.argv[1:]
    num_seeds = N_SEEDS_DEFAULT
    if argv and argv[0] in ("--seeds",) and len(argv) > 1:
        num_seeds = int(argv[1])
        argv = argv[2:]
    if argv:
        raise SystemExit(
            "Usage: python -m difftactile.scripts.script_clip_len_ablation [--seeds N]\n"
            f"Trains {CONFIG} at clip lengths {list(CLIP_LENS)}, N seeds each "
            f"(default {N_SEEDS_DEFAULT})."
        )
    if num_seeds < 1:
        raise SystemExit(f"--seeds must be positive, got {num_seeds}")
    return main(num_seeds=num_seeds)
