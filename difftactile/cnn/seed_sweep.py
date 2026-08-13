"""Train a configuration under N seeds and report the spread of its metrics.

WHY THIS EXISTS. A single training run is not a result on this project. Measured
on C-to-B, seven seeds gave AUROC 0.604-0.779 and AP 0.239-0.342 - a spread
wider than any difference the three paper configurations claim between one
another. The meat training set is 139 clips, small enough that which subset a run
happens to favour dominates the outcome. Quoting one run therefore says more
about the seed than about the model.

This sweeps seeds 0..N-1, trains a model per seed, and writes mean, standard
deviation, min and max of AUROC and AP to AUROC_RESULTS.md, alongside the
per-seed values so nothing is hidden behind a summary statistic.

EVERY SEED'S WEIGHTS ARE KEPT. Each sweep creates a timestamped directory under
`saved_models_sweeps/`, with one subdirectory per (configuration, seed) holding
that model's checkpoint and its test-loader pickle:

    saved_models_sweeps/20260813-142530/
        sweep.json                       every metric, plus the paths below
        C-to-B_seed00/
            final_segmentation_model_gnn_meat_retrained_C-to-B.pt
            test_loader_gnn_meat_retrained_C-to-B.pickle
        C-to-B_seed01/
            ...

The checkpoint and the pickle travel together on purpose: the pickle holds the
normalisation statistics that checkpoint was trained with, and loading one
without the other silently evaluates on mis-normalised inputs. That exact bug
cost a six-fold understatement of the cross-domain result once already (see the
note on `set_stats` in segmentation_gnn.silicone_to_meat).

Because each run is timestamped, sweeping repeatedly accumulates rather than
overwrites. Nothing prunes these - they are ordinary directories, delete them
when done. `saved_models_sweeps/` is gitignored.

ONE THING IT DELIBERATELY DOES NOT DO: it does not pick a winner. The
best-scoring seed is not a result - selecting it is fitting to the test set
exactly as tuning a threshold there would be. The output is a distribution, and
that is the thing to report.

EACH SEED RUNS IN A SUBPROCESS. Within one process the second run would inherit
CUDA context, cuDNN autotuning state and torch's global RNG from the first, so
"seed k" would not mean the same thing standalone as it does mid-sweep - which
would defeat the point. A fresh interpreter per seed is the only way each run is
genuinely the run you would get by setting DIFFTACTILE_SEED yourself. The cost is
process startup and re-importing torch per seed, which is negligible beside
training.
"""

import json
import os
import statistics
import subprocess
import sys
import time

from difftactile.main.paths import repo_path


# Marker the child process prints its metrics behind, so the parent can pick them
# out of a stdout stream that also carries Lightning's progress bars and tables.
# A sentinel rather than "parse the last line": trainer output is verbose and its
# format is not ours to depend on.
METRICS_MARKER = "###SEED_SWEEP_METRICS###"

# Configurations a sweep is meaningful for - i.e. those that train. Evaluation-only
# runs load a fixed published checkpoint, so their metrics do not depend on a seed
# at all and sweeping them would produce N identical rows.
SWEEPABLE = ("A-to-A", "A-to-B", "C-to-B", "A-to-C")

# Root for the per-sweep timestamped directories. Gitignored.
SWEEP_ROOT = "saved_models_sweeps"

# Tells a child where to write its checkpoint and test-loader pickle; read by
# `segmentation_gnn._retrained_path()`. Spelled out here rather than imported
# from that module, which would pull torch onto the parent's critical path when
# the parent only orchestrates subprocesses. `_run_child()` asserts the two
# agree, so a rename cannot silently break the wiring.
ARTIFACT_DIR_ENV_VAR = "DIFFTACTILE_ARTIFACT_DIR"


def run_directory(timestamp=None):
    """Create and return this sweep's timestamped directory.

    One per invocation, so repeated sweeps accumulate side by side instead of
    overwriting each other's weights. The timestamp is taken once by the parent
    and passed down, so every seed of one sweep lands in the same folder.
    """
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    path = repo_path(f"{SWEEP_ROOT}/{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def seed_directory(run_dir, config, seed):
    """Directory holding one (configuration, seed) model's artifacts."""
    path = os.path.join(run_dir, f"{config}_seed{seed:02d}")
    os.makedirs(path, exist_ok=True)
    return path


def _child_command(config, seed):
    """The command that trains `config` once under `seed`, in a fresh process."""
    return [
        sys.executable, "-m", "difftactile.cnn.seed_sweep", "--child", config, str(seed),
    ]


def run_one_seed(config, seed, run_dir, env=None):
    """Train `config` once under `seed`; return its metrics dict, or None.

    Runs in a subprocess (see the module docstring) with DIFFTACTILE_SEED set.
    Returns None when the child fails, so one bad seed cannot sink the sweep -
    the failure is reported and the remaining seeds still run.

    `run_dir` is this sweep's timestamped directory; the child writes its
    checkpoint and test-loader pickle into a per-seed subdirectory of it, so
    every seed's weights survive instead of being overwritten by the next.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["DIFFTACTILE_SEED"] = str(seed)
    # Training writes figures; never block on a window mid-sweep.
    child_env.setdefault("DIFFTACTILE_HEADLESS", "1")
    # Redirects _retrained_path() (segmentation_gnn.py) into this seed's folder.
    artifact_dir = seed_directory(run_dir, config, seed)
    child_env[ARTIFACT_DIR_ENV_VAR] = artifact_dir

    print(f"\n{'=' * 62}\n {config}  seed {seed}\n{'=' * 62}", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(
        _child_command(config, seed),
        env=child_env,
        cwd=repo_path("."),
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - start

    if proc.returncode != 0:
        print(f"FAILED {config} seed {seed} (exit {proc.returncode})")
        # The tail is where a traceback lands; enough to diagnose without
        # dumping an entire training log into the sweep's output.
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        print(tail)
        print(proc.stderr.strip()[-2000:])
        return None

    metrics = _parse_metrics(proc.stdout)
    if metrics is None:
        print(f"FAILED {config} seed {seed}: child printed no metrics marker")
        return None

    metrics["seed"] = seed
    metrics["seconds"] = duration
    metrics["artifact_dir"] = artifact_dir
    # Recorded per seed so a model can be found later without guessing at
    # filenames; see `checkpoint_for()`.
    metrics["checkpoint"] = _find_checkpoint(artifact_dir)
    print(
        f"seed {seed}: AUROC = {metrics['auroc']:.4f}   AP = {metrics['ap']:.4f}"
        f"   ({duration:.1f} s)"
    )
    print(f"  weights: {metrics['checkpoint'] or '(none written)'}")
    return metrics


def _find_checkpoint(artifact_dir):
    """The .pt file a seed's training run wrote, or None.

    Located by extension rather than by rebuilding the name, because the
    basename depends on the configuration (`*_sim` vs `*_meat`) and on
    `_ACTIVE_CONFIG`. One training run writes exactly one checkpoint here.
    """
    if not os.path.isdir(artifact_dir):
        return None
    for name in sorted(os.listdir(artifact_dir)):
        if name.endswith(".pt"):
            return os.path.join(artifact_dir, name)
    return None


def _parse_metrics(stdout):
    """Pull the metrics dict out of a child's stdout, or None if absent."""
    for line in stdout.splitlines():
        if line.startswith(METRICS_MARKER):
            return json.loads(line[len(METRICS_MARKER):])
    return None


def summarise(values):
    """mean / std / min / max of a list of floats.

    Sample standard deviation (n-1), which is the right one for a spread
    estimated from a handful of runs. Undefined for a single seed, reported as
    0.0 there rather than raising - a one-seed "sweep" is legal, just useless.
    """
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def sweep(config, num_seeds, run_dir, seeds=None):
    """Train `config` once per seed and summarise the spread.

    `seeds` overrides the default 0..num_seeds-1 when a specific set is wanted.
    Returns a dict with the per-seed rows and the summary statistics.
    """
    if config not in SWEEPABLE:
        raise ValueError(
            f"cannot sweep {config!r}; expected one of {list(SWEEPABLE)}"
        )
    seeds = list(seeds) if seeds is not None else list(range(num_seeds))

    runs = []
    for seed in seeds:
        result = run_one_seed(config, seed, run_dir)
        if result is not None:
            runs.append(result)

    summary = {
        "config": config,
        "seeds_requested": seeds,
        "runs": runs,
        "auroc": summarise([r["auroc"] for r in runs]),
        "ap": summarise([r["ap"] for r in runs]),
        # IoU gets the same treatment, which is the point of a sweep for the
        # manuscript's table: a single-seed IoU is as seed-dependent as a
        # single-seed AUROC, and more so on the small meat training set.
        "iou_foreground": summarise([r["iou_foreground"] for r in runs]),
        "iou_background": summarise([r["iou_background"] for r in runs]),
    }

    # In-domain (held-out simulation) figures, present only for the
    # simulation-trained configurations - C-to-B trains on meat and has no
    # same-distribution split to report. `None` when absent, which the Markdown
    # renders as a skipped section rather than an empty table.
    if runs and "in_domain_iou_foreground" in runs[0]:
        summary["in_domain"] = {
            "auroc": summarise([r["in_domain_auroc"] for r in runs]),
            "ap": summarise([r["in_domain_ap"] for r in runs]),
            "iou_foreground": summarise([r["in_domain_iou_foreground"] for r in runs]),
            "iou_background": summarise([r["in_domain_iou_background"] for r in runs]),
        }
    if runs:
        # Constant across seeds (same test set), so carried once rather than per row.
        summary["chance"] = runs[0].get("chance")
    return summary


def format_markdown(summaries, run_dir=None):
    """Render sweep results as the Markdown section appended to AUROC_RESULTS.md."""
    lines = [
        "## Seed sweep",
        "",
        "Each configuration trained from scratch once per seed, then scored. **This is the",
        "number to report** - a single training run on this project reflects its seed as much",
        "as the model, because the training sets are small enough that which subset a run",
        "favours dominates the outcome.",
        "",
        "Read the **spread**, not the best row. Selecting the highest-scoring seed would be",
        "fitting to the test set exactly as tuning a decision threshold on it would be. When",
        "comparing two configurations, compare their distributions - a gap smaller than the",
        "seed spread is not evidence of anything.",
        "",
        "### Threshold-free metrics",
        "",
        "| Config | Seeds | AUROC mean ± std | AUROC range | AP mean ± std | AP range | Chance |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        if not s["runs"]:
            lines.append(f"| {s['config']} | 0 | not run | - | - | - | - |")
            continue
        a, p = s["auroc"], s["ap"]
        chance = s.get("chance")
        chance_cell = f"{chance:.4f}" if chance is not None else "-"
        lines.append(
            f"| {s['config']} | {a['n']} | "
            f"**{a['mean']:.4f} ± {a['std']:.4f}** | {a['min']:.4f}–{a['max']:.4f} | "
            f"**{p['mean']:.4f} ± {p['std']:.4f}** | {p['min']:.4f}–{p['max']:.4f} | "
            f"{chance_cell} |"
        )

    lines += [
        "",
        "### Intersection over union",
        "",
        "Over the frame-by-frame marker predictions, at the decision threshold — **this is the",
        "table to quote in the manuscript**, with the ± rather than a single seed's value.",
        "*Foreground* is the vessel-present class and *background* the vessel-absent one; each",
        "is the agreement between prediction and ground truth over that class, so neither is",
        "\"what the ground truth says\" alone. Background IoU is always the flattering number,",
        "since the negative class dominates — read the foreground column as the real result.",
        "",
        "| Config | Seeds | Foreground IoU mean ± std | Foreground range | Background IoU mean ± std | Background range |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        if not s["runs"]:
            lines.append(f"| {s['config']} | 0 | not run | - | - | - |")
            continue
        fg, bg = s["iou_foreground"], s["iou_background"]
        lines.append(
            f"| {s['config']} | {fg['n']} | "
            f"**{fg['mean']:.4f} ± {fg['std']:.4f}** | {fg['min']:.4f}–{fg['max']:.4f} | "
            f"{bg['mean']:.4f} ± {bg['std']:.4f} | {bg['min']:.4f}–{bg['max']:.4f} |"
        )

    # In-domain reference, for the simulation-trained configurations only.
    in_domain = [s for s in summaries if s.get("in_domain")]
    if in_domain:
        lines += [
            "",
            "### In-domain reference (simulation → simulation)",
            "",
            "The same models scored on the **held-out 15% of the simulated dataset** — same",
            "distribution as their training data, never seen during training. The gap between",
            "this and the cross-domain table above **is the sim-to-real transfer cost**, which",
            "is the quantity the project exists to measure, so the pair is worth reporting",
            "together.",
            "",
            "This is the simulated **test** split, not the validation split: validation drives",
            "early stopping and checkpoint selection, so a number read off it is optimistic by",
            "construction. Only the simulation-trained configurations (A→B, A→C) appear here —",
            "C→B trains on meat and has no same-distribution split to report.",
            "",
            "| Config | Seeds | AUROC mean ± std | AP mean ± std | Foreground IoU mean ± std | Background IoU mean ± std |",
            "|---|---|---|---|---|---|",
        ]
        for s in in_domain:
            d = s["in_domain"]
            a, p = d["auroc"], d["ap"]
            fg, bg = d["iou_foreground"], d["iou_background"]
            lines.append(
                f"| {s['config']} | {a['n']} | "
                f"**{a['mean']:.4f} ± {a['std']:.4f}** | "
                f"**{p['mean']:.4f} ± {p['std']:.4f}** | "
                f"**{fg['mean']:.4f} ± {fg['std']:.4f}** | "
                f"{bg['mean']:.4f} ± {bg['std']:.4f} |"
            )

    if run_dir is not None:
        rel = os.path.relpath(run_dir, repo_path("."))
        lines += [
            "",
            f"Weights for every seed above are preserved under `{rel}/`, one subdirectory per",
            "(configuration, seed), each holding the checkpoint **and** the test-loader pickle",
            "carrying the normalisation statistics it was trained with. `sweep.json` in that",
            "directory repeats these metrics in machine-readable form.",
        ]

    lines += ["", "### Per-seed values", "",
              "Shown in full so the summary above can be checked, and so an outlier is visible",
              "rather than averaged away.", ""]
    for s in summaries:
        if not s["runs"]:
            continue
        lines += [f"**{s['config']}**", "",
                  "| Seed | AUROC | AP | IoU fg | IoU bg | Seconds | Weights |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(s["runs"], key=lambda r: r["seed"]):
            ckpt = r.get("checkpoint")
            where = f"`{os.path.basename(os.path.dirname(ckpt))}/`" if ckpt else "-"
            lines.append(
                f"| {r['seed']} | {r['auroc']:.4f} | {r['ap']:.4f} | "
                f"{r['iou_foreground']:.4f} | {r['iou_background']:.4f} | "
                f"{r['seconds']:.1f} | {where} |"
            )
        failed = set(s["seeds_requested"]) - {r["seed"] for r in s["runs"]}
        if failed:
            lines.append("")
            lines.append(f"Seeds that failed to run: {sorted(failed)}")
        lines.append("")

    lines += [
        "> Each sweep writes a **new timestamped directory**, so sweeping repeatedly",
        "> accumulates rather than overwriting. Nothing prunes them - delete them when done.",
        "> The published checkpoints and the ordinary `*_retrained_<config>` artifacts are",
        "> untouched by a sweep.",
        "",
    ]
    return "\n".join(lines)


def write_markdown(summaries, md_path=None, run_dir=None):
    """Append (or add) the seed-sweep section to AUROC_RESULTS.md.

    The scenario table written by `auroc_all_scenarios` is preserved: this
    replaces only a previous "## Seed sweep" section, so the two can coexist in
    one file whichever order they were produced in.
    """
    md_path = md_path or repo_path("AUROC_RESULTS.md")
    section = format_markdown(summaries, run_dir=run_dir)

    existing = ""
    if os.path.exists(md_path):
        with open(md_path) as f:
            existing = f.read()

    marker = "## Seed sweep"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n\n"
    elif existing:
        existing = existing.rstrip() + "\n\n"
    else:
        existing = "# Ranking metrics\n\n"

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w") as f:
        f.write(existing + section)
    return md_path


def main(configs, num_seeds, seeds=None):
    """Sweep each configuration and write the combined summary."""
    # One directory for the whole invocation, stamped once so every seed of
    # every configuration in this sweep lands together.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_directory(timestamp)
    print(f"Sweep artifacts: {run_dir}")

    summaries = [sweep(c, num_seeds, run_dir, seeds=seeds) for c in configs]

    print(f"\n{'=' * 62}\n Seed sweep summary\n{'=' * 62}")
    for s in summaries:
        if not s["runs"]:
            print(f"{s['config']}: no successful runs")
            continue
        a, p = s["auroc"], s["ap"]
        fg, bg = s["iou_foreground"], s["iou_background"]
        print(f"\n{s['config']}  (over {a['n']} seeds)")
        print(f"  AUROC          {a['mean']:.4f} ± {a['std']:.4f}   "
              f"[{a['min']:.4f}, {a['max']:.4f}]")
        print(f"  AP             {p['mean']:.4f} ± {p['std']:.4f}   "
              f"[{p['min']:.4f}, {p['max']:.4f}]")
        print(f"  IoU foreground {fg['mean']:.4f} ± {fg['std']:.4f}   "
              f"[{fg['min']:.4f}, {fg['max']:.4f}]")
        print(f"  IoU background {bg['mean']:.4f} ± {bg['std']:.4f}   "
              f"[{bg['min']:.4f}, {bg['max']:.4f}]")
        d = s.get("in_domain")
        if d:
            print("  in-domain (held-out simulation, same distribution as training):")
            print(f"    AUROC          {d['auroc']['mean']:.4f} ± {d['auroc']['std']:.4f}")
            print(f"    AP             {d['ap']['mean']:.4f} ± {d['ap']['std']:.4f}")
            print(f"    IoU foreground {d['iou_foreground']['mean']:.4f} "
                  f"± {d['iou_foreground']['std']:.4f}")
            print(f"    IoU background {d['iou_background']['mean']:.4f} "
                  f"± {d['iou_background']['std']:.4f}")

    # Machine-readable twin of the Markdown, kept beside the weights it
    # describes so a sweep directory is self-contained: metrics, and the path of
    # the checkpoint that produced each one.
    json_path = os.path.join(run_dir, "sweep.json")
    with open(json_path, "w") as f:
        json.dump(
            {"timestamp": timestamp, "num_seeds": num_seeds, "summaries": summaries},
            f, indent=2,
        )

    md_path = write_markdown(summaries, run_dir=run_dir)
    print(f"\nSeed sweep written to: {md_path}")
    print(f"Weights and sweep.json: {run_dir}")
    return summaries


def run_from_cli():
    """Entrypoint for scripts/script_seed_sweep.py: `<num_seeds> [config ...]`."""
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(
            "Usage: python -m difftactile.scripts.script_seed_sweep <num_seeds> [config ...]\n"
            f"Configurations: {', '.join(SWEEPABLE)} (default: all three)\n"
            "\nPrefer ./docker/score_all_scenarios.sh --seeds N"
        )
    try:
        num_seeds = int(argv[0])
    except ValueError:
        raise SystemExit(f"First argument must be the seed count, got {argv[0]!r}")
    if num_seeds < 1:
        raise SystemExit(f"Seed count must be positive, got {num_seeds}")

    configs = argv[1:] or list(SWEEPABLE)
    unknown = [c for c in configs if c not in SWEEPABLE]
    if unknown:
        raise SystemExit(
            f"Cannot sweep {', '.join(unknown)}; expected one of {', '.join(SWEEPABLE)}"
        )
    return main(configs, num_seeds)


def _run_child(config, seed):
    """Train one configuration under one seed, and print its metrics as JSON.

    The child half of the subprocess split. Imported lazily so the parent, which
    only orchestrates, never pays for importing torch.
    """
    os.environ["DIFFTACTILE_SEED"] = str(seed)
    from difftactile.cnn import segmentation_gnn
    from difftactile.cnn.segmentation_gnn import run_scenario

    # The parent spells this constant out to avoid importing torch; fail loudly
    # here if the two ever drift apart, rather than silently writing the
    # per-seed weights back over each other.
    assert segmentation_gnn.ARTIFACT_DIR_ENV_VAR == ARTIFACT_DIR_ENV_VAR, (
        "artifact-dir env var name disagrees between seed_sweep and segmentation_gnn"
    )

    # run_scenario() parses sys.argv itself for its config and --train/--eval,
    # and rejects flags it does not know - so our own `--child <config> <seed>`
    # would reach it and abort the run. Both arguments are passed explicitly
    # below, so the child's argv has no further job: clear it.
    sys.argv = sys.argv[:1]

    metrics = run_scenario(scenario=config, mode="train")
    if not isinstance(metrics, dict):
        raise SystemExit(
            f"{config} --train returned {type(metrics).__name__}, expected a metrics dict"
        )
    print(METRICS_MARKER + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    # `--child <config> <seed>` is the per-seed worker; anything else is a
    # normal parent invocation. Kept on this module rather than a separate
    # script so the two halves cannot drift apart.
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        _run_child(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(
            "Run a sweep through the scoring entrypoint instead:\n"
            "  ./docker/score_all_scenarios.sh --seeds N [config ...]"
        )
