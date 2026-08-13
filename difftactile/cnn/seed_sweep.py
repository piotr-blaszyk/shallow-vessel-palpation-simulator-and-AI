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

TWO THINGS IT DELIBERATELY DOES NOT DO:

  * It does not pick a winner. The best-scoring seed is not a result - selecting
    it is fitting to the test set exactly as tuning a threshold there would be.
    The output is a distribution, and that is the thing to report.
  * It does not keep the per-seed checkpoints. Training the same configuration
    twice overwrites its `*_retrained_<config>` artifacts in place, so after a
    sweep the checkpoint on disk is simply the last seed's. That is intentional:
    a sweep exists to characterise the spread, not to harvest N models. Train a
    single seed normally if you want a checkpoint you can point at.

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
SWEEPABLE = ("A-to-B", "C-to-B", "A-to-C")


def _child_command(config, seed):
    """The command that trains `config` once under `seed`, in a fresh process."""
    return [
        sys.executable, "-m", "difftactile.cnn.seed_sweep", "--child", config, str(seed),
    ]


def run_one_seed(config, seed, env=None):
    """Train `config` once under `seed`; return its metrics dict, or None.

    Runs in a subprocess (see the module docstring) with DIFFTACTILE_SEED set.
    Returns None when the child fails, so one bad seed cannot sink the sweep -
    the failure is reported and the remaining seeds still run.
    """
    child_env = dict(os.environ if env is None else env)
    child_env["DIFFTACTILE_SEED"] = str(seed)
    # Training writes figures; never block on a window mid-sweep.
    child_env.setdefault("DIFFTACTILE_HEADLESS", "1")

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
    print(
        f"seed {seed}: AUROC = {metrics['auroc']:.4f}   AP = {metrics['ap']:.4f}"
        f"   ({duration:.1f} s)"
    )
    return metrics


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


def sweep(config, num_seeds, seeds=None):
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
        result = run_one_seed(config, seed)
        if result is not None:
            runs.append(result)

    summary = {
        "config": config,
        "seeds_requested": seeds,
        "runs": runs,
        "auroc": summarise([r["auroc"] for r in runs]),
        "ap": summarise([r["ap"] for r in runs]),
    }
    if runs:
        # Constant across seeds (same test set), so carried once rather than per row.
        summary["chance"] = runs[0].get("chance")
    return summary


def format_markdown(summaries):
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

    lines += ["", "### Per-seed values", "",
              "Shown in full so the summary above can be checked, and so an outlier is visible",
              "rather than averaged away.", ""]
    for s in summaries:
        if not s["runs"]:
            continue
        lines += [f"**{s['config']}**", "",
                  "| Seed | AUROC | AP | Seconds |", "|---|---|---|---|"]
        for r in sorted(s["runs"], key=lambda r: r["seed"]):
            lines.append(
                f"| {r['seed']} | {r['auroc']:.4f} | {r['ap']:.4f} | {r['seconds']:.1f} |"
            )
        failed = set(s["seeds_requested"]) - {r["seed"] for r in s["runs"]}
        if failed:
            lines.append("")
            lines.append(f"Seeds that failed to run: {sorted(failed)}")
        lines.append("")

    lines += [
        "> Sweeps do **not** leave one checkpoint per seed: re-training a configuration",
        "> overwrites its `*_retrained_<config>` artifacts, so the checkpoint on disk after a",
        "> sweep is the last seed's. Train a single seed normally if you want a checkpoint to",
        "> point at.",
        "",
    ]
    return "\n".join(lines)


def write_markdown(summaries, md_path=None):
    """Append (or add) the seed-sweep section to AUROC_RESULTS.md.

    The scenario table written by `auroc_all_scenarios` is preserved: this
    replaces only a previous "## Seed sweep" section, so the two can coexist in
    one file whichever order they were produced in.
    """
    md_path = md_path or repo_path("AUROC_RESULTS.md")
    section = format_markdown(summaries)

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
    summaries = [sweep(c, num_seeds, seeds=seeds) for c in configs]

    print(f"\n{'=' * 62}\n Seed sweep summary\n{'=' * 62}")
    for s in summaries:
        if not s["runs"]:
            print(f"{s['config']}: no successful runs")
            continue
        a, p = s["auroc"], s["ap"]
        print(
            f"{s['config']}: AUROC {a['mean']:.4f} ± {a['std']:.4f} "
            f"[{a['min']:.4f}, {a['max']:.4f}]   "
            f"AP {p['mean']:.4f} ± {p['std']:.4f} [{p['min']:.4f}, {p['max']:.4f}]"
            f"   over {a['n']} seeds"
        )

    md_path = write_markdown(summaries)
    print(f"\nSeed sweep written to: {md_path}")
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
    from difftactile.cnn.segmentation_gnn import run_scenario

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
