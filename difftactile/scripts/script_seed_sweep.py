"""Train each configuration under N seeds and report the spread of its metrics.

    python -m difftactile.scripts.script_seed_sweep 5           # all three configs
    python -m difftactile.scripts.script_seed_sweep 5 C-to-B    # one config

Appends a "Seed sweep" section to AUROC_RESULTS.md with mean, standard deviation
and range of AUROC and AP, plus every per-seed value.

This TRAINS once per seed, so it is far slower than plain scoring. Prefer
`./docker/score_all_scenarios.sh --seeds N`, which wraps this.
"""

from difftactile.cnn.seed_sweep import run_from_cli

if __name__ == "__main__":
    run_from_cli()
