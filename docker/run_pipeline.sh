#!/usr/bin/env bash
#
# One entrypoint for every key stage of the project, intended to be run INSIDE
# the container (see docker/docker-run.sh + docker/docker-connect.sh).
#
# The stages map onto the pipeline described in the README:
#   simulation -> synthetic marker displacements -> GNN -> evaluation on real data
#
# Usage:
#   ./docker/run_pipeline.sh <stage> [...]
#
# Stages:
#   check              Verify GPU, Taichi, torch and the restored data bundle.
#   sim-short          Simulator data collection, 1 loop (8 trials, ~3 min).
#   sim-full           Simulator data collection, full run (800 trials, ~2h45m measured).
#   A-to-B             Train on simulation, test on silicone (default: evaluate
#                      the published checkpoint and write the ROC curve).
#   C-to-B             Train on the real meat trials, test on silicone.
#   A-to-C             Train on simulation, test on meat (default: evaluate the
#                      published checkpoint, no retraining).
#   all-scenarios      Run the three configurations above in order. Training
#                      writes *_retrained artifacts, so the published
#                      checkpoints the evaluations read are never overwritten.
#
#   Datasets: A = simulated, B = real silicone phantom, C = real meat phantom.
#   Each configuration accepts a trailing --train or --eval, e.g.
#       ./docker/run_pipeline.sh A-to-B --train
#   The older names sim-to-silicone / sim-to-meat / silicone-to-meat still work.
#
# METRICS. Every configuration reports AUROC and average precision (AP), both
# threshold-free and ranking-based - they read only the ORDER of the predicted
# probabilities, never their scale, so no decision threshold is chosen. That
# matters for a sim-to-real project: the output scale is the first thing to
# shift across a domain gap, and a single-threshold score (the IoU also printed)
# confounds that with what the model actually learned. AP is reported beside
# AUROC because it ignores true negatives, so the ~5-11% positive rate here
# cannot flatter it the way it can flatter AUROC; its baseline is the positive
# rate, so the lift over chance is printed with it.
#
# Each writes difftactile/output/roc_curve_<config>.pdf and pr_curve_<config>.pdf.
# To score all six (config x pretrained/retrained) scenarios side by side into
# one table instead, use ./docker/score_all_scenarios.sh.
#
# Environment:
#   DIFFTACTILE_HEADLESS=1   Skip GUI windows (default inside this script for the
#                            sim stages when no DISPLAY is set).
#   DIFFTACTILE_NUM_LOOPS=N  Override the simulator loop count.
#   DIFFTACTILE_SEED=N       Seed for every RNG a training run touches (default
#                            42, printed at the start of each run). Training is
#                            deterministic: same machine, same code, same seed
#                            reproduces a run exactly.
#
#                            BUT THE SPREAD ACROSS SEEDS IS LARGE. Seven seeds on
#                            C-to-B gave AUROC 0.604-0.779 - wider than the gaps
#                            between the configurations themselves, because the
#                            meat training set is only 139 clips. Report a mean
#                            and spread over several seeds, never one run, and
#                            never the best-scoring seed:
#                              for s in 0 1 2 3 4; do
#                                  DIFFTACTILE_SEED=$s ./run_pipeline.sh C-to-B
#                              done
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

# No X display -> force headless so nothing blocks on a window that cannot open.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

# Print the header comment block as help. Reads to the first non-comment line
# rather than a fixed line range, so editing the header cannot truncate it.
usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

banner() {
    echo
    echo "=============================================================="
    echo " $*"
    echo "=============================================================="
}

stage_check() {
    banner "Environment check"
    echo "repo:     ${REPO_DIR}"
    echo "headless: ${DIFFTACTILE_HEADLESS:-0}   DISPLAY=${DISPLAY:-<unset>}"
    echo
    nvidia-smi --query-gpu=name,memory.total --format=csv || echo "WARNING: nvidia-smi unavailable"
    echo
    python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
import taichi as ti
print(f"taichi {ti.__version__}")
import torch_geometric
print(f"torch_geometric {torch_geometric.__version__}")
from difftactile.main.constants import SYSTEM_PARAMS
print(f"config loaded OK (gnn.latent_dim={SYSTEM_PARAMS.gnn.latent_dim})")
PY
    echo
    ./data/restore_data.sh --verify || {
        echo
        echo "Data missing. Download the Zenodo bundle (DOI 10.5281/zenodo.21900934) and run:"
        echo "  wget https://zenodo.org/records/21900934/files/shallow-vessel-palpation-data.tar.gz"
        echo "  ./data/restore_data.sh shallow-vessel-palpation-data.tar.gz"
        return 1
    }
}

stage_sim() {
    local loops="$1"
    banner "Simulator data collection (${loops:-full} loop(s))"
    [ -n "${loops}" ] && export DIFFTACTILE_NUM_LOOPS="${loops}"
    # apply_scaling and pre_main regenerate the derived config the simulator
    # needs; they must run in this order (see scripts/run_all.sh).
    python -m difftactile.scripts.script_apply_scaling
    python -m difftactile.scripts.script_pre_main
    python -m difftactile.scripts.script_main
}

# Runs one (train -> test) configuration. Any extra arguments (--train/--eval)
# are forwarded verbatim to the Python dispatcher.
stage_scenario() {
    local config="$1"; shift
    banner "Configuration: ${config}${*:+ $*}"
    python -m difftactile.scripts.script_segmentation_gnn "${config}" "$@"
}

case "${1:-}" in
    check)             stage_check ;;
    sim-short)         stage_sim 1 ;;
    sim-full)          stage_sim "" ;;
    # Paper notation, plus the pre-rename aliases. The Python dispatcher
    # resolves the aliases itself, so they are simply passed through.
    A-to-B|C-to-B|A-to-C|sim-to-silicone|sim-to-meat|silicone-to-meat|meat-to-silicone)
        config="$1"; shift
        stage_scenario "${config}" "$@"
        ;;
    all-scenarios)
        stage_scenario A-to-B
        stage_scenario A-to-C
        stage_scenario C-to-B
        ;;
    -h|--help|"")      usage ;;
    *)                 echo "Unknown stage: $1" >&2; echo; usage; exit 1 ;;
esac
