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
# Environment:
#   DIFFTACTILE_HEADLESS=1   Skip GUI windows (default inside this script for the
#                            sim stages when no DISPLAY is set).
#   DIFFTACTILE_NUM_LOOPS=N  Override the simulator loop count.
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
        echo "Data missing. Download the Zenodo bundle and run:"
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
    python -m difftactile.scripts.script_iros_gnn "${config}" "$@"
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
