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
#   sim-to-silicone    Evaluate the sim-trained GNN on the real silicone phantom
#                      and write the ROC curve.
#   sim-to-meat        Train on the real meat trials, test on silicone.
#   silicone-to-meat   Test the silicone-trained checkpoint on meat (no training).
#   all-scenarios      Run the three scenarios above in order. Training writes
#                      *_retrained artifacts, so the published checkpoint used by
#                      sim-to-silicone is never overwritten.
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

usage() { sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

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
        echo "  ./data/restore_data.sh difftactile-data.tar.gz"
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

stage_scenario() {
    banner "Scenario: $1"
    python -m difftactile.scripts.script_iros_gnn "$1"
}

case "${1:-}" in
    check)             stage_check ;;
    sim-short)         stage_sim 1 ;;
    sim-full)          stage_sim "" ;;
    sim-to-silicone)   stage_scenario sim-to-silicone ;;
    sim-to-meat)       stage_scenario sim-to-meat ;;
    silicone-to-meat)  stage_scenario silicone-to-meat ;;
    all-scenarios)
        stage_scenario sim-to-silicone
        stage_scenario silicone-to-meat
        stage_scenario sim-to-meat
        ;;
    -h|--help|"")      usage ;;
    *)                 echo "Unknown stage: $1" >&2; echo; usage; exit 1 ;;
esac
