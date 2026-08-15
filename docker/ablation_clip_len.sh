#!/usr/bin/env bash
#
# Clip-length ablation: how many video frames should the ML model see at once?
#
# Trains the A-to-B configuration (train on simulation, test on real silicone)
# at each temporal window length in {1, 3, 5, 7} and reports which one performs
# best. The deciding metric is FOREGROUND IoU (the vessel-present class) on the
# silicone test set - the same quantity the manuscript's IoU table reports;
# AUROC and AP are recorded alongside for context.
#
# Each clip length is trained once per seed (default 3 seeds) and reported as
# mean +/- std, because a single training run on this project reflects its seed
# as much as the model (see the seed sweep notes in score_all_scenarios.sh).
#
# A short wrapper for:
#     python -m difftactile.scripts.script_clip_len_ablation [--seeds N]
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# Trains 4 x N models, so expect it to take a while.
#
# Usage:
#   ./docker/ablation_clip_len.sh [--seeds N]
#
# Outputs:
#   CLIP_LEN_ABLATION.md                          the summary table
#   saved_models_ablation/<timestamp>/clip_len_XX/  per-run checkpoints + pickles
#   saved_models_ablation/<timestamp>/ablation.json machine-readable results
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --seeds)
            shift
            if [ "$#" -eq 0 ]; then
                echo "ERROR: --seeds needs a count, e.g. --seeds 3" >&2; exit 1
            fi
            case "$1" in ''|*[!0-9]*)
                echo "ERROR: --seeds needs a positive integer, got '$1'" >&2; exit 1 ;;
            esac
            ARGS+=(--seeds "$1") ;;
        --seeds=*)
            N="${1#*=}"
            case "${N}" in ''|*[!0-9]*)
                echo "ERROR: --seeds needs a positive integer, got '${N}'" >&2; exit 1 ;;
            esac
            ARGS+=(--seeds "${N}") ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done

# No display -> never block on a window; the files on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

python -m difftactile.scripts.script_clip_len_ablation "${ARGS[@]}"

echo
echo "Written:"
[ -f CLIP_LEN_ABLATION.md ] && echo "  CLIP_LEN_ABLATION.md    the summary table"
