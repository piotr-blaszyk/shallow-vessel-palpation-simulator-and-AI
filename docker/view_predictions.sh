#!/usr/bin/env bash
#
# Interactive per-frame prediction viewer: step through test-set video frames and
# inspect the confusion overlay (green TP, yellow TN, red FP, blue FN) for any of
# the six canonical scenarios.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# It needs a display: the whole point is the OpenCV windows.
#
# Usage:
#   ./docker/view_predictions.sh <config> [--pretrained|--retrained]
#
# Configurations (A = simulation, B = real silicone, C = real meat):
#   A-to-B    model trained on simulation, predictions shown on silicone
#   C-to-B    model trained on meat,       predictions shown on silicone
#   A-to-C    model trained on simulation, predictions shown on meat
#
# Weights:
#   --pretrained  the published Zenodo checkpoint (default)
#   --retrained   the checkpoint written by a local `--train` run of the same
#                 configuration, i.e. saved_models_*/*_retrained_<config>.pt
#
# The configuration selects BOTH the model weights and the test dataset, so the
# six combinations above are the six scenarios.
#
# Windows: Ground Truth, Hard Prediction, Confusion Matrix, Soft Prediction and
# Metadata are tiled across the screen. Press q to quit.
#
# Examples:
#   ./docker/view_predictions.sh A-to-B
#   ./docker/view_predictions.sh A-to-C --retrained
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

CONFIG=""
WEIGHTS="--pretrained"

for arg in "$@"; do
    case "${arg}" in
        A-to-B|C-to-B|A-to-C) CONFIG="${arg}" ;;
        --pretrained|--retrained) WEIGHTS="${arg}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

if [ -z "${CONFIG}" ]; then
    echo "ERROR: no configuration given." >&2
    echo
    usage
    exit 1
fi

# This viewer exists to be watched, so opt into blocking windows rather than the
# project default of never waiting on a GUI.
export DIFFTACTILE_INTERACTIVE=1

if [ -z "${DISPLAY:-}" ]; then
    cat >&2 <<'EOF'
ERROR: DISPLAY is not set, so no windows can be opened.

This tool is interactive by nature. Start the container with X forwarding
(docker/docker-run.sh passes the host display through) and make sure `xhost`
permits it on the host:

    xhost +local:docker
EOF
    exit 1
fi

echo "Viewing predictions: ${CONFIG} ${WEIGHTS}"
echo "Press q in any window to quit."
exec python -m difftactile.scripts.script_visualise "${CONFIG}" "${WEIGHTS}"
