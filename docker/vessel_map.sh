#!/usr/bin/env bash
#
# Bird's-eye vessel localisation map for the silicone phantom.
#
# Lifts each per-marker prediction to 3D using the sensor pose from robot
# kinematics, then projects it onto the horizontal phantom surface at 1 mm per
# pixel (the 2D->3D->2D projection), and renders that top view against the
# ground-truth vessel location.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# The figures are written to disk, so a display is optional.
#
# Usage:
#   ./docker/vessel_map.sh [--cached]
#
# Options:
#   --cached   Reuse the per-marker probabilities in difftactile/output/exp_probs.npz
#              instead of running the model again. Fast, but only works once a
#              previous run has produced that cache - it is NOT in the published
#              data bundle, so the first run must do inference.
#
# SILICONE ONLY. The workspace bounds, the sensor-to-phantom offset and the
# marker reshapes are all specific to the silicone phantom rig, so there is no
# meat equivalent of this map - which is also why the paper omits the
# sim->meat localisation map.
#
# Outputs (under difftactile/output/):
#   confusion_overlay_vein_map.png            ground truth vs prediction
#                                             (TP white, FP red, FN blue, TN black)
#   segmentation_mask_predicted_aggregated.png  predicted vessel mask alone
#   exp_overlay_downscaled.pdf                the multi-panel comparison figure:
#                                             photo-derived ground truth,
#                                             video-derived ground truth,
#                                             prediction, confusion overlay
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

RERUN=1
for arg in "$@"; do
    case "${arg}" in
        --cached)  RERUN=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

# No display -> never block on a window; the figures on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

CACHE="difftactile/output/exp_probs.npz"
if [ "${RERUN}" = "0" ] && [ ! -f "${CACHE}" ]; then
    echo "ERROR: --cached given but ${CACHE} does not exist." >&2
    echo "Run without --cached once to produce it." >&2
    exit 1
fi

if [ "${RERUN}" = "1" ]; then
    echo "Running inference over the silicone clips, then building the vessel map."
    export DIFFTACTILE_RERUN_INFERENCE=1
else
    echo "Reusing cached probabilities from ${CACHE}."
fi

python -m difftactile.scripts.script_predict_exp

echo
echo "Vessel map written to difftactile/output/:"
for f in confusion_overlay_vein_map.png \
         segmentation_mask_predicted_aggregated.png \
         exp_overlay_downscaled.pdf; do
    [ -f "difftactile/output/${f}" ] && echo "  ${f}"
done
