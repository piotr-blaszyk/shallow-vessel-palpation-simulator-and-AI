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
#   ./docker/vessel_map.sh [--cached] [--model sim|meat]
#
# Options:
#   --model M  Which trained model draws the map. "meat" (default) is the
#              compact meat-trained C-to-B checkpoint - the historical default
#              behind the published figures, whose outputs keep their unsuffixed
#              names. "sim" is the large simulation-trained A-to-B checkpoint;
#              its outputs get a `_sim` suffix (confusion_overlay_vein_map_sim.png
#              etc.) so the two maps sit side by side. Both are evaluated on the
#              same silicone clips.
#   --cached   Reuse the per-marker probabilities in difftactile/output/exp_probs.npz
#              (or exp_probs_sim.npz with --model sim) instead of running the
#              model again. Only works once a previous run of the SAME model
#              has produced that cache, so the first run must do inference.
#              That cache is deliberately not in the data bundle: it is pure model
#              output, and recomputing it from the shipped dataset and checkpoint
#              takes under a second. Running without --cached is the normal path.
#
# SILICONE ONLY. The workspace bounds, the sensor-to-phantom offset and the
# marker reshapes are all specific to the silicone phantom rig, so there is no
# meat equivalent of this map - which is also why the paper omits the
# sim->meat localisation map.
#
# THE COLOUR SCHEME, used by every confusion map this script writes. Read it as
# "what is there, and did we find it":
#
#   GREEN  both say vessel                         (agreement, positive)
#   RED    the reference says vessel, the other does not   (a MISS)
#   BLUE   the other says vessel, the reference does not   (a FALSE ALARM)
#   BLACK  neither says vessel                     (agreement, negative)
#
# Red for misses rather than for false alarms is deliberate: in a palpation
# setting a missed vessel is the dangerous error, so it gets the warning colour.
# It lives in one place, Visualisation.CONFUSION_COLOURS_RGB in cnn/visualise.py.
#
# Outputs (under difftactile/output/):
#
#   1. Prediction vs ground truth - the vessel map proper.
#      confusion_overlay_vein_map.png   the raw map
#      confusion_overlay_vein_map.pdf   the same, with a title and a legend
#      Reference = the VIDEO-derived ground truth, because it is reprojected
#      onto the same grid as the prediction, so the comparison is like for like.
#
#   2. Ground truth from video vs from top-view photo - how well the two
#      INDEPENDENT ground truths agree, which bounds how well any model could
#      possibly score against either.
#      ground_truth_sources_overlay.png   the raw map
#      ground_truth_sources_overlay.pdf   the same, with a title and a legend
#      Same colour scheme, with the VIDEO-derived mask as the reference and the
#      PHOTO-derived one in the "prediction" role - so red means the video saw a
#      vessel the photo did not, and blue the reverse. Neither is a model output;
#      those are roles, not claims about which one is right. Expect blue wherever
#      the sensor never went: the photo sees the whole phantom, the video only
#      the swept region.
#      Also prints the video-vs-photo IoU.
#
#   3. Supporting figures.
#      segmentation_mask_predicted_aggregated.png  predicted vessel mask alone
#      exp_overlay_downscaled.pdf                  the multi-panel comparison:
#                                                  photo-derived ground truth,
#                                                  video-derived ground truth,
#                                                  prediction, confusion overlay
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

RERUN=1
MODEL="meat"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cached)  RERUN=0 ;;
        --model)
            shift
            if [ "$#" -eq 0 ]; then
                echo "ERROR: --model needs 'sim' or 'meat'" >&2; exit 1
            fi
            MODEL="$1" ;;
        --model=*) MODEL="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done
case "${MODEL}" in
    sim|meat) ;;
    *) echo "ERROR: --model must be 'sim' or 'meat', got '${MODEL}'" >&2; exit 1 ;;
esac
export DIFFTACTILE_VESSEL_MAP_MODEL="${MODEL}"

# The sim model's outputs carry a _sim suffix (see predict_exp.py).
SUFFIX=""
[ "${MODEL}" = "sim" ] && SUFFIX="_sim"

# No display -> never block on a window; the figures on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

CACHE="difftactile/output/exp_probs${SUFFIX}.npz"
if [ "${RERUN}" = "0" ] && [ ! -f "${CACHE}" ]; then
    echo "ERROR: --cached given but ${CACHE} does not exist." >&2
    echo "Run without --cached once (with the same --model) to produce it." >&2
    exit 1
fi

if [ "${RERUN}" = "1" ]; then
    echo "Running inference over the silicone clips with the '${MODEL}' model,"
    echo "then building the vessel map."
    export DIFFTACTILE_RERUN_INFERENCE=1
else
    echo "Reusing cached probabilities from ${CACHE}."
fi

python -m difftactile.scripts.script_predict_exp

echo
echo "Written to difftactile/output/:"
echo
echo "  prediction vs ground truth (model: ${MODEL}):"
for f in confusion_overlay_vein_map${SUFFIX}.png \
         confusion_overlay_vein_map${SUFFIX}.pdf; do
    [ -f "difftactile/output/${f}" ] && echo "    ${f}"
done
echo "  ground truth from video vs from top-view photo:"
for f in ground_truth_sources_overlay.png \
         ground_truth_sources_overlay.pdf; do
    [ -f "difftactile/output/${f}" ] && echo "    ${f}"
done
echo "  supporting:"
for f in segmentation_mask_predicted_aggregated${SUFFIX}.png \
         exp_overlay_downscaled${SUFFIX}.pdf; do
    [ -f "difftactile/output/${f}" ] && echo "    ${f}"
done
echo
echo "Colours: green = both say vessel, red = reference says vessel and the other"
echo "does not, blue = the other says vessel and the reference does not, black = neither."
