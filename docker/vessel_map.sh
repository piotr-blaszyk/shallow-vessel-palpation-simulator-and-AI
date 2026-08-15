#!/usr/bin/env bash
#
# Bird's-eye vessel localisation map for ONE configuration.
#
# Lifts each per-marker prediction to 3D through the fisheye model, moves it
# into the world frame with the sensor pose from robot kinematics (or the
# simulator's own pose for the Sim dataset), and drops it onto a top-view grid
# of the phantom at 1 mm per pixel (the 2D->3D->2D projection). The map is
# thresholded, compared with a ground-truth map on the same grid, and scored
# pixel by pixel - at the native resolution and with the ground-truth region
# grown by an L2 disc of 0, 1 and 2 mm.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# Figures are written to disk, so a display is optional.
#
# Usage:
#   ./docker/vessel_map.sh <config> [--ground-truth video|photo] [--model best|legacy]
#                                   [--threshold T]
#
# Configurations (A = simulation, B = real silicone, C = real meat):
#   A-to-A    Sim -> Sim          one simulated slide (needs vessel_map_sim_trajectory.sh once)
#   A-to-B    Sim -> Silicone     the ten silicone sweeps
#   C-to-B    Meat -> Silicone    same sweeps, meat-trained model
#   A-to-C    Sim -> Meat         one map per meat trial (ten)
#
# Options:
#   --ground-truth S  Where the true vessel pixels come from.
#                       video  (default) the test data's per-marker labels
#                              reprojected exactly like the predictions:
#                              manual video annotation (Silicone), labels from
#                              robot kinematics (Meat), the simulator's own vein
#                              projection (Sim; written as "simulator").
#                       photo  the silicone phantom's top-view photograph,
#                              segmented and block-downsampled onto the map grid
#                              and restricted to the swept region. Silicone only.
#   --model M         best    (default) the best-of-5-seeds instance of this
#                             configuration from the published sweep
#                             (files.published_sweep; highest AP, see
#                             cnn/model_selection.py)
#                     legacy  the pre-2026-08-15 checkpoint that produced the
#                             accepted manuscript's Fig. 8 / Table 4
#                             (saved_models_legacy/README.md). A-to-B and C-to-B
#                             only. Sets DIFFTACTILE_CLIP_LEN=7 for you, since
#                             those weights were trained with a 7-frame window.
#   --threshold T     Use decision threshold T instead of the rule below.
#
# THE DECISION THRESHOLD is chosen per run, never assumed: the smallest cut at
# which the map's pixel-level PRECISION is >= 0.9, i.e. the one that maximises
# RECALL under that constraint, pooled over the run's maps, with a predicted
# pixel counted as correct within 3 mm of a true pixel (the reprojected truth is
# sparse marker points, so at 0 mm the rule is met by an empty map). No 0.5 /
# 0.58 anywhere. If 0.9 is out of reach the run falls back to the F1-optimal
# threshold and SAYS SO in report.md / run.json - read that line first.
#
# OUTPUT goes to a versioned directory that this script picks itself:
#
#   difftactile/output/vessel_maps/<train>-to-<test>_gt-<source>/<timestamp>/
#
# e.g. vessel_maps/sim-to-silicone_gt-video/20260815-153000/. Legacy-model runs
# get a "-legacy" suffix on the timestamp. Inside (per map, in a subfolder when
# a run has several maps, i.e. the meat trials):
#
#   prediction.png            predicted vessel pixels (white on black)
#   ground_truth.png          true vessel pixels
#   confusion_rNN.png/.pdf    confusion overlay, truth grown by NN mm (00, 01, 02)
#   l2_distances_rNN.png      decile histogram of predicted-to-nearest-true distance
#   metrics_by_radius.md      TP FP FN TN MCC F1 precision recall accuracy, per radius
#   *_big.png                 5x nearest-neighbour twins, for documents
#   report.md, run.json       run-level summary (model, threshold, per-map/pooled)
#   threshold_selection.png   precision & recall vs threshold, chosen point marked
#
# COLOURS (Visualisation.CONFUSION_COLOURS_RGB): green = both say vessel,
# RED = truth says vessel and the map does not (a miss), BLUE = the map says
# vessel and the truth does not (a false alarm), black = neither.
#
# Examples:
#   ./docker/vessel_map.sh A-to-B
#   ./docker/vessel_map.sh A-to-B --ground-truth photo
#   ./docker/vessel_map.sh C-to-B --model legacy
#   ./docker/vessel_map.sh A-to-C --threshold 0.6
#   ./docker/vessel_map_all.sh                       # every configuration
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

CONFIG=""
GT=""
MODEL="best"
THRESHOLD=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        A-to-A|A-to-B|C-to-B|A-to-C) CONFIG="$1" ;;
        --ground-truth) shift; [ "$#" -gt 0 ] || { echo "ERROR: --ground-truth needs video|photo" >&2; exit 1; }; GT="$1" ;;
        --ground-truth=*) GT="${1#*=}" ;;
        --model) shift; [ "$#" -gt 0 ] || { echo "ERROR: --model needs best|legacy" >&2; exit 1; }; MODEL="$1" ;;
        --model=*) MODEL="${1#*=}" ;;
        --threshold) shift; [ "$#" -gt 0 ] || { echo "ERROR: --threshold needs a value" >&2; exit 1; }; THRESHOLD="$1" ;;
        --threshold=*) THRESHOLD="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done
[ -n "${CONFIG}" ] || { echo "ERROR: no configuration given." >&2; echo; usage; exit 1; }
case "${GT}" in ""|video|photo|simulator) ;; *) echo "ERROR: --ground-truth must be video or photo, got '${GT}'" >&2; exit 1 ;; esac
case "${MODEL}" in best|legacy) ;; *) echo "ERROR: --model must be best or legacy, got '${MODEL}'" >&2; exit 1 ;; esac
# "video" is the reprojected-labels source; for the Sim dataset those labels
# come from the simulator, and the folder is named accordingly.
if [ "${CONFIG}" = "A-to-A" ] && { [ -z "${GT}" ] || [ "${GT}" = "video" ]; }; then GT="simulator"; fi

export DIFFTACTILE_MAP_CONFIG="${CONFIG}"
export DIFFTACTILE_MAP_MODEL="${MODEL}"
[ -n "${GT}" ] && export DIFFTACTILE_MAP_GT="${GT}"
[ -n "${THRESHOLD}" ] && export DIFFTACTILE_VESSEL_MAP_THRESHOLD="${THRESHOLD}"
# The legacy checkpoints were trained with a 7-frame window and their input
# layer is sized to it; the current default is 5.
[ "${MODEL}" = "legacy" ] && export DIFFTACTILE_CLIP_LEN=7

# No display -> never block on a window; the files on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

echo "Bird's-eye vessel map: ${CONFIG}, ground truth from ${GT:-video}, model ${MODEL}${THRESHOLD:+, threshold ${THRESHOLD}}"
python -m difftactile.scripts.script_vessel_map
