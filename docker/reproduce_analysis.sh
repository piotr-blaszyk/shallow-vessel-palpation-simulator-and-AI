#!/usr/bin/env bash
#
# Regenerate EVERY number and figure of the poster's Methods and Results blocks,
# in dependency order, from the published checkpoints and the restored data bundle.
#
# Nothing here trains a model or runs the simulator: the four published
# checkpoints are evaluated, the published top-view maps are read off disk, and
# everything else is derived from those. A full run is ~20-30 minutes, of which
# the two overlay stages (--with-overlays, off by default) are the bulk.
#
# Stages, in the order they must run:
#
#   1  frame_space_predictions.py    per-marker probabilities of all four models
#                                    -> analysis/results/frame_space_predictions_<cfg>.npz
#                                    ASSERTS the pooled TP/FP/FN/TN against
#                                    FRAME_SPACE_METRICS.md, so a wrong checkpoint
#                                    or a wrong test split fails here, not silently.
#   2  symmetric_distances.py        nearest-neighbour distances + foreground IoU,
#                                    video-frame and top-view space
#   3  comparable_metrics.py         our predictions under each baseline's own metric
#   4  detection_ospa.py             counts, Hungarian matching, F1 and OSPA
#   5  detection_ospa_report.py      the Markdown write-up of stage 4
#   6  segmentation_bars.py          foreground-IoU and distance bar figures
#   7  detection_bars.py             detection/OSPA bars, count confusion, 2-class confusion
#   8  pr_curves_with_ap.py          precision-recall panels + shared legend and x label
#   9  gnn_graph_figure.py           3D render of the ST-GNN input graph
#  10  workflow_centreline_panels.py the workflow diagram's centreline/centroid panels
#  11  select_sim_to_meat_frames.py  per Sim->Meat trial, the frame whose TP/FP/FN/TN mix
#                                    is closest to the pooled one (a typical frame)
#  12  make_workflow_images.py       the remaining workflow diagram blocks (needs docs/videos/
#                                    and the published vessel-map runs)
#
#  optional (--with-overlays), diagnostics rather than published figures:
#  13  centreline_overlays.py        one per-vessel overlay per scored data point
#  14  detection_overlays.py         the same for the detection/OSPA extraction
#
# Everything is written under analysis/ (results/, figures/, reports/, overlays/);
# nothing outside that tree is touched.
#
# Run INSIDE the container, from the repository root or from docker/:
#   ./docker/reproduce_analysis.sh [--with-overlays] [--from N] [--only N[,N...]]
#
# Options:
#   --with-overlays   also run stages 13-14 (~15 min, ~4000 images, ~200 MB)
#   --from N          start at stage N (its inputs must already exist)
#   --only N[,N...]   run only these stages
#   --list            print the stage table and exit
#
set -euo pipefail
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'; }

PY="${DIFFTACTILE_PYTHON:-python}"
S="analysis/scripts"

# stage number -> "label|command"
STAGES=(
    "1|per-marker predictions of the four published models|${PY} ${S}/frame_space_predictions.py"
    "2|nearest-neighbour distances and foreground IoU|${PY} ${S}/symmetric_distances.py"
    "3|our predictions under each baseline's own metric|${PY} ${S}/comparable_metrics.py"
    "4|detection counts, matching, F1 and OSPA|${PY} ${S}/detection_ospa.py"
    "5|OSPA report|${PY} ${S}/detection_ospa_report.py"
    "6|segmentation bar figures (IoU, distance)|${PY} ${S}/segmentation_bars.py"
    "7|detection bars, count confusion, 2-class confusion|${PY} ${S}/detection_bars.py"
    "8|precision-recall panels|${PY} ${S}/pr_curves_with_ap.py"
    "9|ST-GNN input-graph render|${PY} ${S}/gnn_graph_figure.py"
    "10|workflow centreline/centroid panels|${PY} ${S}/workflow_centreline_panels.py"
    "11|representative Sim->Meat frame per trial|${PY} ${S}/select_sim_to_meat_frames.py"
    "12|workflow diagram image blocks|${PY} ${S}/make_workflow_images.py"
    "13|per-vessel centreline overlays (diagnostic)|${PY} ${S}/centreline_overlays.py"
    "14|detection/OSPA overlays (diagnostic)|${PY} ${S}/detection_overlays.py"
)
LAST_DEFAULT=12          # stages 13-14 only with --with-overlays

FROM=1
ONLY=""
WITH_OVERLAYS=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-overlays) WITH_OVERLAYS=1 ;;
        --from) shift; FROM="${1:?--from needs a stage number}" ;;
        --from=*) FROM="${1#*=}" ;;
        --only) shift; ONLY="${1:?--only needs a stage list}" ;;
        --only=*) ONLY="${1#*=}" ;;
        --list)
            for st in "${STAGES[@]}"; do printf '%3s  %s\n' "${st%%|*}" "$(echo "${st}" | cut -d'|' -f2)"; done
            exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done

selected() {   # $1 = stage number
    if [ -n "${ONLY}" ]; then
        case ",${ONLY}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
    fi
    [ "$1" -ge "${FROM}" ] || return 1
    [ "$1" -le "${LAST_DEFAULT}" ] || [ "${WITH_OVERLAYS}" = "1" ] || return 1
    return 0
}

mkdir -p analysis/results analysis/figures analysis/reports
START=$(date +%s)
for st in "${STAGES[@]}"; do
    n="${st%%|*}"
    label="$(echo "${st}" | cut -d'|' -f2)"
    cmd="$(echo "${st}" | cut -d'|' -f3-)"
    selected "${n}" || continue
    echo
    echo "==================== [${n}/${#STAGES[@]}] ${label} ===================="
    echo "+ ${cmd}"
    ( set -x; eval "${cmd}" )
done

echo
echo "Done in $(( $(date +%s) - START ))s."
echo "Results   analysis/results/    (JSON + .npz: every number the poster prints)"
echo "Figures   analysis/figures/    (PDF/PNG: every figure the poster prints)"
echo "Reports   analysis/reports/    (Markdown write-ups)"
echo "See analysis/README.md for which artefact belongs to which poster block."
