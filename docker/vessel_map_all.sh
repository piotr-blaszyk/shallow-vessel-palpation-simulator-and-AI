#!/usr/bin/env bash
#
# Bird's-eye vessel maps for EVERY available configuration, in one go.
#
# Runs ./docker/vessel_map.sh once per (configuration, ground-truth source):
#
#   A-to-A  Sim -> Sim           ground truth: simulator
#   A-to-B  Sim -> Silicone      ground truth: video, then photo
#   C-to-B  Meat -> Silicone     ground truth: video, then photo
#   A-to-C  Sim -> Meat          ground truth: video
#
# Each run lands in its own versioned folder,
#   difftactile/output/vessel_maps/<train>-to-<test>_gt-<source>/<timestamp>/
# so nothing is overwritten and the six runs of one invocation are told apart
# by their timestamps. If the Sim->Sim trajectory has not been simulated yet
# (difftactile/output/vessel_map_sim/raw_reordered_dense/), it is simulated
# first via ./docker/vessel_map_sim_trajectory.sh.
#
# Run INSIDE the container. Options are passed through to every run:
#   --threshold T     one decision threshold for all runs, instead of the
#                     per-run "precision >= 0.9, maximise recall" rule
#   --model legacy    the pre-2026-08-15 checkpoints; only A-to-B and C-to-B
#                     have legacy models, so only those four runs are made
#
# Usage:
#   ./docker/vessel_map_all.sh [--threshold T] [--model best|legacy]
#
set -euo pipefail
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

PASS=()
MODEL="best"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --threshold) shift; [ "$#" -gt 0 ] || { echo "ERROR: --threshold needs a value" >&2; exit 1; }; PASS+=(--threshold "$1") ;;
        --threshold=*) PASS+=(--threshold "${1#*=}") ;;
        --model) shift; [ "$#" -gt 0 ] || { echo "ERROR: --model needs best|legacy" >&2; exit 1; }; MODEL="$1" ;;
        --model=*) MODEL="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done
PASS+=(--model "${MODEL}")

if [ "${MODEL}" = "legacy" ]; then
    RUNS=("A-to-B video" "A-to-B photo" "C-to-B video" "C-to-B photo")
else
    RUNS=("A-to-A simulator" "A-to-B video" "A-to-B photo" "C-to-B video" "C-to-B photo" "A-to-C video")
    if [ -z "$(ls difftactile/output/vessel_map_sim/raw_reordered_dense/*.npz 2>/dev/null)" ]; then
        echo "No simulated Sim->Sim trajectory yet - simulating one first."
        ./docker/vessel_map_sim_trajectory.sh
    fi
fi

for run in "${RUNS[@]}"; do
    set -- ${run}
    echo
    echo "==================== $1, ground truth from $2 ===================="
    ./docker/vessel_map.sh "$1" --ground-truth "$2" "${PASS[@]}"
done

echo
echo "All runs written under difftactile/output/vessel_maps/ (one folder per configuration"
echo "and ground-truth source, one timestamped subfolder per run)."
