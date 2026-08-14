#!/usr/bin/env bash
#
# Manuscript alignment panels: simulated vs real markers, on a WHITE background.
#
# Draws the four panels of
#
#   "Alignment between simulated (red) and real (green) marker positions after
#    domain adaptation for four canonical interactions: (a) press, (b) twist
#    about the z-axis, (c) twist about the x-axis, and (d) slide."
#
# Each corresponding (red, green) pair is joined by a blue segment, so the
# direction and size of the misalignment is readable per marker rather than
# inferred from two overlapping clouds.
#
# NOT THE SAME AS THE da_overlay_*.png FIGURES that domain_adaptation.sh writes.
# Those draw the same two point sets over the sensor PHOTOGRAPH, which answers
# "where are the markers on the real sensor". These are for print at small size,
# where the photograph is texture competing with the data.
#
# Runs the four trajectories ONCE each at whatever is in system-params.json -
# there is no optimisation here.
#
# CACHING. Each trajectory costs minutes of simulation, while the figures are
# pure styling over two point sets, so the marker positions are cached as
# `markers_<name>.npz` and reused. With --source pointing at a directory that
# already has them (a published BO run, say), nothing is simulated at all and
# the figures appear in seconds.
#
# Usage:
#   ./alignment_figures.sh                 use the published BO run's cache if
#                                          complete, else simulate
#   ./alignment_figures.sh --source DIR    take cached markers from DIR
#   ./alignment_figures.sh --force         re-simulate even if a cache exists
#   ./alignment_figures.sh --out DIR       write the figures to DIR
#
# Output, per interaction and as a 2x2 sheet:
#   alignment_press.png  alignment_twist_z.png
#   alignment_twist_x.png  alignment_slide.png
#   alignment_all.png
#
# Run this INSIDE the container when it needs to simulate (Taichi + GPU). With a
# complete cache it only draws, so it will run anywhere the ML env is available.
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

# Default to the published BO run: those markers were produced at the
# configuration now in system-params.json, so the figures match the adopted
# parameters without re-simulating them.
export DIFFTACTILE_ALIGNMENT_SOURCE="${DIFFTACTILE_ALIGNMENT_SOURCE:-${REPO_DIR}/difftactile/output/domain_adaptation_published/joint_bo}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --source) shift; export DIFFTACTILE_ALIGNMENT_SOURCE="$1" ;;
        --out) shift; export DIFFTACTILE_ALIGNMENT_OUT="$1" ;;
        --force) export DIFFTACTILE_ALIGNMENT_FORCE=1 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done

# Figures are written to disk; a window is never needed.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

echo "Alignment panels: simulated markers RED, real markers GREEN,"
echo "blue segments joining corresponding pairs, on white."
python -m difftactile.scripts.script_alignment_figures
