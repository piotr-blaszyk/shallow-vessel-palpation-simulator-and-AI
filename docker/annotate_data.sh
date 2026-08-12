#!/usr/bin/env bash
#
# Manual annotation and annotation review for the two real-world datasets.
#
# One command covers both jobs, because in each dataset a single tool does both:
# it loads whatever annotations already exist, draws them over the data, and lets
# you step through frames - adding to them where that is meaningful.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# It needs a display: these are mouse-and-keyboard tools.
#
# Usage:
#   ./docker/annotate_data.sh --silicone [--source DIR]
#   ./docker/annotate_data.sh --meat
#
# Datasets:
#   --silicone   ViTacTip videos of the silicone vascular phantom. Click to place
#                up to 4 vessel points per frame; existing points are redrawn.
#                Produces the paper's annotated-frame figure (large red/green
#                circles = clicked points).
#                Editing is LOCKED until you press g, so browsing cannot alter
#                annotations by accident. z undoes the last point added in this
#                session only - it never touches points loaded from disk.
#                Keys: g toggle editing, left click add, z undo, d clear frame,
#                      m/n video, k/j frame, p save, q save and quit,
#                      x quit WITHOUT saving (press twice to confirm).
#                Nothing is written until p or q.
#
#   --meat       Meat-phantom trials: ground-truth vessel labels per marker
#                (red = vessel present, green = absent). Review only - meat
#                labels are derived analytically from robot kinematics and the
#                straw geometry, not by clicking.
#                Keys: m/n trial, k/j frame, q quit.
#
# Options:
#   --source DIR  Where to stage the silicone videos and annotations from, when
#                 they are not already in the working tree (see below).
#
# NOTE ON DATA. The published Zenodo bundle deliberately ships only the final
# processed datasets, not the raw/intermediate videos (data/MANIFEST.md), so:
#   * --meat works straight from the bundle: it renders the marker labels in
#     clean/<trial>/marker_{positions,labels}.npz, with no video needed.
#   * --silicone needs the dilated videos and their annotation pickles, which are
#     NOT in the bundle. If they are missing this script looks for them in the
#     author's local tree and stages copies into place; override with --source.
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

DATASET=""
# Default staging source: the author's frozen pre-rename tree, where the silicone
# videos and annotations live under the old `endgame/` directory name.
SOURCE_DIR="${DIFFTACTILE_ANNOTATION_SOURCE:-/home/psb120/Documents/phd/data/masters/diff-tactile-fork-IROS-submission-state/diff-tactile-fork/difftactile/manual_or_experimental_data/endgame}"

while [ $# -gt 0 ]; do
    case "$1" in
        --silicone) DATASET="silicone" ;;
        --meat)     DATASET="meat" ;;
        --source)   SOURCE_DIR="${2:?--source needs a directory}"; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done

if [ -z "${DATASET}" ]; then
    echo "ERROR: pick a dataset: --silicone or --meat" >&2
    echo
    usage
    exit 1
fi

if [ -z "${DISPLAY:-}" ]; then
    cat >&2 <<'EOF'
ERROR: DISPLAY is not set, so no windows can be opened.

These are interactive tools. Start the container with X forwarding
(docker/docker-run.sh passes the host display through) and allow it on the host:

    xhost +local:docker
EOF
    exit 1
fi

# These tools exist to be driven by hand, so opt into blocking windows rather
# than the project default of never waiting on a GUI.
export DIFFTACTILE_INTERACTIVE=1

if [ "${DATASET}" = "meat" ]; then
    echo "Meat annotations: red = vessel present, green = absent."
    echo "Keys: m/n trial, k/j frame, q quit."
    exec python -m difftactile.scripts.script_browse_meat_annotations
fi

# --- silicone: stage the videos and annotations if the bundle did not carry them
SIL_ROOT="difftactile/manual_or_experimental_data/silicone_training_data"
DILATED="${SIL_ROOT}/20250901-131547_dilated"
ANNOTATIONS="${SIL_ROOT}/20250901-131547_annotations"

if [ ! -d "${DILATED}" ] || [ -z "$(ls -A "${DILATED}" 2>/dev/null)" ]; then
    echo "Silicone videos not present in the working tree."
    if [ -d "${SOURCE_DIR}/20250901-131547_dilated" ]; then
        echo "Staging from: ${SOURCE_DIR}"
        mkdir -p "${DILATED}" "${ANNOTATIONS}"
        cp -n "${SOURCE_DIR}/20250901-131547_dilated/." "${DILATED}/" -r
        if [ -d "${SOURCE_DIR}/20250901-131547_annotations" ]; then
            cp -n "${SOURCE_DIR}/20250901-131547_annotations/." "${ANNOTATIONS}/" -r
        fi
        echo "Staged $(ls "${DILATED}"/*.avi 2>/dev/null | wc -l) videos."
    else
        cat >&2 <<EOF
ERROR: no silicone videos, and none found to stage from.

Looked in: ${SOURCE_DIR}/20250901-131547_dilated

The dilated videos and their annotation pickles are excluded from the published
Zenodo bundle (data/MANIFEST.md) because they are intermediate preprocessing
artifacts. Point --source at a tree that has them:

    ./docker/annotate_data.sh --silicone --source /path/to/manual_or_experimental_data/endgame

Note that inside the container the host path must be mounted to be visible.
EOF
        exit 1
    fi
fi

echo "Silicone annotator. Existing annotations are loaded and redrawn."
echo "Editing starts LOCKED - press g to allow clicks (lasts the whole session)."
echo "Keys: g toggle editing, left click add (max 4/frame), z undo this session's"
echo "      last point, d clear frame, m/n video, k/j frame, p save, q save+quit,"
echo "      x quit WITHOUT saving (press x twice to confirm)."
exec python -m difftactile.scripts.script_annotate_silicone
