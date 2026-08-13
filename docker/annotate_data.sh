#!/usr/bin/env bash
#
# Manual annotation and annotation review for the two real-world datasets.
#
# One command covers both jobs, because in each dataset a single tool does both:
# it loads whatever annotations already exist, draws them over the data, and lets
# you step through frames - adding to them where that is meaningful.
#
# RUN THIS ON BARE METAL, NOT IN DOCKER. These are mouse-and-keyboard tools that
# you drive frame by frame, so responsiveness matters more here than for any
# other entrypoint in the project. Inside the container every repaint crosses a
# forwarded X socket, which makes stepping through frames visibly choppy; run
# natively and it is immediate. This is the one script that is expected to run
# outside the container, and it needs no part of the Docker stack - no Taichi,
# no CUDA, no torch.
#
# It uses its own small micromamba environment, created once with:
#
#     micromamba env create -f requirements/annotator-env.yml
#
# The script activates that env itself, so just run it directly:
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

These are interactive tools and must run on a machine with a display. Run this
script from a normal desktop terminal rather than over a plain SSH session.

On a Wayland desktop DISPLAY is normally still set (Xwayland); if it is not,
check that you are in a graphical session.
EOF
    exit 1
fi

# These tools exist to be driven by hand, so opt into blocking windows rather
# than the project default of never waiting on a GUI.
export DIFFTACTILE_INTERACTIVE=1

# --- pick the interpreter -----------------------------------------------------
#
# Prefer the dedicated `vessel-palpation-annotator` micromamba env. It is the
# only environment this script is tested against, and crucially it installs the
# GUI-capable `opencv-python` wheel: several of the project's other envs (and the
# Docker image's batch tooling) deliberately install `opencv-python-headless`,
# which is built with GUI: NONE and cannot open a window at all.
#
# If the caller has already activated something themselves - or is running this
# inside the container, which is supported but slower - we leave their
# interpreter alone rather than second-guessing it.
ENV_NAME="vessel-palpation-annotator"

# Look for the env's interpreter directly rather than requiring `micromamba` on
# PATH: a desktop terminal often has not run the shell hook, and MAMBA_ROOT_PREFIX
# may be unset, but the binary is still sitting in a predictable place.
ENV_PYTHON=""
for root in "${MAMBA_ROOT_PREFIX:-}" "${HOME}/micromamba" "${HOME}/.micromamba" \
            "${CONDA_PREFIX:-}/envs/.." "/opt/conda"; do
    [ -n "${root}" ] || continue
    if [ -x "${root}/envs/${ENV_NAME}/bin/python" ]; then
        ENV_PYTHON="${root}/envs/${ENV_NAME}/bin/python"
        break
    fi
done

if [ -n "${DIFFTACTILE_ANNOTATOR_PYTHON:-}" ]; then
    PYTHON="${DIFFTACTILE_ANNOTATOR_PYTHON}"
elif [ "${CONDA_DEFAULT_ENV:-}" = "${ENV_NAME}" ]; then
    PYTHON="python"
elif [ -n "${ENV_PYTHON}" ]; then
    PYTHON="${ENV_PYTHON}"
elif command -v micromamba >/dev/null 2>&1 \
     && micromamba env list 2>/dev/null | grep -qE "(^|[[:space:]])${ENV_NAME}[[:space:]]"; then
    PYTHON="micromamba run -n ${ENV_NAME} python"
elif [ -f /.dockerenv ]; then
    # Inside the container: the image already carries a GUI-capable OpenCV.
    PYTHON="python"
else
    cat >&2 <<EOF
ERROR: the '${ENV_NAME}' micromamba environment does not exist.

Create it once (takes about a minute, ~500 MB, no torch or CUDA):

    micromamba env create -f requirements/annotator-env.yml

Then re-run this script. To use a different interpreter instead, set
DIFFTACTILE_ANNOTATOR_PYTHON to its path - it needs numpy, scipy, tqdm and a
GUI-capable opencv-python (NOT opencv-python-headless).
EOF
    exit 1
fi

# Frames are 1080p; the viewers scale them down for display so each repaint
# pushes ~4x less data. Override to change the on-screen width, or set 0 to
# disable scaling entirely.
export DIFFTACTILE_VIEW_WIDTH="${DIFFTACTILE_VIEW_WIDTH:-960}"

if [ "${DATASET}" = "meat" ]; then
    echo "Meat annotations: red = vessel present, green = absent."
    echo "Keys: m/n trial, k/j frame, q quit."
    exec ${PYTHON} -m difftactile.scripts.script_browse_meat_annotations
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
exec ${PYTHON} -m difftactile.scripts.script_annotate_silicone
