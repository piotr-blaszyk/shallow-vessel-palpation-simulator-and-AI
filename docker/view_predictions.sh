#!/usr/bin/env bash
#
# Interactive per-frame prediction viewer: step through test-set frames and
# inspect the confusion overlay (green TP, yellow TN, red FP, blue FN) for any of
# the six canonical scenarios.
#
# Runs IN THE DOCKER CONTAINER - it needs the GNN stack (torch, torch-geometric,
# CUDA) to run inference, so unlike the annotation viewers there is no bare-metal
# variant of this one. Structured like docker/annotate_data_docker.sh, and for
# the same reasons: launched from the host, it execs into the running container
# and opens a native Wayland window by default.
#
# Usage (from the HOST):
#   ./docker/view_predictions.sh <config> [--pretrained|--retrained] [--x11]
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
# Options:
#   --x11     Force the X11/Xwayland backend instead of Wayland. Expect this to
#             be choppy (see the note below); it is a fallback for X11-only
#             hosts and an A/B switch for isolating Wayland-specific bugs.
#   --shell   Print the docker exec command instead of running it.
#
# It can also be run from INSIDE the container (e.g. after docker-connect.sh),
# in which case it skips the exec and launches the viewer directly.
#
# DISPLAY. One window, Qt 6 (PySide6), stepped by hand - which is three changes
# from the OpenCV version this replaces:
#
#   * Qt instead of OpenCV, so the window is a native Wayland client. The
#     opencv-python wheel ships only the `xcb` platform plugin, so every cv2
#     window on a Wayland desktop is an Xwayland client; PySide6 bundles the
#     Wayland plugins. This reuses the same difftactile/main/qt_viewer.py the
#     annotation viewers use.
#   * One window instead of five. The five panels (Ground Truth, Hard
#     Prediction, Confusion Matrix, Soft Prediction, Metadata) are composited
#     into a single image and tiled by us. They used to be five separate cv2
#     windows placed with cv2.moveWindow(), which does nothing at all under
#     Wayland - a Wayland client cannot position itself, by design - so they
#     landed wherever the compositor chose, on top of each other.
#   * Stepped, not auto-played. The old loop advanced on timed wait_key()
#     delays, so the frame you wanted had usually gone before you registered
#     it. Now nothing moves unless you press a key.
#
# Keys, one per action - three nested levels of navigation:
#   i / o   previous / next trial
#   j / k   previous / next clip, within the trial
#   n / m   previous / next frame, within the clip
#   q       quit
# Changing trial or clip lands on that unit's first frame, and all moves clamp
# at the ends rather than wrapping.
#
# As with the annotators, the Wayland path is smooth and the --x11 path is
# visibly choppy, because every repaint takes an extra copy through Xwayland.
# That is Xwayland's cost, not Docker's, and is not something this script can
# fix - prefer the default.
#
# Examples:
#   ./docker/view_predictions.sh A-to-B
#   ./docker/view_predictions.sh A-to-C --retrained
#   ./docker/view_predictions.sh A-to-B --x11
#
set -euo pipefail

CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

CONFIG=""
WEIGHTS="--pretrained"
BACKEND="wayland"
PRINT_ONLY=0

for arg in "$@"; do
    case "${arg}" in
        A-to-B|C-to-B|A-to-C) CONFIG="${arg}" ;;
        --pretrained|--retrained) WEIGHTS="${arg}" ;;
        --x11) BACKEND="x11" ;;
        --shell) PRINT_ONLY=1 ;;
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

# --- Stage 1: on the host, hand the job to the container ----------------------
#
# Re-executes this same script inside the container so the argument parsing and
# the viewer launch live in one place. /.dockerenv stops that from recursing.
if [ ! -f /.dockerenv ]; then
    if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
        cat >&2 <<EOF
ERROR: container '${CONTAINER_NAME}' is not running.

Start it first:

    ./docker/docker-run.sh
EOF
        exit 1
    fi

    if [ "${BACKEND}" = "wayland" ]; then
        if ! docker exec "${CONTAINER_NAME}" bash -lc \
                '[ -S "${XDG_RUNTIME_DIR:-/nonexistent}/${WAYLAND_DISPLAY:-nonexistent}" ]' \
                >/dev/null 2>&1; then
            cat >&2 <<EOF
ERROR: no Wayland socket inside container '${CONTAINER_NAME}'.

Either the host session is not Wayland, or the container was started before
docker-run.sh gained the socket mount. A container keeps the mounts it was
created with, so restart it:

    docker stop ${CONTAINER_NAME} && ./docker/docker-run.sh

Or run this script with --x11 to use the X11/Xwayland path instead.
EOF
            exit 1
        fi
    fi

    # -it only when there is a terminal on both ends; `docker exec -i` fails
    # outright when stdin is a pipe. The viewer takes its input from the Qt
    # window, so a missing TTY costs only terminal echo of the banners.
    TTY_ARGS=()
    [ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)
    CMD=(docker exec "${TTY_ARGS[@]}" "${CONTAINER_NAME}"
         ./docker/view_predictions.sh "${CONFIG}" "${WEIGHTS}")
    [ "${BACKEND}" = "x11" ] && CMD+=(--x11)

    if [ "${PRINT_ONLY}" -eq 1 ]; then
        printf '%q ' "${CMD[@]}"; echo
        exit 0
    fi

    echo "Running the prediction viewer inside '${CONTAINER_NAME}' (backend: ${BACKEND})."
    exec "${CMD[@]}"
fi

# --- Stage 2: inside the container --------------------------------------------
cd "${REPO_DIR}"

# This viewer exists to be watched, so opt into blocking windows rather than the
# project default of never waiting on a GUI.
export DIFFTACTILE_INTERACTIVE=1

if [ "${BACKEND}" = "wayland" ]; then
    if [ -z "${WAYLAND_DISPLAY:-}" ] || [ ! -S "${XDG_RUNTIME_DIR:-/nonexistent}/${WAYLAND_DISPLAY}" ]; then
        echo "ERROR: no Wayland socket at \${XDG_RUNTIME_DIR}/\${WAYLAND_DISPLAY}." >&2
        echo "Restart the container with ./docker/docker-run.sh, or pass --x11." >&2
        exit 1
    fi
    # Unsetting DISPLAY is load-bearing, not belt-and-braces: docker-run.sh sets
    # it container-wide for the project's X11-only tools, and a stray DISPLAY is
    # the classic way a Qt app silently picks xcb. No `;xcb` fallback either, so
    # a misconfiguration is an error rather than a silent downgrade.
    unset DISPLAY
    export QT_QPA_PLATFORM=wayland
else
    if [ -z "${DISPLAY:-}" ]; then
        echo "ERROR: --x11 given but DISPLAY is unset inside the container." >&2
        exit 1
    fi
    unset WAYLAND_DISPLAY
    export QT_QPA_PLATFORM=xcb
fi

if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

echo "Qt platform: ${QT_QPA_PLATFORM}  (DISPLAY=${DISPLAY:-<unset>}, WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>})"
echo "Viewing predictions: ${CONFIG} ${WEIGHTS}"
echo "Keys: i/o trial, j/k clip, n/m frame, q quit."
exec python -m difftactile.scripts.script_visualise "${CONFIG}" "${WEIGHTS}"
