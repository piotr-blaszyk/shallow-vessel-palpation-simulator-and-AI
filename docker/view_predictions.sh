#!/usr/bin/env bash
#
# Interactive per-frame prediction viewer: step through test-set frames and
# inspect the confusion overlay (green = both say vessel, red = missed vessel,
# blue = false alarm, grey = neither - the project-wide scheme) for any of the
# four configurations.
#
# Runs IN THE DOCKER CONTAINER - it needs the GNN stack (torch, torch-geometric,
# CUDA) to run inference, so unlike the annotation viewers there is no bare-metal
# variant of this one. Structured like docker/annotate_data_docker.sh, and for
# the same reasons: launched from the host, it execs into the running container
# and opens a native Wayland window by default.
#
# Usage (from the HOST):
#   ./docker/view_predictions.sh <config> [--central|--all] [--pretrained|--retrained] [--x11]
#
# Configurations (A = simulation, B = real silicone, C = real meat):
#   A-to-A    model trained on simulation, predictions shown on held-out simulation
#   A-to-B    model trained on simulation, predictions shown on silicone
#   C-to-B    model trained on meat,       predictions shown on silicone
#   A-to-C    model trained on simulation, predictions shown on meat
#
# Weights:
#   --best        (default) the best-of-5-seeds instance of this configuration
#                 from the published sweep (highest AP; cnn/model_selection.py) -
#                 the project convention wherever ONE model is shown
#   --pretrained  the checkpoint at the published path (saved_models_*/), which
#                 is that same best-of-5 instance for A-to-B / C-to-B
#   --retrained   the checkpoint written by a local `--train` run of the same
#                 configuration, i.e. saved_models_*/*_retrained_<config>.pt
#   --legacy      the pre-2026-08-15 checkpoint (saved_models_legacy/), A-to-B
#                 and C-to-B only; sets DIFFTACTILE_CLIP_LEN=7 for you
#   --sweep TS [--seed N]
#                 one seed's model out of a seed sweep, from
#                 saved_models_sweeps/TS/ (TS is the sweep's timestamp, or a full
#                 path). --seed picks which, default 0. This is the only way to
#                 say WHICH trained model to view once a sweep has produced
#                 several; --retrained alone means "whatever was trained last".
#                 The checkpoint and its normalisation statistics are taken from
#                 the same per-seed directory, which is what keeps them matched.
#
# ONE MODEL, NOT AN AVERAGE. This viewer deliberately shows a single model even
# when a sweep has trained many. A mean prediction over N seeds is an ENSEMBLE -
# a different model, whose AUROC/AP are not the numbers any table in this project
# reports - so showing it here would make the figures disagree with the results.
# Ensembling is a research direction that would need its own entrypoint and its
# own reported numbers, not a display toggle. To compare seeds, open two viewers
# with different --seed values, or read the spread in AUROC_RESULTS.md.
#
# The configuration selects BOTH the model weights and the test dataset, so the
# six combinations above are the six scenarios.
#
# Which frames to show. --central is the DEFAULT, because it is what the paper
# reports; pass --all only when you specifically want to inspect the rest.
#
#   --central  (default) Only each window's CENTRAL frame, navigated over two
#              levels (trial / central frame). This is the prediction that is
#              actually reported and scored: the central frame is the one with
#              temporal context on both sides, and it is the only one the
#              val/test metrics look at (dataset.py::get_mask() masks to
#              clip_len // 2, and segmentation_gnn.shared_step() applies it).
#              Consecutive sliding windows have consecutive centres, so j/k
#              walks the trial frame by frame.
#
#              Its one cost: the first and last clip_len // 2 frames of each
#              trial are never any window's centre, so they have no prediction
#              to show and are skipped. That is inherent to central-frame
#              reporting, not a viewer limitation.
#
#   --all      Every frame of every sliding window, navigated over three levels
#              (trial / clip / frame). The model takes a clip_len-frame window
#              and predicts a label for every frame in it, so this shows
#              everything it emits - including the off-centre predictions that
#              training learns from but reporting ignores. A debugging view.
#
# The default is deliberately the reported one: if the two ever disagree about
# how good the model looks, the reported view is the one that should be seen
# first. The script prints which mode it is in either way.
#
# Options:
#   --trials SPEC
#             Show only some trials of the test set: a comma-separated list of
#             trial-id substrings (a meat trial directory name, or a simulated /
#             silicone file stem such as trajectory_0426), or the token
#             first-vessel-present. The i/o keys then step over just those.
#             Mainly for the simulated test set, whose 75 held-out trajectories
#             are ~23k sliding windows - the README recording shows one.
#   --record PATH
#             Record instead of opening a window: the viewer is stepped through
#             every trial and frame automatically (one key press per 500 ms of
#             video) and written to PATH as .mp4, rendered offscreen so nothing
#             appears on screen. Used by docker/record_videos.sh.
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
# Keys, one per action. --all has three nested levels of navigation:
#   i / o   previous / next trial
#   j / k   previous / next clip, within the trial
#   n / m   previous / next frame, within the clip
#   q       quit
#
# --central has two, since showing one frame per window collapses the innermost
# level away:
#   i / o   previous / next trial
#   j / k   previous / next central frame, within the trial
#   q       quit
#
# Changing trial (or clip) lands on that unit's first frame, and all moves clamp
# at the ends rather than wrapping.
#
# As with the annotators, the Wayland path is smooth and the --x11 path is
# visibly choppy, because every repaint takes an extra copy through Xwayland.
# That is Xwayland's cost, not Docker's, and is not something this script can
# fix - prefer the default.
#
# Examples:
#   ./docker/view_predictions.sh A-to-B                # central frames (default)
#   ./docker/view_predictions.sh A-to-B --all          # every frame of every window
#   ./docker/view_predictions.sh A-to-C --retrained
#   ./docker/view_predictions.sh A-to-B --all --x11
#
set -euo pipefail

CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

CONFIG=""
WEIGHTS="--best"
# Defaults to the frames the paper actually reports; --all is the opt-in
# debugging view (see the header).
FRAMES="--central"
FRAMES_GIVEN=""
BACKEND="wayland"
PRINT_ONLY=0
SWEEP_DIR=""
SWEEP_SEED="0"
TRIALS=""
RECORD=""

# A while loop rather than `for arg`, because --sweep and --seed consume the
# argument that follows them.
while [ "$#" -gt 0 ]; do
    case "$1" in
        A-to-A|A-to-B|C-to-B|A-to-C) CONFIG="$1" ;;
        --pretrained|--retrained|--best|--legacy) WEIGHTS="$1" ;;
        --sweep)
            shift
            [ "$#" -gt 0 ] || { echo "ERROR: --sweep needs a sweep timestamp or path." >&2; exit 1; }
            SWEEP_DIR="$1"
            ;;
        --sweep=*) SWEEP_DIR="${1#*=}" ;;
        --seed)
            shift
            [ "$#" -gt 0 ] || { echo "ERROR: --seed needs a seed number." >&2; exit 1; }
            SWEEP_SEED="$1"
            ;;
        --seed=*) SWEEP_SEED="${1#*=}" ;;
        --trials)
            shift
            [ "$#" -gt 0 ] || { echo "ERROR: --trials needs a selection." >&2; exit 1; }
            TRIALS="$1"
            ;;
        --trials=*) TRIALS="${1#*=}" ;;
        --record)
            shift
            [ "$#" -gt 0 ] || { echo "ERROR: --record needs an output .mp4 path." >&2; exit 1; }
            RECORD="$1"
            ;;
        --record=*) RECORD="${1#*=}" ;;
        --all|--central)
            # Asking for both is contradictory, so reject it rather than letting
            # the last one silently win. Tracked separately from FRAMES, which
            # already holds the default.
            if [ -n "${FRAMES_GIVEN}" ] && [ "${FRAMES_GIVEN}" != "$1" ]; then
                echo "ERROR: --all and --central are mutually exclusive; pass exactly one." >&2
                exit 1
            fi
            FRAMES_GIVEN="$1"
            FRAMES="$1"
            ;;
        --x11) BACKEND="x11" ;;
        --shell) PRINT_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done

if [ -n "${SWEEP_DIR}" ]; then
    case "${SWEEP_SEED}" in
        ''|*[!0-9]*) echo "ERROR: --seed needs a non-negative integer, got '${SWEEP_SEED}'" >&2; exit 1 ;;
    esac
fi

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

    if [ "${BACKEND}" = "wayland" ] && [ -z "${RECORD}" ]; then
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
         ./docker/view_predictions.sh "${CONFIG}" "${WEIGHTS}" "${FRAMES}")
    [ -n "${SWEEP_DIR}" ] && CMD+=(--sweep "${SWEEP_DIR}" --seed "${SWEEP_SEED}")
    [ -n "${TRIALS}" ] && CMD+=(--trials "${TRIALS}")
    [ -n "${RECORD}" ] && CMD+=(--record "${RECORD}")
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
# The legacy checkpoints were trained with a 7-frame window (their input layer
# is sized to it); the current default is 5.
[ "${WEIGHTS}" = "--legacy" ] && export DIFFTACTILE_CLIP_LEN=7

if [ -n "${RECORD}" ]; then
    # Recording renders offscreen: no window, no display of any kind needed,
    # and nothing appears on anyone's desktop. qt_viewer.run_browser drives the
    # viewer from the key script the navigator supplies.
    unset DISPLAY WAYLAND_DISPLAY
    export QT_QPA_PLATFORM=offscreen
    export DIFFTACTILE_RECORD_MP4="${RECORD}"
elif [ "${BACKEND}" = "wayland" ]; then
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
VIEW_ARGS=("${CONFIG}" "${WEIGHTS}" "${FRAMES}")
[ -n "${TRIALS}" ] && VIEW_ARGS+=(--trials "${TRIALS}")
[ -n "${RECORD}" ] && echo "Recording to ${RECORD} (offscreen, automatic stepping)."
if [ -n "${SWEEP_DIR}" ]; then
    VIEW_ARGS+=(--sweep "${SWEEP_DIR}" --seed "${SWEEP_SEED}")
    echo "Viewing predictions: ${CONFIG} ${FRAMES} (sweep ${SWEEP_DIR}, seed ${SWEEP_SEED})"
else
    echo "Viewing predictions: ${CONFIG} ${WEIGHTS} ${FRAMES}"
fi
if [ "${FRAMES}" = "--central" ]; then
    echo "Showing each sliding window's CENTRAL frame only (what the metrics report)."
    echo "Keys: i/o trial, j/k central frame, q quit."
else
    echo "Showing every frame of every sliding window."
    echo "Keys: i/o trial, j/k clip, n/m frame, q quit."
fi
exec python -m difftactile.scripts.script_visualise "${VIEW_ARGS[@]}"
