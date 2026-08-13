#!/usr/bin/env bash
#
# Manual annotation and annotation review, IN THE DOCKER CONTAINER.
#
# This is the in-container twin of ./docker/annotate_data.sh. Both drive exactly
# the same two viewers, through the same modules, against the same library
# versions (PySide6 6.9.1 for the windows, PyAV 15.1.0 for decoding) - the only
# difference is where the process runs. Keeping the two side by side is the
# point: run one, run the other, and any difference in responsiveness or
# behaviour is attributable to the container rather than to the code.
#
# See annotate_data.sh for what the two viewers do, their keybindings, and the
# note on which data each needs. That file is the reference; this one documents
# only what is different about running inside Docker.
#
# Usage (from the HOST - it execs into the running container for you):
#   ./docker/annotate_data_docker.sh --silicone [--source DIR]
#   ./docker/annotate_data_docker.sh --meat
#
# Options:
#   --silicone / --meat   Which dataset. Same meaning as in annotate_data.sh.
#   --source DIR          Staging source for the silicone videos, as a path
#                         INSIDE the container. Defaults to the same location
#                         annotate_data.sh uses, which only resolves if you have
#                         mounted it (see "Data" below).
#   --x11                 Force the X11/Xwayland backend instead of Wayland.
#                         This is the A/B switch for isolating a Wayland-specific
#                         bug, and the automatic fallback on an X11-only host.
#   --shell               Print the docker exec command instead of running it,
#                         for when you want to poke at the environment by hand.
#
# It can also be run from INSIDE the container (e.g. after docker-connect.sh), in
# which case it skips the exec and launches the viewer directly.
#
# GRAPHICS. The container gets both display transports and this script picks one
# per process, which is the only level at which the choice exists - there is no
# container-wide "this is an X11 container" state:
#
#   Wayland (default)  docker-run.sh bind-mounts the host compositor's Unix
#                      socket. The client/compositor interface IS that socket, so
#                      a containerised Qt client talks to Mutter over the very
#                      same IPC path a host-native one would: no proxy, no relay,
#                      no nested compositor, nothing left to be slow. Buffers
#                      cross as file descriptors over SCM_RIGHTS, which survives
#                      the namespace boundary, so frames are shared zero-copy;
#                      keyboard events arrive on the same socket via wl_seat with
#                      no /dev/input access needed. This should be within noise
#                      of bare metal for a frame stepper.
#
#                      DISPLAY is unset below before launching. That single line
#                      matters more than it looks: a stray DISPLAY is the most
#                      common way a Wayland-capable toolkit silently ends up on
#                      xcb, and since docker-run.sh sets DISPLAY container-wide
#                      for the project's X11-only tools (Taichi GGUI, Gmsh, cv2),
#                      it is definitely set here. QT_QPA_PLATFORM is set to plain
#                      `wayland` with NO `;xcb` fallback, so a broken Wayland
#                      setup fails loudly instead of quietly degrading into the
#                      thing we are trying to measure against.
#
#   X11 (--x11)        Uses the host's existing X server via /tmp/.X11-unix. On a
#                      Wayland desktop that is the session's own Xwayland, so the
#                      window is a normal sibling on your desktop rather than
#                      living inside a nested compositor - host window management
#                      and the X<->Wayland clipboard bridge both keep working.
#                      Expect this to be the choppier of the two: every repaint
#                      takes an extra copy through the compatibility layer.
#
# Verifying which one you actually got: run `xlsclients` on the host while a
# viewer is up. In Wayland mode it must NOT be listed; in --x11 mode it must be.
# The script also prints Qt's own view of the selection at startup.
#
# DATA. Everything under the repository is bind-mounted, so the meat viewer and
# any silicone videos already staged into the working tree just work. The
# silicone staging *source* is a path on the author's host that is NOT mounted
# into the container - so if the silicone videos are not already in the tree,
# either stage them once on bare metal with ./docker/annotate_data.sh --silicone
# (they land in the repo, which the container sees), or pass --source with a path
# you have mounted yourself.
#
set -euo pipefail

CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

DATASET=""
SOURCE_ARGS=()
BACKEND="wayland"
PRINT_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --silicone) DATASET="silicone" ;;
        --meat)     DATASET="meat" ;;
        --source)   SOURCE_ARGS=(--source "${2:?--source needs a directory}"); shift ;;
        --x11)      BACKEND="x11" ;;
        --shell)    PRINT_ONLY=1 ;;
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

# --- Stage 1: on the host, hand the job to the container ----------------------
#
# Re-executes this same script inside the container, so all the argument parsing
# and viewer logic below lives in exactly one place. /.dockerenv is what stops
# that from recursing: it exists only in the container, so stage 2 never re-runs
# stage 1.
if [ ! -f /.dockerenv ]; then
    if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
        cat >&2 <<EOF
ERROR: container '${CONTAINER_NAME}' is not running.

Start it first:

    ./docker/docker-run.sh

If it was started before Wayland passthrough was added, it will not have the
compositor socket mounted and this script will fall back to X11. Restart it to
pick the mount up:

    docker stop ${CONTAINER_NAME} && ./docker/docker-run.sh
EOF
        exit 1
    fi

    # Warn early and specifically if the running container predates the Wayland
    # mount, rather than letting Qt fail with a bare "could not connect".
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

    # -it only when there really is a terminal on both ends: `docker exec -i`
    # fails outright ("cannot attach stdin to a TTY-enabled container") when
    # stdin is a pipe, which is what happens under CI or `| tee`. The viewers
    # read their input from the Qt window, not from stdin, so dropping the TTY
    # costs nothing beyond terminal echo of the banners.
    TTY_ARGS=()
    [ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)
    CMD=(docker exec "${TTY_ARGS[@]}" "${CONTAINER_NAME}"
         ./docker/annotate_data_docker.sh "--${DATASET}")
    [ "${BACKEND}" = "x11" ] && CMD+=(--x11)
    [ ${#SOURCE_ARGS[@]} -gt 0 ] && CMD+=("${SOURCE_ARGS[@]}")

    if [ "${PRINT_ONLY}" -eq 1 ]; then
        printf '%q ' "${CMD[@]}"; echo
        exit 0
    fi

    echo "Running the annotator inside '${CONTAINER_NAME}' (backend: ${BACKEND})."
    exec "${CMD[@]}"
fi

# --- Stage 2: inside the container --------------------------------------------
cd "${REPO_DIR}"

# These tools exist to be driven by hand, so opt into blocking windows rather
# than the project default of never waiting on a GUI. Same as on bare metal.
export DIFFTACTILE_INTERACTIVE=1

if [ "${BACKEND}" = "wayland" ]; then
    if [ -z "${WAYLAND_DISPLAY:-}" ] || [ ! -S "${XDG_RUNTIME_DIR:-/nonexistent}/${WAYLAND_DISPLAY}" ]; then
        echo "ERROR: no Wayland socket at \${XDG_RUNTIME_DIR}/\${WAYLAND_DISPLAY}." >&2
        echo "Restart the container with ./docker/docker-run.sh, or pass --x11." >&2
        exit 1
    fi
    # The whole point of the exercise: a real Wayland client, not an Xwayland
    # one. Unsetting DISPLAY is not belt-and-braces - docker-run.sh sets it
    # container-wide for the project's X11-only tools, and leaving it set is the
    # classic way a toolkit silently picks xcb. No `;xcb` fallback either, so a
    # misconfiguration is an error rather than a silent downgrade into the exact
    # code path this script exists to avoid.
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

# Qt's Wayland plugin locates the compositor socket relative to XDG_RUNTIME_DIR
# and aborts without it. docker-run.sh sets it, but keep the same guard the
# bare-metal script has so a hand-built `docker run` still works.
if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

PYTHON="${DIFFTACTILE_ANNOTATOR_PYTHON:-python}"

# The image gained PySide6 and PyAV for exactly this script (Dockerfile section
# 9). If they are missing, the container is older than that layer.
if ! ${PYTHON} -c "import PySide6, av" >/dev/null 2>&1; then
    MISSING=$(${PYTHON} -c "
import importlib
print(' '.join(m for m in ('PySide6', 'av') if not importlib.util.find_spec(m)))
" 2>/dev/null || echo "PySide6 av")
    cat >&2 <<EOF
ERROR: the container's interpreter is missing: ${MISSING}

    interpreter: ${PYTHON}

The image installs PySide6 and PyAV for this script, so this container was built
from an older image. Rebuild and restart it:

    ./docker/docker-build.sh
    docker stop ${CONTAINER_NAME} && ./docker/docker-run.sh
EOF
    exit 1
fi

# Report what Qt actually resolved, so the Wayland-vs-X11 comparison is never
# guesswork. Cross-check from the host with `xlsclients`: in Wayland mode the
# viewer must not appear in that list.
echo "Qt platform: ${QT_QPA_PLATFORM}  (DISPLAY=${DISPLAY:-<unset>}, WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>})"

# Hand off to the bare-metal script for everything that is not display-related:
# argument meaning, the silicone staging step, the key-binding banners and the
# module entrypoints. It already has an "inside the container" branch that keeps
# the current interpreter, and the environment set above survives into it, so the
# viewer logic genuinely is shared rather than duplicated.
exec ./docker/annotate_data.sh "--${DATASET}" "${SOURCE_ARGS[@]}"
