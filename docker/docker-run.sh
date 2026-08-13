#!/usr/bin/env bash
#
# Start (or attach to) the long-lived shallow-vessel-palpation-simulator-and-AI container.
#
# Workflow:
#   1) ./docker/docker-build.sh      # build the image once
#   2) ./docker/docker-run.sh        # start the container as a daemon (idempotent)
#   3) ./docker/docker-connect.sh    # interactive shell inside it
#
# Safe to run repeatedly: it never starts a duplicate. The container is created
# with --rm, so `docker stop vessel-palpation` removes it and the next run creates a
# fresh one from the current image (i.e. stop + run picks up a rebuild).
#
# The repository is bind-mounted at /workspace/shallow-vessel-palpation-simulator-and-AI, so edits on the
# host apply immediately inside the container with no rebuild.
#
set -euo pipefail

IMAGE_NAME="${VESSEL_PALPATION_IMAGE:-vessel-palpation:cuda12.6}"
CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_next_steps() {
    echo
    echo "Next steps (run these from the HOST):"
    echo "  Open a shell:          ./docker/docker-connect.sh"
    echo "  Run the full pipeline: docker exec -it ${CONTAINER_NAME} bash -lc './docker/run_pipeline.sh --help'"
    echo "  Stop the container:    docker stop ${CONTAINER_NAME}"
}

# --- Guard: already running? --------------------------------------------------
if [ -n "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
    echo "Container '${CONTAINER_NAME}' is already running."
    print_next_steps
    exit 0
fi

# --- A stopped leftover blocks the name: remove it ----------------------------
# Tolerate failure: containers are created with --rm, so a just-stopped one may
# be removed by the daemon between the check and the `docker rm`.
if [ -n "$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")" ]; then
    echo "Removing stopped leftover container '${CONTAINER_NAME}'..."
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# --- GUI passthrough ----------------------------------------------------------
# The container hosts two kinds of GUI application, and they use different
# transports. Both sockets are therefore mounted, and each app picks its own:
# backend selection is per PROCESS (via DISPLAY / WAYLAND_DISPLAY /
# QT_QPA_PLATFORM), never per container, so there is no conflict.
#
#   X11 (the default here)  Taichi GGUI, Gmsh's FLTK viewer, the cv2 windows and
#                           matplotlib. These are X clients with no Wayland
#                           backend, so they go through the host's X server -
#                           which on a Wayland desktop is the session's own
#                           Xwayland, reached at /tmp/.X11-unix.
#   Wayland (opt-in)        The two PySide6 annotation viewers launched by
#                           docker/annotate_data_docker.sh. That script unsets
#                           DISPLAY and forces QT_QPA_PLATFORM=wayland in its own
#                           process, so it talks straight to the compositor over
#                           the socket mounted below. See the header there.
#
# DISPLAY is still set container-wide because the X11 group is the majority and
# has no alternative; the Wayland opt-in is what overrides it, not the reverse.
RUN_DISPLAY="${DISPLAY:-}"
if [ -z "${RUN_DISPLAY}" ]; then
    for sock in /tmp/.X11-unix/X*; do
        [ -e "${sock}" ] && RUN_DISPLAY=":${sock##*/X}" && break
    done
    RUN_DISPLAY="${RUN_DISPLAY:-:0}"
fi
echo "Using DISPLAY=${RUN_DISPLAY} for X11 GUI windows."

# --- Wayland passthrough ------------------------------------------------------
# The entire Wayland client/compositor interface is one Unix socket, so a
# bind-mount of it makes a containerised client indistinguishable from a native
# one: same IPC path, no proxy, no relay, no nested compositor. Buffers are
# passed as file descriptors over SCM_RIGHTS, which crosses namespaces intact,
# so wl_shm frames are shared zero-copy exactly as on the host; input arrives on
# the same socket via wl_seat, needing no /dev/input access.
#
# XDG_RUNTIME_DIR inside the container is set to the host's own /run/user/<uid>
# path so Qt's Wayland plugin - which resolves the socket relative to it - finds
# it where it expects. Only the one socket is mounted, not the whole runtime
# directory, which would hand the container the session bus and much more.
WAYLAND_ARGS=()
HOST_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -S "${HOST_RUNTIME_DIR}/${WAYLAND_DISPLAY}" ]; then
    CTR_RUNTIME_DIR="/run/user/$(id -u)"
    WAYLAND_ARGS=(
        -e XDG_RUNTIME_DIR="${CTR_RUNTIME_DIR}"
        -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY}"
        -v "${HOST_RUNTIME_DIR}/${WAYLAND_DISPLAY}:${CTR_RUNTIME_DIR}/${WAYLAND_DISPLAY}"
    )
    echo "Using WAYLAND_DISPLAY=${WAYLAND_DISPLAY} for native Wayland windows."
else
    echo "No Wayland socket found; X11 only (annotate_data_docker.sh will use xcb)."
fi

# Let the container talk to the host X server. The container runs as the
# invoking user (not root), so grant local access broadly rather than to root
# specifically. Harmless when there is no X server (headless): CPU/GPU compute
# still works, only GUIs need this.
if command -v xhost >/dev/null 2>&1; then
    xhost +local: >/dev/null 2>&1 || true
fi

# X11 authorisation. The container runs as the invoking user with HOME=/tmp, so
# the cookie is mounted somewhere that user can actually read and XAUTHORITY is
# pointed at it explicitly rather than relying on the ~/.Xauthority default.
#
# On a GNOME Wayland session XAUTHORITY is often unset, because Mutter writes its
# Xwayland cookie to a randomly-named file that is regenerated every login
# ($XDG_RUNTIME_DIR/.mutter-Xwaylandauth.XXXXXX); that file is picked up here as
# the last resort. If none of the three is found we simply skip the mount:
# `xhost +local:` above already grants local access, and Mutter's Xwayland
# commonly accepts unauthenticated local connections anyway.
XAUTH_ARGS=()
XAUTH_FILE=""
if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY}" ]; then
    XAUTH_FILE="${XAUTHORITY}"
elif [ -f "${HOME}/.Xauthority" ]; then
    XAUTH_FILE="${HOME}/.Xauthority"
else
    for cookie in "${HOST_RUNTIME_DIR}"/.mutter-Xwaylandauth.*; do
        [ -f "${cookie}" ] && XAUTH_FILE="${cookie}" && break
    done
fi
if [ -n "${XAUTH_FILE}" ]; then
    XAUTH_ARGS=(-v "${XAUTH_FILE}:/tmp/.Xauthority:ro" -e XAUTHORITY=/tmp/.Xauthority)
fi

# Exposing /dev/dri helps some GL paths (and gives a Mesa fallback).
DRI_ARGS=()
if [ -e /dev/dri ]; then
    DRI_ARGS=(--device /dev/dri)
fi

# --- Optional external data bundle -------------------------------------------
# By default the Zenodo bundle is restored into the repo itself (restore_data.sh),
# so no extra mount is needed and this is left unset.
# NOTE: DIFFTACTILE_DATA_ROOT currently redirects only the real meat trials
# (dataset.py:MEAT_CLEAN_DATA_DIR, the sole data_path() caller). Everything else
# — the simulated dataset, the silicone dataset, both checkpoints — resolves
# under the repository root, so this is not yet a general "bundle lives
# elsewhere" switch. Restore into the repo unless you specifically need that one
# directory moved.
DATA_ARGS=()
if [ -n "${DIFFTACTILE_DATA_ROOT:-}" ]; then
    if [ ! -d "${DIFFTACTILE_DATA_ROOT}" ]; then
        echo "ERROR: DIFFTACTILE_DATA_ROOT=${DIFFTACTILE_DATA_ROOT} does not exist." >&2
        exit 1
    fi
    ABS_DATA="$(cd "${DIFFTACTILE_DATA_ROOT}" && pwd)"
    DATA_ARGS=(-v "${ABS_DATA}:/workspace/data:rw" -e DIFFTACTILE_DATA_ROOT=/workspace/data)
    echo "Mounting external data bundle: ${ABS_DATA} -> /workspace/data"
fi

echo "Live-mounting repository: ${REPO_DIR} -> /workspace/shallow-vessel-palpation-simulator-and-AI"
echo "Starting container '${CONTAINER_NAME}' from image '${IMAGE_NAME}'..."

# Run as the invoking user so files written into the bind-mounted repository
# (difftactile/output/, logs/, saved_models_*/) are owned by you rather than by
# root. Without this every simulator run leaves root-owned artifacts on the host
# that a later non-root run cannot overwrite.
USER_ARGS=(--user "$(id -u):$(id -g)")
# Supplementary groups needed for the GPU and DRI device nodes.
for grp in video render; do
    gid="$(getent group "${grp}" | cut -d: -f3)"
    [ -n "${gid}" ] && USER_ARGS+=(--group-add "${gid}")
done
# Give the container user a name: mount the host's account databases read-only
# so the invoking UID/GID (and the video/render groups added above) resolve.
# Without this every shell greets you with "groups: cannot find name for group
# ID ..." and an "I have no name!" prompt.
USER_ARGS+=(-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)
# HOME must be writable for matplotlib/Taichi caches; point it at /tmp, since
# the home directory named in the host passwd entry does not exist in the image.
USER_ARGS+=(-e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/.cache)

# --gpus all + NVIDIA_DRIVER_CAPABILITIES=all (set in the image) gives both CUDA
# compute (Taichi/torch) and OpenGL (GGUI rendering).
# --ipc host raises the shared-memory limit, which PyTorch DataLoader workers
# need; the default 64 MB causes "bus error" crashes with num_workers > 0.
# --ulimit nofile raises the open-file limit from Docker's default soft 1024.
# torch's default file_descriptor sharing strategy passes one fd per shared
# tensor between workers, so the GNN training runs (num_workers_large = 16) blow
# through 1024 and die with "RuntimeError: received 0 items of ancdata" partway
# through an epoch. The hard limit is left at the daemon's own value, which is
# already far higher, so this needs no privileged configuration.
# --hostname matches the host's, because X clients set WM_CLIENT_MACHINE from it.
# When it does not match, the compositor cannot tie a window back to a local PID,
# so "Force Quit" on a hung window degrades to XKillClient. Costs nothing.
docker run -d --rm \
    --name "${CONTAINER_NAME}" \
    --hostname "$(hostname)" \
    --gpus all \
    --ipc host \
    --ulimit nofile=65535:524288 \
    "${USER_ARGS[@]}" \
    -e DISPLAY="${RUN_DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "${XAUTH_ARGS[@]}" \
    "${WAYLAND_ARGS[@]}" \
    "${DRI_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    -v "${REPO_DIR}:/workspace/shallow-vessel-palpation-simulator-and-AI:rw" \
    "${IMAGE_NAME}" \
    sleep infinity >/dev/null

echo
echo "Container '${CONTAINER_NAME}' is up."
print_next_steps
