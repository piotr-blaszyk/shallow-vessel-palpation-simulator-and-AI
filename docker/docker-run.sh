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
# with --rm, so `docker stop difftactile` removes it and the next run creates a
# fresh one from the current image (i.e. stop + run picks up a rebuild).
#
# The repository is bind-mounted at /workspace/shallow-vessel-palpation-simulator-and-AI, so edits on the
# host apply immediately inside the container with no rebuild.
#
set -euo pipefail

IMAGE_NAME="${DIFFTACTILE_IMAGE:-difftactile:cuda12.6}"
CONTAINER_NAME="${DIFFTACTILE_CONTAINER:-difftactile}"
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
# Taichi GGUI, Gmsh's FLTK viewer, the cv2 annotation tool and matplotlib are all
# genuine X11 GUI applications, so the container needs the host's X server.
RUN_DISPLAY="${DISPLAY:-}"
if [ -z "${RUN_DISPLAY}" ]; then
    for sock in /tmp/.X11-unix/X*; do
        [ -e "${sock}" ] && RUN_DISPLAY=":${sock##*/X}" && break
    done
    RUN_DISPLAY="${RUN_DISPLAY:-:0}"
fi
echo "Using DISPLAY=${RUN_DISPLAY} for GUI windows."

# Let the container talk to the host X server. The container runs as the
# invoking user (not root), so grant local access broadly rather than to root
# specifically. Harmless when there is no X server (headless): CPU/GPU compute
# still works, only GUIs need this.
if command -v xhost >/dev/null 2>&1; then
    xhost +local: >/dev/null 2>&1 || true
fi

XAUTH_ARGS=()
if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY}" ]; then
    XAUTH_ARGS=(-v "${XAUTHORITY}:/root/.Xauthority:rw" -e XAUTHORITY=/root/.Xauthority)
elif [ -f "${HOME}/.Xauthority" ]; then
    XAUTH_ARGS=(-v "${HOME}/.Xauthority:/root/.Xauthority:rw" -e XAUTHORITY=/root/.Xauthority)
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
# (dataset.py:IROS_CLEAN_DATA_DIR, the sole data_path() caller). Everything else
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
# HOME must be writable for matplotlib/Taichi caches; point it at /tmp since the
# container user has no /etc/passwd entry of its own.
USER_ARGS+=(-e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/.cache)

# --gpus all + NVIDIA_DRIVER_CAPABILITIES=all (set in the image) gives both CUDA
# compute (Taichi/torch) and OpenGL (GGUI rendering).
# --ipc host raises the shared-memory limit, which PyTorch DataLoader workers
# need; the default 64 MB causes "bus error" crashes with num_workers > 0.
docker run -d --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc host \
    "${USER_ARGS[@]}" \
    -e DISPLAY="${RUN_DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "${XAUTH_ARGS[@]}" \
    "${DRI_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    -v "${REPO_DIR}:/workspace/shallow-vessel-palpation-simulator-and-AI:rw" \
    "${IMAGE_NAME}" \
    sleep infinity >/dev/null

echo
echo "Container '${CONTAINER_NAME}' is up."
print_next_steps
