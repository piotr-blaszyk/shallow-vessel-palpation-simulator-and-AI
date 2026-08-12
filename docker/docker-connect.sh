#!/usr/bin/env bash
#
# Open an interactive shell inside the running shallow-vessel-palpation-simulator-and-AI container.
# Start it first with ./docker/docker-run.sh
#
set -euo pipefail

CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"

if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
    echo "Container '${CONTAINER_NAME}' is not running."
    echo "Start it with: ./docker/docker-run.sh"
    exit 1
fi

# -l gives a login shell. Note the container runs as your (non-root) UID with
# HOME=/tmp, so the image's /root/.bashrc is NOT read; DIFFTACTILE_ROOT and
# PYTHONPATH arrive via the image's ENV and the cwd via WORKDIR instead.
exec docker exec -it "${CONTAINER_NAME}" bash -l
