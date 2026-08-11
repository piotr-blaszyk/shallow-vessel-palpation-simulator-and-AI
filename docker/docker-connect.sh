#!/usr/bin/env bash
#
# Open an interactive shell inside the running diff-tactile-fork container.
# Start it first with ./docker/docker-run.sh
#
set -euo pipefail

CONTAINER_NAME="${DIFFTACTILE_CONTAINER:-difftactile}"

if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
    echo "Container '${CONTAINER_NAME}' is not running."
    echo "Start it with: ./docker/docker-run.sh"
    exit 1
fi

# -l gives a login shell so /root/.bashrc (DIFFTACTILE_ROOT, PYTHONPATH, cd) applies.
exec docker exec -it "${CONTAINER_NAME}" bash -l
