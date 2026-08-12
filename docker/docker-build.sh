#!/usr/bin/env bash
#
# Build the shallow-vessel-palpation-simulator-and-AI container image.
#
# Everything (Taichi simulator + PyTorch/PyG GNN stack + analysis tools) lives in
# a single image on top of nvidia/cuda:12.6.3-devel-ubuntu22.04, using ONE system
# Python 3.10 interpreter (no conda/micromamba).
#
# Usage:
#   ./docker/docker-build.sh                 # normal build (uses layer cache)
#   ./docker/docker-build.sh --no-cache      # extra args pass straight through
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${VESSEL_PALPATION_IMAGE:-vessel-palpation:cuda12.6}"

echo "=============================================================="
echo " Building image: ${IMAGE_NAME}"
echo " Dockerfile:     ${SCRIPT_DIR}/Dockerfile"
echo "=============================================================="
echo
echo "NOTE: the first build downloads the CUDA base image plus the torch /"
echo "      torch-geometric wheels (several GB) and can take 15-30 minutes."
echo

# The build context is the docker/ dir only: the image installs dependencies and
# does not COPY the source (the source is bind-mounted at run time instead), so
# keeping the context tiny avoids shipping the multi-GB dataset to the daemon.
DOCKER_BUILDKIT=1 docker build \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "$@" \
    "${SCRIPT_DIR}"

echo
echo "Build complete: ${IMAGE_NAME}"
echo "Start the container with:  ./docker/docker-run.sh"
