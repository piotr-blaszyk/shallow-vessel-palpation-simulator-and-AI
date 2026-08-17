#!/usr/bin/env bash
#
# Six orthogonal-view screenshots of the ViTacTip tetrahedral mesh as Gmsh
# draws it - line of sight along +x, -x, +y, -y, +z and -z - saved as small
# WebP files for the project page:
#
#   docs/images/sensor_mesh/vitactip_mesh_along_{x,y,z}_{plus,minus}.webp
#
# ("along +x" = the line of sight points along +x, i.e. seen from the -x side.)
# The uncompressed PNGs are kept in difftactile/output/sensor_mesh_screenshots/
# (gitignored). Only docs/ is committed, which is why the images are WebP.
#
# Run this FROM THE HOST with the container running (docker/docker-run.sh):
# it execs into the container, where Gmsh lives. Gmsh can only rasterise
# through its FLTK window, so a Gmsh window opens on the display for a few
# seconds and closes by itself. The mesh (difftactile/output/gmsh_debug.msh)
# is generated first if it is missing; pass --regenerate to rebuild it anyway
# (that also rewrites the simulator's mesh pickles, exactly as
# script_generate_vitactip_mesh_gmsh does).
#
# Usage:
#   ./docker/sensor_mesh_screenshots.sh
#   ./docker/sensor_mesh_screenshots.sh --regenerate
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
CONTAINER_REPO="/workspace/$(basename "${REPO_DIR}")"
MSH="difftactile/output/gmsh_debug.msh"

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'; }

REGENERATE=0
for arg in "$@"; do
    case "${arg}" in
        --regenerate) REGENERATE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
    echo "ERROR: container '${CONTAINER_NAME}' is not running (./docker/docker-run.sh)." >&2
    exit 1
fi

in_container() {
    # Run a command inside the container from the repository root.
    docker exec "${CONTAINER_NAME}" bash -lc "cd '${CONTAINER_REPO}' && $*"
}

if [ "${REGENERATE}" -eq 1 ] || [ ! -f "${REPO_DIR}/${MSH}" ]; then
    echo "Generating the ViTacTip mesh (${MSH})..."
    in_container "python -m difftactile.scripts.script_generate_vitactip_mesh_gmsh"
fi

echo "Rendering the six views in Gmsh (a Gmsh window opens briefly)..."
in_container "python -m difftactile.scripts.script_sensor_mesh_screenshots"
echo "Done: ${REPO_DIR}/docs/images/sensor_mesh/"
