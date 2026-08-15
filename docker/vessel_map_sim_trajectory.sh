#!/usr/bin/env bash
#
# Simulate the ONE vein-present slide (with sensor poses) behind the Sim->Sim
# bird's-eye vessel map, then Hungarian-reorder it into the dataset layout.
#
# The published simulated dataset does not record the sensor pose, so none of
# its trajectories can be reprojected onto the phantom plane the way the real
# datasets are. This draws a fresh slide from the same generator under its own
# seed (DIFFTACTILE_SEED, default 2026 - deliberately not the dataset's 42),
# records `T_BA` per frame and the vein's world centreline, and writes
#
#   difftactile/output/vessel_map_sim/raw/                  the raw trajectory
#   difftactile/output/vessel_map_sim/raw_reordered_dense/  what vessel_map.sh A-to-A reads
#
# Both ship in the data bundle, so this only needs running to draw a DIFFERENT
# trajectory. Run it INSIDE the container (Taichi + GPU); the simulator window
# is shown when a display is available. ~2 minutes.
#
# Usage:
#   ./docker/vessel_map_sim_trajectory.sh
#
set -euo pipefail
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi
python -m difftactile.scripts.script_vessel_map_trajectory
