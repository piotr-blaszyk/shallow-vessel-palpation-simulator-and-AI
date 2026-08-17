#!/usr/bin/env bash
#
# Bird's-eye vessel maps for the PROJECT PAGE (docs/index.html), written as
# small lossless WebP files to docs/images/vessel_maps/ (committed).
#
# Runs the vessel-map job for the four models exactly as the page shows them
# and converts each map's raw confusion overlay (confusion_r00.png, 1 px = 1 mm,
# ground truth not grown) into a x5 nearest-neighbour lossless WebP:
#
#   Sim -> Sim          10 maps: the ten held-out trajectories of the Sim -> Sim
#                       prediction video, in the video's order (needs
#                       ./docker/vessel_map_sim_test_trajectories.sh once)
#   Sim -> Silicone      1 map: the whole silicone phantom (video ground truth)
#   Sim -> Meat         10 maps: one per meat trial, in the video's order
#   Meat -> Silicone     1 map: the silicone phantom, meat-trained model
#
# docs/images/vessel_maps/manifest.md records the run each image came from.
# The full runs stay under difftactile/output/vessel_maps/ as usual.
#
# Run INSIDE the container. ~3 minutes.
#
# Usage:
#   ./docker/website_vessel_maps.sh
#
set -euo pipefail
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

if [ -z "$(ls difftactile/output/vessel_map_sim/test_trajectories/*.npz 2>/dev/null)" ]; then
    echo "The ten Sim->Sim video trajectories have not been re-simulated with poses yet:"
    echo "run ./docker/vessel_map_sim_test_trajectories.sh (inside the container) first."
    exit 1
fi
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi
python -m difftactile.scripts.script_website_vessel_maps
