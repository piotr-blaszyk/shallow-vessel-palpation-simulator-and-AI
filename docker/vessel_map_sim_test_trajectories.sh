#!/usr/bin/env bash
#
# Re-simulate the TEN held-out simulated trajectories shown in the project-page
# Sim->Sim prediction video, WITH sensor poses, so that their bird's-eye vessel
# maps can be drawn (docker/vessel_map.sh A-to-A --test-trajectories, and
# docker/website_vessel_maps.sh).
#
# The published simulated dataset records no sensor pose, but it was collected
# seeded, so replaying its RNG stream reproduces any of its trials exactly
# (main.py::vessel_map_test_trajectories_main - which also VERIFIES each
# re-simulated trial against the published file and stops on a mismatch). The
# ten are the video's selection: 7 vessel-present + 3 vessel-absent trajectories
# drawn with a fixed seed (DIFFTACTILE_VIEW_TRIALS_SEED, default 0) and
# interleaved a a b a a b a a b a - the same function the viewer's
# `--trials interleaved:7:3` uses (cnn/trial_selection.py).
#
# Writes difftactile/output/vessel_map_sim/test_trajectories/ (published
# markers + re-simulated poses, plus selection.json with the display order),
# which ships in the data bundle - so this only needs running to regenerate.
# Run INSIDE the container (Taichi + GPU); ~15 minutes.
#
# Usage:
#   ./docker/vessel_map_sim_test_trajectories.sh
#
set -euo pipefail
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi
python -m difftactile.scripts.script_vessel_map_test_trajectories
