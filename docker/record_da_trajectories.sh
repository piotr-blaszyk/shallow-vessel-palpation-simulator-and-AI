#!/usr/bin/env bash
#
# Record the four domain-adaptation interactions to video.
#
# Runs press, twist about z, twist about x and slide ONCE each, at the
# parameters already in system-params.json, and records the default simulator
# camera's view to .mp4.
#
# THIS IS SEPARATE FROM domain_adaptation.sh ON PURPOSE. Recording is a
# visualisation step you want occasionally, not a cost paid on every
# calibration run: this script proposes no parameters, scores nothing and runs
# each trajectory exactly once, where domain_adaptation.sh replays all four
# per BO iteration. Neither invokes the other.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
#
# NEEDS A DISPLAY. Frames are read from the Taichi GGUI colour buffer, and
# GGUI's offscreen path segfaults in this image, so there is no headless route -
# the script exits with a clear message rather than writing empty files. The
# simulator window appears on screen while it records; that window IS the
# recording.
#
# Usage:
#   ./record_da_trajectories.sh
#
# EVERY RUN GETS ITS OWN TIMESTAMPED DIRECTORY under
# difftactile/output/da_recordings/<YYYYmmdd-HHMMSS>/, so running it twice
# accumulates rather than overwrites. Each contains:
#
#   press.mp4 twist_z.mp4 twist_x.mp4 slide.mp4
#                                one video per trajectory
#   all_trajectories.mp4         the four concatenated, in panel order
#   recording.json               frame counts, fps, and the parameters used
#
# Environment:
#   DIFFTACTILE_VIDEO_FPS         output frame rate (default 30).
#   DIFFTACTILE_VIDEO_SCOPE       both (default) | per-trajectory | combined.
#   DIFFTACTILE_DA_MAX_TIMESTEPS  per-trajectory cap (default 400). `slide` is
#                                 expected to reach this - the cap is what stops
#                                 the sensor sliding off the edge of the phantom.
#   DIFFTACTILE_VEIN=1            embed the subsurface vein (drawn YELLOW). OFF
#                                 by default, matching domain adaptation, whose
#                                 reference photographs are of a plain phantom.
#   DIFFTACTILE_SEED              seed for the trajectory randomisation
#                                 (default 42).
#
# Reading the scene: the SENSOR is green, the PHANTOM is blue, the vein is
# yellow. (The red/green convention in the Fig. 5 overlays is a different,
# 2D marker-space figure - not these 3D particle colours.)
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

# Deliberately NOT forcing headless when DISPLAY is unset - unlike the other
# entrypoints, this one cannot do anything useful without a window, so let it
# report that rather than silently producing nothing.
if [ -z "${DISPLAY:-}" ]; then
    echo "ERROR: DISPLAY is unset, and this script records the GGUI window." >&2
    echo "       Taichi GGUI segfaults offscreen in this image, so there is" >&2
    echo "       no headless route. Start the container with an X socket" >&2
    echo "       mounted (docker/docker-run.sh does this)." >&2
    exit 1
fi

echo "Recording the four domain-adaptation interactions."
echo "The simulator window will appear on screen; sensor GREEN, phantom BLUE."
python -m difftactile.scripts.script_record_da_trajectories
