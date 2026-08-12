#!/usr/bin/env bash
# Run ONE already-configured iteration and screenshot it.
#
# Assumes set_spacing.py + the mesh/pre_main regeneration have already been done
# for this factor; this script only does step (4) of the sweep. Split out of
# run_spacing_sweep.sh so a single iteration can be retried (e.g. with a
# different TI_DEVICE_MEMORY_GB) without redoing the meshing.
#
# Usage: run_one_iteration.sh <factor> [timeout_s] [ti_device_memory_gb]
set -u

FACTOR="$1"
TIMEOUT="${2:-60}"
TI_MEM="${3:-9}"
SHOT_INTERVAL=5
XDISP=":95"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C_REPO=/workspace/shallow-vessel-palpation-simulator-and-AI
REL_OUT="difftactile/manual_or_experimental_data/particle_spacing_study/iteration_${FACTOR}"
OUT="${REPO_ROOT}/${REL_OUT}"
CONTAINER="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"

mkdir -p "${OUT}"
rm -f "${OUT}"/shot_*.png

docker exec "${CONTAINER}" pkill -9 -f script_main >/dev/null 2>&1
sleep 3

docker exec -d -e DISPLAY="${XDISP}" -e TI_DEVICE_MEMORY_GB="${TI_MEM}" \
    -e DIFFTACTILE_TRAJECTORIES=3 -e DIFFTACTILE_NUM_LOOPS=1 "${CONTAINER}" \
    bash -lc "cd ${C_REPO} && python -m difftactile.scripts.script_main \
              > ${C_REPO}/${REL_OUT}/sim.log 2>&1"

i=0
ELAPSED=0
while [ "${ELAPSED}" -lt "${TIMEOUT}" ]; do
    sleep "${SHOT_INTERVAL}"
    ELAPSED=$((ELAPSED + SHOT_INTERVAL))
    i=$((i + 1))
    SHOT=$(printf "%s/shot_%02d_t%03ds.png" "${OUT}" "${i}" "${ELAPSED}")
    # Grab the 3D scene window specifically; the "tactile readout" window
    # overlaps it, so a root grab would capture the wrong one.
    WID=$(DISPLAY="${XDISP}" xdotool search --name "high-level camera" 2>/dev/null | head -1)
    if [ -n "${WID}" ]; then
        DISPLAY="${XDISP}" xdotool windowraise "${WID}" 2>/dev/null
        eval "$(DISPLAY="${XDISP}" xdotool getwindowgeometry --shell "${WID}" 2>/dev/null)"
        ffmpeg -y -loglevel error -f x11grab -video_size "${WIDTH}x${HEIGHT}" \
               -i "${XDISP}+${X},${Y}" -frames:v 1 "${SHOT}" 2>/dev/null
        [ -s "${SHOT}" ] && echo "captured ${SHOT}"
    else
        echo "t=${ELAPSED}s: no 3D window yet"
    fi
    docker exec "${CONTAINER}" pgrep -f script_main >/dev/null 2>&1 || {
        echo "sim process gone at t=${ELAPSED}s"; break; }
done

docker exec "${CONTAINER}" pkill -9 -f script_main >/dev/null 2>&1
echo "--- iteration ${FACTOR}: $(ls -1 "${OUT}"/shot_*.png 2>/dev/null | wc -l) screenshots"
