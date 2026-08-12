#!/usr/bin/env bash
# Particle-spacing refinement sweep.
#
# Motivation: the ViTacTip FEM particles (green) ghost into the MPM phantom
# particles (blue) — the two bodies interpenetrate instead of the sensor
# deforming against the phantom surface. The fix is to refine the "atom" spacing
# of both bodies so the contact stencil can resolve the interface.
#
# For each multiplication factor f in {0.5, 0.25, 0.125} this script:
#   1. scales every source-of-truth spacing knob by f (see set_spacing.py),
#   2. scales contact.dt_override by f — CFL gives dt ~ C*dx/c and the wave speed
#      c is unchanged, so the stable timestep shrinks *linearly* with spacing,
#   3. regenerates the Gmsh meshes and the pre_main geometry,
#   4. runs ONE press-and-slide trajectory (DIFFTACTILE_TRAJECTORIES=3,
#      DIFFTACTILE_NUM_LOOPS=1), screenshotting the 3D scene every 5 s until
#      TIMEOUT seconds have elapsed.
#
# Architecture: taichi is only installed inside the `difftactile` container, and
# the GGUI window must actually render for screenshots to exist. The container
# therefore draws into a *host-side* Xvfb virtual display, which this script
# screenshots with ffmpeg's x11grab. A virtual display is used rather than the
# real one (:1) so the sweep neither disturbs nor captures the user's desktop.
#
# Screenshots land in iteration_<f>/ next to this script.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STUDY_DIR="${REPO_ROOT}/difftactile/manual_or_experimental_data/particle_spacing_study"
# Path of the study dir as seen from inside the container (repo is bind-mounted).
C_REPO=/workspace/shallow-vessel-palpation-simulator-and-AI
C_STUDY="${C_REPO}/difftactile/manual_or_experimental_data/particle_spacing_study"
CONTAINER="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
TIMEOUT="${TIMEOUT:-60}"
SHOT_INTERVAL=5
XDISP=":95"

cd "${REPO_ROOT}"

dexec() { docker exec "$@"; }

# --- one dedicated virtual display for the whole sweep ------------------------
pkill -9 -f "Xvfb ${XDISP}" 2>/dev/null
sleep 1
Xvfb "${XDISP}" -screen 0 1280x1024x24 -nolisten tcp >/dev/null 2>&1 &
XVFB_PID=$!
sleep 2
trap 'kill -9 ${XVFB_PID} 2>/dev/null' EXIT

for FACTOR in 0.5 0.25 0.125; do
    OUT="${STUDY_DIR}/iteration_${FACTOR}"
    rm -rf "${OUT}"; mkdir -p "${OUT}"
    echo "=============================================================="
    echo "=== ITERATION factor=${FACTOR}  ->  ${OUT}"
    echo "=============================================================="

    # (1)+(2) rewrite the source-of-truth JSONs for this factor.
    python3 "${STUDY_DIR}/set_spacing.py" "${FACTOR}" 2>&1 | tee "${OUT}/params.log"

    # (3) regenerate meshes + derived geometry. characteristic_length_factor only
    # takes effect once the Gmsh meshes are rebuilt, and pre_main re-derives the
    # particle counts from the new spacing.
    FAILED=0
    for STEP in script_apply_scaling script_generate_vitactip_mesh_gmsh \
                script_generate_vein_mesh_gmsh script_pre_main; do
        echo "--- ${STEP}"
        if ! dexec -e DIFFTACTILE_HEADLESS=1 "${CONTAINER}" \
                bash -lc "cd ${C_REPO} && timeout 1800 python -m difftactile.scripts.${STEP}" \
                > "${OUT}/${STEP}.log" 2>&1; then
            echo "!!! ${STEP} FAILED (see ${OUT}/${STEP}.log)"
            tail -25 "${OUT}/${STEP}.log"
            FAILED=1
            break
        fi
    done
    [ "${FAILED}" -eq 1 ] && continue

    # (4) run the simulator windowed on the virtual display.
    # Trajectory type 3 = "slide" (the press-and-slide domain-adaptation run).
    dexec -d -e DISPLAY="${XDISP}" -e DIFFTACTILE_TRAJECTORIES=3 \
          -e DIFFTACTILE_NUM_LOOPS=1 "${CONTAINER}" \
          bash -lc "cd ${C_REPO} && python -m difftactile.scripts.script_main \
                    > ${C_STUDY}/iteration_${FACTOR}/sim.log 2>&1"

    i=0
    ELAPSED=0
    while [ "${ELAPSED}" -lt "${TIMEOUT}" ]; do
        sleep "${SHOT_INTERVAL}"
        ELAPSED=$((ELAPSED + SHOT_INTERVAL))
        i=$((i + 1))
        SHOT=$(printf "%s/shot_%02d_t%03ds.png" "${OUT}" "${i}" "${ELAPSED}")
        # Grab the 3D scene window specifically. The "tactile readout" window
        # overlaps it, so a root-window grab would capture the wrong one.
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
        dexec "${CONTAINER}" pgrep -f script_main >/dev/null 2>&1 || {
            echo "sim process gone at t=${ELAPSED}s"; break; }
    done

    dexec "${CONTAINER}" pkill -9 -f script_main 2>/dev/null
    sleep 3
    echo "--- iteration ${FACTOR}: $(ls -1 "${OUT}"/shot_*.png 2>/dev/null | wc -l) screenshots"
done

echo "=== sweep complete"
