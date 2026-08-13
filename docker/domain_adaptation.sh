#!/usr/bin/env bash
#
# Domain-adaptation alignment overlays  ->  MANUSCRIPT FIGURE 5.
#
# "Alignment between simulated (red) and real (green) marker positions after
# domain adaptation for four canonical interactions: (a) press, (b) twist about
# the z-axis, (c) twist about the x-axis, and (d) slide."
#
# Each interaction is driven through the simulator to its apex pose, and there
# the simulated marker positions are overlaid on the real ones photographed from
# the physical sensor in the same configuration. Close agreement is what
# justifies training on simulated data at all.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# It needs Taichi and a GPU, so there is no bare-metal path. The figures are
# written to disk, so a display is optional.
#
# Usage:
#   ./domain_adaptation.sh          (from the docker/ directory, inside the container)
#
# Outputs, one per panel of Fig. 5 (under difftactile/output/):
#   da_overlay_press.png     (a) press
#   da_overlay_twist_z.png   (b) twist about the z-axis
#   da_overlay_twist_x.png   (c) twist about the x-axis
#   da_overlay_slide.png     (d) slide
#
# Measured MAE between simulated and real markers at each apex (1920x1080;
# ~55 px = 2 mm inter-marker spacing), ~160 s for all four:
#
#     press    6.3 px (0.23 mm)      twist_x  14.9 px (0.54 mm)
#     twist_z 10.7 px (0.39 mm)      slide    10.5 px (0.38 mm)
#
# The manuscript quotes 13.5 px aggregated, so these sit in the expected regime
# and well inside one inter-marker spacing.
#
# WHAT WAS BROKEN, in case it regresses. domain_adaptation() used to hang
# forever: its forward loop is `while last_target_reached != 1`, and nothing
# advanced the sensor between timesteps, so the PID position error sat at a
# constant 0.0982 against a 0.005 tolerance. update() advances
# vertices_undeformed_A[frame] -> [frame+1] across the sub-frames of ONE
# timestep; copy_frame() is what copies the result back to frame 0 for the next
# one. collect_training_data() had always called it, the DA loop never did - it
# relied on memory_to_cache(), which is a no-op (bare `return`). Adding
# copy_frame() to the DA loop is the fix; press now reaches its last target at
# ts 189 rather than never.
#
# FORWARD PASS ONLY, by default. domain_adaptation() also contains a parameter-
# OPTIMISATION half (backward pass, gradients, optimiser step) that cannot run
# here: set_up_torch_params(), update_params(),
# set_optimisation_params_from_log() and save_gradients_for_calibration() were
# deleted from main.py along with the optimiser and scheduler, though
# domain_adaptation() still calls them. They survive on the archival branch
# `domain-adaptation-vascular-multiple-trajectories` (~150 lines) if anyone wants
# to port them back; DIFFTACTILE_DA_OPTIMISE=1 opts in and will fail until they
# are.
#
# The figure needs none of it: the published parameters are already in
# system-params.json, so this run MEASURES the alignment they produce rather
# than re-deriving them - and update_params() was guarded by `if opts > 0`
# anyway, so the first optimisation step never changed a parameter.
#
# Environment:
#   DIFFTACTILE_ENABLE_GRAD=1     set below; allocates Taichi .grad buffers,
#                                 which the shipped config leaves off because
#                                 data collection never differentiates.
#   DIFFTACTILE_DA_MAX_TIMESTEPS  safety cap on the forward loop (default 400),
#                                 so a future regression cannot spin forever.
#   DIFFTACTILE_DA_OPTIMISE=1     attempt the deleted optimisation half.
#   DIFFTACTILE_SNAPSHOT_DIR      render the 3D scene periodically to PNGs, to
#                                 check a trajectory by eye. Needs a DISPLAY:
#                                 Taichi GGUI segfaults offscreen in this image.
#   DIFFTACTILE_SNAPSHOT_EVERY    snapshot interval in timesteps (default 20).
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

# No display -> never block on a window; the PNGs on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

# Domain adaptation differentiates through the simulator, so it needs Taichi's
# .grad buffers - which are allocated at field-construction time from
# meta.enable_grad, and are OFF in the shipped config because training-data
# collection never differentiates and the buffers roughly double GPU memory.
# Turned on here, for this process only, rather than in the shared JSON.
export DIFFTACTILE_ENABLE_GRAD="${DIFFTACTILE_ENABLE_GRAD:-1}"

echo "Running the four canonical interactions through the simulator."
echo "Simulated markers are drawn RED, real markers GREEN."
python -m difftactile.scripts.script_domain_adaptation

echo
echo "Manuscript Fig. 5 panels, written to difftactile/output/:"
for panel in "press:(a) press" \
             "twist_z:(b) twist about the z-axis" \
             "twist_x:(c) twist about the x-axis" \
             "slide:(d) slide"; do
    name="${panel%%:*}"; label="${panel#*:}"
    f="difftactile/output/da_overlay_${name}.png"
    [ -f "${f}" ] && echo "  ${f}   ${label}"
done
