#!/usr/bin/env bash
#
# Domain-adaptation alignment overlays  ->  MANUSCRIPT FIGURE 5.
#
# "Alignment between simulated (red) and real (green) marker positions after
# domain adaptation for four canonical interactions: (a) press, (b) twist about
# the z-axis, (c) twist about the x-axis, and (d) slide."
#
# Each interaction is driven through the simulator until it reaches its target
# pose, and at that moment the simulated marker positions are overlaid on the
# real ones photographed from the physical sensor in the same configuration.
# Close agreement is what justifies training on simulated data at all: it is the
# evidence that the simulator's marker field resembles the real sensor's.
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
# Also prints the mean absolute marker distance for each interaction, which is
# the number the alignment is optimised against.
#
# ############################################################################
# KNOWN BROKEN AS SHIPPED - this script does NOT currently reproduce Fig. 5.
# ############################################################################
#
# The entrypoint and the wiring are correct and committed; the underlying
# Contact.domain_adaptation() has two independent pre-existing faults, both
# measured rather than inferred:
#
#   1. THE CONTROLLER NEVER CONVERGES. The forward loop was
#      `while last_target_reached != 1`, and that flag never flips: the PID's
#      position error sits at a CONSTANT 0.0982 against a 0.005 tolerance, and
#      is byte-identical after 120 timesteps - the sensor never moves toward the
#      waypoint. The tip vertex the PID measures is offset from the pose
#      set_up_pose() sets. This is NOT specific to domain adaptation: the
#      TRAINING trajectories show exactly the same 0.0982 and also never reach
#      their target. Data collection survives it only because
#      collect_training_data() loops `for ts in range(...)` and treats the flag
#      as an early exit, so it never depends on it. DA's unbounded `while` is
#      what turned the same condition into an infinite loop (measured: 50 min at
#      ~11 timesteps/second, producing nothing).
#
#      The loop is now bounded by DIFFTACTILE_DA_MAX_TIMESTEPS (default 400) so
#      this script terminates. That makes it safe to run; it does not make the
#      figure correct, since a sensor that never deformed would be overlaid on a
#      real photograph.
#
#   2. THE BACKWARD PASS CANNOT RUN. Past the forward loop, clear_grad_helper()
#      dies with "'NoneType' object has no attribute 'num_active_indices'":
#      system-params.json sets meta.enable_grad = 0, so the .grad fields it
#      fills are never allocated. domain_adaptation() is a differentiable-
#      optimisation loop and needs them.
#
# So reproducing Fig. 5 needs the PID fixed and enable_grad turned on (which in
# turn needs the memory budget checked - gradient fields roughly double it).
# The published figure predates this repository state.
#
# For reference, the shape of a working run: per trajectory a forward pass, a
# backward pass replaying every timestep, and an optimiser step per mini-batch,
# for each of the 4 interactions, repeated num_opt_steps (2) times. The forward
# pass itself is fast - 0.09 s/timestep measured - so the simulator is not the
# bottleneck the runtime suggests. Overlays are written PROGRESSIVELY, each at
# the end of its own trajectory, and update_params() is guarded by
# `if opts > 0`, so the first optimisation step changes no parameters: its
# panels are already the "after domain adaptation" figures.
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
