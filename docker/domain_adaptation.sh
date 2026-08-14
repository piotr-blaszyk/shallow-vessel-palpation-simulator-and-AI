#!/usr/bin/env bash
#
# Domain adaptation: calibrate the simulator against the real sensor via
# Bayesian optimisation, and produce MANUSCRIPT FIGURE 5.
#
# Each BO iteration proposes a set of material and contact parameters, replays
# the four canonical interactions (press, twist about z, twist about x, slide)
# with them, and scores the set by the MAE between simulated and real marker
# positions at each apex. Lower is better; the GP then proposes the next set.
#
# NOT DIFFERENTIABLE. An earlier design backpropagated through the Taichi
# simulation to fit these parameters; that was abandoned, and every piece of
# machinery supporting it has been removed from the codebase. BO treats the
# simulator as a black box, so only forward simulation is needed.
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# It needs Taichi and a GPU, so there is no bare-metal path. Everything is
# written to disk, so a display is optional.
#
# Usage:
#   ./domain_adaptation.sh          (from the docker/ directory, inside the container)
#
# EVERY RUN GETS ITS OWN TIMESTAMPED DIRECTORY under
# difftactile/output/domain_adaptation/<YYYYmmdd-HHMMSS>/, so running it twice
# accumulates rather than overwrites - the same convention the training pipeline
# uses. Each contains:
#
#   bo_results.json              best configuration + every iteration tried
#   bo_all_params.json           every parameter set, in order
#   bo_all_targets.json          the MAE each one scored
#   bo_convergence.png           MAE per iteration, with the running best
#   best_da_overlay_<name>.png   Fig. 5 panels, from the BEST configuration
#   da_overlay_<name>.png        the same, from the LAST iteration
#   trajectories/iterNNN_<name>.npz
#                                the collected trajectory itself - simulated and
#                                real marker positions, MAE, and the parameters
#                                that produced it, per (iteration, trajectory)
#
# Fig. 5 panels: (a) press, (b) twist about z, (c) twist about x, (d) slide -
# simulated markers RED, real markers GREEN.
#
# Environment:
#   DIFFTACTILE_BO_ITERATIONS     how many parameter sets to try (default:
#                                 contact.num_opt_steps). ~35 s per iteration.
#   DIFFTACTILE_BO_RANDOM         how many of those are random before the
#                                 acquisition function takes over (default 4).
#                                 The GP needs a few samples before it can
#                                 propose usefully.
#   DIFFTACTILE_DA_MAX_TIMESTEPS  safety cap on each forward pass (default 400).
#   DIFFTACTILE_SNAPSHOT_DIR      render the 3D scene periodically to PNGs, to
#                                 check a trajectory by eye. Needs a DISPLAY:
#                                 Taichi GGUI segfaults offscreen in this image.
#   DIFFTACTILE_SNAPSHOT_EVERY    snapshot interval in timesteps (default 20).
#
# A diverged parameter set (the FEM solve blows up and the markers come back
# NaN) is scored as the worst possible value and the search continues - that is
# a legitimate answer from the objective, not a crash.
#
# Measured: 5 iterations improved the aggregated MAE from 13.88 px (0.50 mm) to
# 11.40 px (0.41 mm), with the best set found by the acquisition function. The
# manuscript reports 14 px -> 13.5 px over a longer run. The inter-marker
# spacing is ~55 px (2 mm), so these all align to a small fraction of one grid
# step.
#
# ADOPTING THE RESULT is manual and deliberate: the best configuration is
# printed and stored in bo_results.json, but nothing writes it back into
# system-params.json. Copy it across yourself once you are satisfied with it.
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
