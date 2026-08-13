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
# EXPECT THIS TO TAKE HOURS, not minutes. It is not a forward-only render: the
# underlying Contact.domain_adaptation() is a differentiable-simulation
# optimisation loop, and per trajectory it does
#
#     forward pass  (step to the target pose, caching every timestep)
#   + backward pass (replay every timestep in reverse, computing gradients)
#   + optimiser step per mini-batch
#
# for each of the 4 interactions, repeated num_opt_steps (2) times. The backward
# pass through the FEM sim costs more than the forward one, and `slide` alone
# runs 327 timesteps. It is also the one figure script that cannot run without
# Taichi and a GPU.
#
# THE OVERLAYS APPEAR PROGRESSIVELY - each is written at the end of its own
# trajectory's forward pass, so da_overlay_press.png exists long before the run
# finishes. And note update_params() is guarded by `if opts > 0`, so the FIRST
# optimisation step changes no parameters: the panels it writes are already the
# "after domain adaptation" figures, using the committed parameters. The second
# step refines parameters that this script does not re-export, so you may stop
# once all four PNGs exist.
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
