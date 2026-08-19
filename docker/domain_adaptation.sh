#!/usr/bin/env bash
#
# Domain adaptation: calibrate the simulator against the real sensor via
# Bayesian optimisation, then validate on the four canonical interactions
# (MANUSCRIPT FIGURE 5).
#
# ONE Bayesian optimisation over the sensor stiffness and the sensor<->vein
# contact stiffness. Every iteration runs TWO slides at the proposed parameters
# - vessel-ABSENT (scored by marker MAE against the real photograph, i.e.
# fidelity) and vessel-PRESENT (scored by how far the vessel holds the sensor
# up, i.e. sensitivity) - and maximises a single objective that trades the two
# off (main.py::domain_adaptation_joint). The winning configuration is then
# VALIDATED, not searched, on all four canonical interactions (press, twist
# about z, twist about x, slide, vessel-free) - those MAEs and overlays are
# what Fig. 4 shows.
#
# The older two-stage design (a vessel-absent BO over the sensor material, then
# a vessel-present BO at the chosen sensor; DIFFTACTILE_DA_MODE=staged) is
# OBSOLETE and has been removed. The published run is the joint one.
#
# NOT DIFFERENTIABLE. An earlier design backpropagated through the Taichi
# simulation to fit these parameters; that was abandoned, and every piece of
# machinery supporting it has been removed from the codebase. BO treats the
# simulator as a black box, so only forward simulation is needed.
#
# THE SEARCH SPACE IS LOG-SCALED: both parameters span a decade (sensor Young's
# modulus in [1e5, 1e6] Pa, sensor<->vein normal stiffness in [1e4, 1e5]) and
# each is mapped "log-transform then min-max" into [0, 1]
# (difftactile/data_analysis/experiment/bo_gp.py, BoGp.LOG_SCALED), so every
# decade gets an equal share of the range the GP searches and a fixed
# MULTIPLICATIVE change means the same thing anywhere in it. BoGp.__init__
# explains how the bounds were chosen.
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
#   bo_joint_results.json        every iteration (both objective terms) + the best
#   iteration_log.csv            the same, one line per iteration, written live
#   final_joint_validation.json  all four interactions at the winning configuration
#   da_overlay_<name>.png        Fig. 4 panels from that validation - simulated
#                                markers RED, real markers GREEN
#   vein_iterNNN_overlay_slide.png   the vessel-present slide, per iteration
#   snapshots/                   rendered 3D frames per iteration (needs a DISPLAY)
#
# The published run is difftactile/output/domain_adaptation_published/joint_bo/;
# docker/alignment_figures.sh redraws Fig. 4 (white background) from it.
#
# Environment:
#   DIFFTACTILE_BO_JOINT_ITERATIONS
#                                 how many parameter sets to try (default 10).
#                                 ~35 s per iteration.
#   DIFFTACTILE_BO_JOINT_RANDOM   how many of those are random before the
#                                 acquisition function takes over (default 5).
#                                 The GP needs a few samples before it can
#                                 propose usefully.
#   DIFFTACTILE_DA_MAX_TIMESTEPS_NO_VEIN
#                                 safety cap on a VESSEL-ABSENT forward pass
#                                 (default 200). That objective is measured at
#                                 the trajectory's apex, so the run has to reach
#                                 it; `slide` is the only trajectory that hits
#                                 the cap at all.
#   DIFFTACTILE_DA_MAX_TIMESTEPS_VEIN
#                                 safety cap on a VESSEL-PRESENT forward pass
#                                 (default 400). That objective short-circuits
#                                 as soon as the vessel passes under the sensor
#                                 centre, so this is only a backstop for a
#                                 vessel that never arrives.
#   DIFFTACTILE_SNAPSHOT_DIR      render the 3D scene periodically to PNGs, to
#                                 check a trajectory by eye. Needs a DISPLAY:
#                                 Taichi GGUI segfaults offscreen in this image.
#   DIFFTACTILE_SNAPSHOT_EVERY    snapshot interval in timesteps (default 20).
#
# A diverged parameter set (the FEM solve blows up and the markers come back
# NaN) is scored as the worst possible value and the search continues - that is
# a legitimate answer from the objective, not a crash.
#
# Measured (published joint_bo run, 10 iterations): best at iteration 5, found
# by the acquisition function - sensor Young's modulus 881 359 Pa, sensor<->vein
# normal stiffness 94 908; vessel-absent slide MAE 12.43 px, validation MAEs
# 11.9 / 15.8 / 14.5 / 12.4 px (press / twist_z / twist_x / slide). The
# inter-marker spacing is ~55 px (2 mm), so these all align to a small fraction
# of one grid step.
#
# ADOPTING THE RESULT is manual and deliberate: the best configuration is
# printed and stored in bo_joint_results.json, but nothing writes it back into
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
echo "Manuscript Fig. 4 panels, written to difftactile/output/:"
for panel in "press:(a) press" \
             "twist_z:(b) twist about the z-axis" \
             "twist_x:(c) twist about the x-axis" \
             "slide:(d) slide"; do
    name="${panel%%:*}"; label="${panel#*:}"
    f="difftactile/output/da_overlay_${name}.png"
    [ -f "${f}" ] && echo "  ${f}   ${label}"
done
