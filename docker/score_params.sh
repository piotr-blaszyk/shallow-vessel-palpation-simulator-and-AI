#!/usr/bin/env bash
#
# Score the CURRENT system-params.json configuration - a sanity check, not a
# search.
#
# Runs ONE joint-objective evaluation at the parameters already on disk:
#
#   1. the vessel-PRESENT slide, stopping when the vessel passes under the
#      sensor centre, and
#   2. the vessel-FREE slide, stopped at that same timestep and measured on that
#      same sensor node,
#
# then reports the two reward terms and the composite:
#
#   vpn  how far the vessel held the sensor up, on [0, 1.5]
#   van  vessel-absent mean marker error / 55 px, on [0, 1]
#   opt  vpn - van        (a diverged sensor scores -1 outright)
#
# THIS IS NOT domain_adaptation.sh. That script PROPOSES parameters and searches
# for good ones over 10 BO iterations; this one proposes nothing, so the number
# it prints is attributable entirely to the configuration in system-params.json.
# Use it to answer "what does the current config score", or to check the
# simulator's behaviour with a setting toggled - where a search would confound
# the answer with its own exploration.
#
# THE SENSOR<->PHANTOM CONTACT PAIR. Pair 0 is OFF by default: the phantom's
# particles are pinned, so it does not deform and resolving the pair changes
# nothing any objective measures. Enable it to check that contact resolution
# works AT ALL - if the sensor visibly deforms against the phantom with the pair
# on, the machinery is sound and a null vessel response needs another
# explanation; if it does not, the contact code itself is suspect.
#
#   ./score_params.sh                  pair 0 off (the default)
#   ./score_params.sh --phantom-contact  pair 0 on, using the coefficients in
#                                        system-params.json (44.8 / 3.9 / 34.0 /
#                                        0.66)
#
# Equivalently, set contact.enable_phantom_contact_pair in system-params.json,
# or DIFFTACTILE_PHANTOM_CONTACT=1/0 in the environment (which overrides it).
#
# Run this INSIDE the container. It needs Taichi and a GPU.
#
# EVERY RUN GETS ITS OWN TIMESTAMPED DIRECTORY under
# difftactile/output/score_params/<YYYYmmdd-HHMMSS>/, containing score.json (all
# the terms plus the parameters that produced them) and the alignment overlays.
#
# Environment:
#   DIFFTACTILE_PHANTOM_CONTACT   1/0, overrides the config seam.
#   DIFFTACTILE_SEED              seed for the trajectory randomisation (42).
#   DIFFTACTILE_VEIN_TRIGGER_RADIUS_PX
#                                 how close the projected vessel must come to
#                                 the sensor centre to trigger the snapshot
#                                 (default 55 px, one inter-marker spacing).
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
        --phantom-contact) export DIFFTACTILE_PHANTOM_CONTACT=1 ;;
        --no-phantom-contact) export DIFFTACTILE_PHANTOM_CONTACT=0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

# No display -> never block on a window; everything is written to disk.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

echo "Scoring the configuration in difftactile/system_params/system-params.json"
echo "(no optimisation - nothing is proposed or tuned)."
python -m difftactile.scripts.script_score_params
