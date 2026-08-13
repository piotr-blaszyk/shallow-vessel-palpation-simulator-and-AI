#!/usr/bin/env bash
#
# Threshold-free ranking metrics for every canonical scenario, in one pass.
#
# A short wrapper for:
#     python -m difftactile.scripts.script_auroc_all_scenarios [args]
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# The figures and the Markdown table are written to disk, so a display is optional.
#
# Usage:
#   ./docker/score_all_scenarios.sh [config ...] [--pretrained|--retrained]
#
# Configurations (A = simulation, B = real silicone, C = real meat):
#   A-to-B    train on simulation, test on silicone
#   C-to-B    train on meat,       test on silicone
#   A-to-C    train on simulation, test on meat
# With none given, all three are scored.
#
# Weights:
#   --pretrained  only the published Zenodo checkpoints
#   --retrained   only the checkpoints a local `--train` run wrote
# With neither given, BOTH are scored - the three configurations x two weight
# sources are the six scenarios. Any whose checkpoint is absent is skipped with
# a note rather than failing the run, so this is safe on a fresh restore where
# nothing has been retrained yet.
#
# WHAT IT REPORTS, and why these two metrics. Both AUROC and average precision
# (AP) are threshold-free and ranking-based: they read only the ORDER of the
# predicted probabilities, never their absolute scale. No decision threshold is
# chosen anywhere. That is the point - this is a sim-to-real project, and the
# first thing to shift when a model crosses a domain gap is the output scale,
# not the ranking. A single-threshold score (IoU, F1) confounds the two.
#
# Both are reported because each is blind to something the other sees:
#
#   AUROC  normalises false positives by the total negative count. This problem
#          is heavily imbalanced (~5% positive), so a large absolute number of
#          false alarms barely moves it - AUROC can look reassuring where
#          precision is poor. Its baseline is always 0.5, so it is the number
#          that compares across papers.
#   AP     ignores true negatives entirely, so the negative majority cannot
#          flatter it. Its baseline is the POSITIVE RATE, so it must be read
#          against that - which is why the table gives both the chance level and
#          the lift (AP / chance) beside it.
#
# Outputs:
#   AUROC_RESULTS.md                                  the summary table
#   difftactile/output/roc_curves/roc_curve_<config>_<weights>.pdf
#   difftactile/output/pr_curves/pr_curve_<config>_<weights>.pdf
#
# Both figure types share their styling (threshold colourmap, marked operating
# points), so a ROC and a PR panel can sit side by side in a manuscript. The PR
# figure additionally draws its chance baseline as a dashed line, because a PR
# curve cannot be read without it: the same curve is excellent on a 1% positive
# set and worthless on a 50% one.
#
# Examples:
#   ./docker/score_all_scenarios.sh                     # all six scenarios
#   ./docker/score_all_scenarios.sh --pretrained        # published checkpoints only
#   ./docker/score_all_scenarios.sh A-to-B              # one config, both weightings
#   ./docker/score_all_scenarios.sh A-to-B --pretrained # one scenario
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

for arg in "$@"; do
    case "${arg}" in
        A-to-B|C-to-B|A-to-C|--pretrained|--retrained) ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

# No display -> never block on a window; the files on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

echo "Scoring scenarios (AUROC + average precision, both threshold-free)."
python -m difftactile.scripts.script_auroc_all_scenarios "$@"

echo
echo "Written:"
[ -f AUROC_RESULTS.md ] && echo "  AUROC_RESULTS.md                    the summary table"
for d in difftactile/output/roc_curves difftactile/output/pr_curves; do
    if [ -d "${d}" ]; then
        n=$(find "${d}" -name '*.pdf' | wc -l)
        echo "  ${d}/  (${n} PDF(s))"
    fi
done
