#!/usr/bin/env bash
#
# IoU, AUROC and average precision for every canonical scenario, in one pass.
#
# THIS IS THE SCRIPT THAT REPORTS THE PAPER'S IoU TABLE: foreground IoU (vessel
# present) and background IoU (vessel absent), over the frame-by-frame marker
# predictions - NOT the reprojected bird's-eye phantom map, whose separate IoU
# comes from vessel_map.sh. Each is |pred AND true| / |pred OR true| over that
# class, so "foreground" names the class, not one side of the comparison.
# Background IoU is always the flattering number, since the negative class is
# ~89% of nodes on silicone and ~93% on meat - read foreground as the result.
#
# Unlike AUROC and AP, IoU depends on the decision threshold, so it carries every
# caveat about that choice. Use --seeds N to get mean +/- std instead of one run.
#
# A short wrapper for:
#     python -m difftactile.scripts.script_auroc_all_scenarios [args]
#
# Run this INSIDE the container (see docker/docker-run.sh + docker/docker-connect.sh).
# The figures and the Markdown table are written to disk, so a display is optional.
#
# Usage:
#   ./docker/score_all_scenarios.sh [config ...] [--pretrained|--retrained]
#   ./docker/score_all_scenarios.sh --seeds N [config ...]
#   ./docker/score_all_scenarios.sh --replot [SWEEP_DIR|TIMESTAMP]
#
# Configurations (A = simulation, B = real silicone, C = real meat):
#   A-to-A    train on simulation, test on held-out simulation
#   A-to-B    train on simulation, test on silicone
#   C-to-B    train on meat,       test on silicone
#   A-to-C    train on simulation, test on meat
# With none given, all four are scored.
#
# Weights:
#   --pretrained  only the published Zenodo checkpoints
#   --retrained   only the checkpoints a local `--train` run wrote
# With neither given, BOTH are scored - the four configurations x two weight
# sources are the eight scenarios. Any whose checkpoint is absent is skipped with
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
# --seeds N: THE SEED SWEEP. Trains each configuration from scratch N times, once
# per seed (0..N-1), and reports mean, standard deviation and range of AUROC and
# AP - plus every per-seed value, so nothing hides behind a summary statistic.
#
# This is the number to report. A single training run on this project reflects
# its seed as much as it reflects the model: measured on C-to-B, seven seeds gave
# AUROC 0.604-0.779, a spread WIDER than the differences between the three
# configurations. The meat training set is 139 clips, small enough that which
# subset a run favours dominates.
#
# Read the spread, not the best row - selecting the highest-scoring seed is
# fitting to the test set exactly as tuning a decision threshold there would be.
# When comparing configurations, compare distributions; a gap smaller than the
# seed spread is not evidence of anything.
#
# Note this TRAINS, so it is far slower than the default scoring pass, and it
# leaves only the last seed's checkpoint (re-training a configuration overwrites
# its *_retrained artifacts in place). Sweeps characterise the spread; train a
# single seed normally when you want a checkpoint to keep. Evaluation-only
# scoring ignores --seeds, since a fixed published checkpoint has no seed.
#
# Outputs:
#   AUROC_RESULTS.md                                  the summary table
#   difftactile/output/roc_curves/roc_curve_<config>_<weights>.pdf
#   difftactile/output/pr_curves/pr_curve_<config>_<weights>.pdf
#
# With --seeds N the sweep section is written into the SAME AUROC_RESULTS.md,
# below the scenario table rather than replacing it, so the two can coexist.
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
#   ./docker/score_all_scenarios.sh --seeds 5 C-to-B    # 5-seed sweep of one config
#   ./docker/score_all_scenarios.sh --seeds 5           # 5-seed sweep of all three
#   ./docker/score_all_scenarios.sh --replot            # redraw the published sweep's curves
#
# --replot [SWEEP]: redraws ONLY the mean PR/ROC curves and the standalone
#   shared legend + x labels (mean_{pr,roc}_curve_<config>.pdf, curve_legend.pdf,
#   xlabel_*.pdf) of
#   an existing sweep directory from its saved per-seed scores - no training, no
#   inference, seconds rather than hours. Default SWEEP is files.published_sweep.
#   Use it after changing the figure styling in cnn/curve_plots.py.
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'
}

NUM_SEEDS=""
REPLOT=""
REPLOT_SWEEP=""
CONFIGS=()
PASSTHROUGH=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        A-to-A|A-to-B|C-to-B|A-to-C)
            CONFIGS+=("$1"); PASSTHROUGH+=("$1") ;;
        --pretrained|--retrained)
            PASSTHROUGH+=("$1") ;;
        --seeds)
            shift
            if [ "$#" -eq 0 ]; then
                echo "ERROR: --seeds needs a count, e.g. --seeds 5" >&2; exit 1
            fi
            NUM_SEEDS="$1"
            ;;
        --seeds=*)
            NUM_SEEDS="${1#*=}" ;;
        --replot)
            REPLOT=1
            # Optional sweep dir / timestamp follows, unless the next word is a flag.
            if [ "$#" -gt 1 ] && [ "${2#-}" = "$2" ]; then
                shift; REPLOT_SWEEP="$1"
            fi
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: $1" >&2; echo; usage; exit 1 ;;
    esac
    shift
done

if [ -n "${NUM_SEEDS}" ]; then
    case "${NUM_SEEDS}" in
        ''|*[!0-9]*) echo "ERROR: --seeds needs a positive integer, got '${NUM_SEEDS}'" >&2; exit 1 ;;
    esac
    if [ "${NUM_SEEDS}" -lt 1 ]; then
        echo "ERROR: --seeds needs a positive integer, got '${NUM_SEEDS}'" >&2; exit 1
    fi
fi

# No display -> never block on a window; the files on disk are the output.
if [ -z "${DISPLAY:-}" ]; then
    export DIFFTACTILE_HEADLESS="${DIFFTACTILE_HEADLESS:-1}"
fi

if [ -n "${REPLOT}" ]; then
    # Replot mode: restyle an existing sweep's figures from its saved scores.
    python -m difftactile.scripts.script_seed_sweep --replot ${REPLOT_SWEEP:+"${REPLOT_SWEEP}"}
    exit 0
fi

if [ -n "${NUM_SEEDS}" ]; then
    # Sweep mode: TRAINS each configuration NUM_SEEDS times. Weight-source flags
    # are meaningless here (every model is trained fresh), so they are not passed.
    echo "Seed sweep: training ${CONFIGS[*]:-all three configurations} under ${NUM_SEEDS} seed(s)."
    echo "This trains from scratch per seed, so it is much slower than plain scoring."
    python -m difftactile.scripts.script_seed_sweep "${NUM_SEEDS}" "${CONFIGS[@]}"
else
    echo "Scoring scenarios (AUROC + average precision, both threshold-free)."
    python -m difftactile.scripts.script_auroc_all_scenarios "${PASSTHROUGH[@]}"
fi

echo
echo "Written:"
[ -f AUROC_RESULTS.md ] && echo "  AUROC_RESULTS.md                    the summary table"
for d in difftactile/output/roc_curves difftactile/output/pr_curves; do
    if [ -d "${d}" ]; then
        n=$(find "${d}" -name '*.pdf' | wc -l)
        echo "  ${d}/  (${n} PDF(s))"
    fi
done
