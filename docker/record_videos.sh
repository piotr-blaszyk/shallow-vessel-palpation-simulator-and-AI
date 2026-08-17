#!/usr/bin/env bash
#
# Record the README demonstration videos and write them, H.264-compressed, to
# videos/ at the repository root.
#
# Run this FROM THE HOST, from anywhere. It needs the container running
# (docker/docker-run.sh) AND the bare-metal annotator env
# (`micromamba env create -f requirements/annotator-env.yml`), because the
# recordings come from three different tools:
#
#   1. Simulator (in the container, on the GGUI window - which appears on
#      screen while it records; that is the recording):
#        press, twist_x, twist_z, slide with no vein   -> docker/record_da_trajectories.sh
#        slide with the vein embedded (DIFFTACTILE_VEIN=1)
#   2. Annotation viewers (bare metal, docker/annotate_data_bare_metal.sh --record):
#        every frame of every silicone video / meat trial
#   3. Prediction viewer (in the container, docker/view_predictions.sh --record):
#        one video per configuration, the best-of-five model of each
#        (A-to-A: ten held-out trajectories, 7 vessel-present + 3 vessel-absent
#        drawn with a fixed seed and interleaved a a b a a b a a b a - the SAME
#        ten, in the SAME order, as docker/website_vessel_maps.sh maps; the
#        others: every trial)
#
# The GUI tools are driven automatically - one key press per 500 ms of video
# (DIFFTACTILE_RECORD_INTERVAL_MS) - and rendered OFFSCREEN, so no viewer
# window opens anywhere; see difftactile/main/qt_viewer.py "Record mode".
# Everything uses the parameters currently in system-params.json and the
# published best-of-five checkpoints.
#
# Raw recordings land in difftactile/output/videos_raw/ (gitignored); the
# compressed copies (libx264, CRF 30, ~10-30x smaller) in videos/, which IS
# committed and is what the README embeds.
#
# Usage:
#   ./docker/record_videos.sh              # everything
#   ./docker/record_videos.sh --sim        # only the simulator videos
#   ./docker/record_videos.sh --annotations
#   ./docker/record_videos.sh --predictions
#   ./docker/record_videos.sh --compress   # only re-compress what is in videos_raw/
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
CONTAINER_REPO="/workspace/$(basename "${REPO_DIR}")"
RAW_DIR="${REPO_DIR}/difftactile/output/videos_raw"
OUT_DIR="${REPO_DIR}/videos"
CRF="${DIFFTACTILE_VIDEO_CRF:-30}"

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'; }

DO_SIM=0; DO_ANN=0; DO_PRED=0
if [ "$#" -eq 0 ]; then DO_SIM=1; DO_ANN=1; DO_PRED=1; fi
for arg in "$@"; do
    case "${arg}" in
        --sim) DO_SIM=1 ;;
        --annotations) DO_ANN=1 ;;
        --predictions) DO_PRED=1 ;;
        --compress) ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

cd "${REPO_DIR}"
mkdir -p "${RAW_DIR}" "${OUT_DIR}"

in_container() {
    # Run a command inside the container from the repository root.
    docker exec "${CONTAINER_NAME}" bash -lc "cd '${CONTAINER_REPO}' && $*"
}

if [ "${DO_SIM}" -eq 1 ] || [ "${DO_PRED}" -eq 1 ]; then
    if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
        echo "ERROR: container '${CONTAINER_NAME}' is not running (./docker/docker-run.sh)." >&2
        exit 1
    fi
fi

# --- 1. simulator ---------------------------------------------------------------
#
# record_da_trajectories.sh writes to a fresh timestamped directory each run and
# is known to segfault in Taichi GGUI teardown AFTER writing everything (exit
# 139, see CLAUDE.md), so its exit code is ignored and the files are checked
# instead. The four vein-free interactions come from one run; the vessel-present
# slide from a second run with the vein enabled.
record_sim() {
    local vein="$1"; shift    # 0 or 1
    local subset="$1"; shift  # comma list of trajectory names, or empty for all
    local before after
    before=$(ls -1 difftactile/output/da_recordings 2>/dev/null || true)
    in_container "DIFFTACTILE_VEIN=${vein} DIFFTACTILE_RECORD_TRAJECTORIES='${subset}' \
        DIFFTACTILE_VIDEO_SCOPE=per-trajectory ./docker/record_da_trajectories.sh" >&2 || true
    after=$(ls -1 difftactile/output/da_recordings)
    # The one directory that did not exist before this run.
    comm -13 <(echo "${before}") <(echo "${after}") | tail -1
}

if [ "${DO_SIM}" -eq 1 ]; then
    echo "=== 1/3 simulator: press, twist_x, twist_z, slide (no vein) ==="
    run_dir=$(record_sim 0 "")
    for name in press twist_x twist_z; do
        cp "difftactile/output/da_recordings/${run_dir}/${name}.mp4" "${RAW_DIR}/sim_${name}.mp4"
    done
    cp "difftactile/output/da_recordings/${run_dir}/slide.mp4" "${RAW_DIR}/sim_slide_vessel_absent.mp4"
    echo "=== 1/3 simulator: slide (vein embedded) ==="
    run_dir=$(record_sim 1 "slide")
    cp "difftactile/output/da_recordings/${run_dir}/slide.mp4" "${RAW_DIR}/sim_slide_vessel_present.mp4"
fi

# --- 2. annotation viewers (bare metal) --------------------------------------------
if [ "${DO_ANN}" -eq 1 ]; then
    echo "=== 2/3 annotation viewers ==="
    # 1920x1080 frames plus the status bar: 1280x760 fits without letterboxing.
    export DIFFTACTILE_RECORD_SIZE=1280x760
    ./docker/annotate_data_bare_metal.sh --silicone --record "${RAW_DIR}/dataset_annotations_silicone.mp4"
    ./docker/annotate_data_bare_metal.sh --meat --record "${RAW_DIR}/dataset_annotations_meat.mp4"
    unset DIFFTACTILE_RECORD_SIZE
fi

# --- 3. prediction viewer (container) ----------------------------------------------
if [ "${DO_PRED}" -eq 1 ]; then
    echo "=== 3/3 prediction viewer ==="
    raw_rel="difftactile/output/videos_raw"
    ./docker/view_predictions.sh A-to-A --trials interleaved:7:3 --record "${raw_rel}/predictions_sim_to_sim.mp4"
    ./docker/view_predictions.sh A-to-B --record "${raw_rel}/predictions_sim_to_silicone.mp4"
    ./docker/view_predictions.sh A-to-C --record "${raw_rel}/predictions_sim_to_meat.mp4"
    ./docker/view_predictions.sh C-to-B --record "${raw_rel}/predictions_meat_to_silicone.mp4"
fi

# --- compress -----------------------------------------------------------------------
#
# libx264 at CRF 30, yuv420p (the pixel format every browser plays), even
# dimensions, faststart for streaming. Host ffmpeg if there is one, else the
# container's.
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG=(ffmpeg)
else
    FFMPEG=(docker exec "${CONTAINER_NAME}" ffmpeg)
fi
echo "=== compressing -> ${OUT_DIR} ==="
for raw in "${RAW_DIR}"/*.mp4; do
    name=$(basename "${raw}")
    case "${name}" in test_*) continue ;; esac
    "${FFMPEG[@]}" -v error -y -i "${raw}" -an -c:v libx264 -preset slow -crf "${CRF}" \
        -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -movflags +faststart \
        "${OUT_DIR}/${name}"
    printf '  %-40s %6s -> %6s\n' "${name}" \
        "$(du -h "${raw}" | cut -f1)" "$(du -h "${OUT_DIR}/${name}" | cut -f1)"
done
echo "Done. Videos in ${OUT_DIR}"
