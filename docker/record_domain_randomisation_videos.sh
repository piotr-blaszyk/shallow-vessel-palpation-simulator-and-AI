#!/usr/bin/env bash
#
# Record the DOMAIN-RANDOMISATION demonstration videos for the project page:
# one vessel-PRESENT slide per configuration, from the simulator's own camera,
# H.264-compressed into videos/ (committed) and copied to docs/videos/.
#
# Dataset collection (main.py::collect_training_data) randomises exactly three
# things per trial - the slide heading (+-15 deg about the nominal crossing
# direction) and the sensor<->vessel normal stiffness k_n ~ U(5e3, 5e4) and
# normal damping c_n ~ U(0, 100). The seven configurations below sweep each
# one while holding the other two at their midpoint (heading 0 deg,
# k_n = 27 500, c_n = 50):
#
#   heading -15 / 0 / +15 deg    (k_n 27 500, c_n 50)        -> 3 videos
#   k_n 5e3 / 5e4                (heading 0, c_n 50)         -> 2 videos
#   c_n 0 / 100                  (heading 0, k_n 27 500)     -> 2 videos
#
# Everything else is as in dataset collection: the adopted sensor Young's
# modulus from system-params.json, the vessel embedded, no sensor<->phantom
# contact. Each configuration is one run of docker/record_da_trajectories.sh
# (in the container, on the GGUI window - it appears on screen while it
# records) with the three pins exposed by main.py::record_da_trajectories_main:
# DIFFTACTILE_SLIDE_HEADING_DEG, DIFFTACTILE_VEIN_NORMAL_STIFFNESS,
# DIFFTACTILE_VEIN_NORMAL_DAMPING.
#
# Run this FROM THE HOST, from anywhere, with the container running
# (docker/docker-run.sh). Raw recordings land in
# difftactile/output/videos_raw/ (gitignored); the compressed copies
# (libx264, CRF 30) in videos/ and docs/videos/ as dr_<config>.mp4.
#
# Usage:
#   ./docker/record_domain_randomisation_videos.sh             # record + compress
#   ./docker/record_domain_randomisation_videos.sh --compress  # only re-compress
#   ./docker/record_domain_randomisation_videos.sh --only=heading_m15,heading_0,heading_p15
#
set -euo pipefail

REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="${VESSEL_PALPATION_CONTAINER:-vessel-palpation}"
CONTAINER_REPO="/workspace/$(basename "${REPO_DIR}")"
RAW_DIR="${REPO_DIR}/difftactile/output/videos_raw"
OUT_DIR="${REPO_DIR}/videos"
DOCS_DIR="${REPO_DIR}/docs/videos"
CRF="${DIFFTACTILE_VIDEO_CRF:-30}"

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^# \?//'; }

DO_RECORD=1
ONLY=""
for arg in "$@"; do
    case "${arg}" in
        --compress) DO_RECORD=0 ;;
        --only=*) ONLY="${arg#--only=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unrecognised argument: ${arg}" >&2; echo; usage; exit 1 ;;
    esac
done

cd "${REPO_DIR}"
mkdir -p "${RAW_DIR}" "${OUT_DIR}" "${DOCS_DIR}"

# name  heading_deg  k_n     c_n  camera   -- the midpoints are 0 / 27500 / 50
# camera: "top" = straight above the phantom, +y up the image, so the slide
# runs bottom -> top and the heading is visible; "side" = the usual view.
CONFIGS=(
    "heading_m15   -15  27500  50  top"
    "heading_0       0  27500  50  top"
    "heading_p15    15  27500  50  top"
    "stiffness_5e3   0   5000  50  side"
    "stiffness_5e4   0  50000  50  side"
    "damping_0       0  27500   0  side"
    "damping_100     0  27500 100  side"
)

in_container() {
    docker exec "${CONTAINER_NAME}" bash -lc "cd '${CONTAINER_REPO}' && $*"
}

if [ "${DO_RECORD}" -eq 1 ]; then
    if [ -z "$(docker ps -q -f "name=^/${CONTAINER_NAME}$")" ]; then
        echo "ERROR: container '${CONTAINER_NAME}' is not running (./docker/docker-run.sh)." >&2
        exit 1
    fi
    for cfg in "${CONFIGS[@]}"; do
        read -r name heading kn cn camera <<< "${cfg}"
        if [ -n "${ONLY}" ] && ! grep -q "\(^\|,\)${name}\(,\|$\)" <<< "${ONLY}"; then continue; fi
        echo "=== ${name}: heading ${heading} deg, k_n ${kn}, c_n ${cn}, camera ${camera} ==="
        before=$(ls -1 difftactile/output/da_recordings 2>/dev/null || true)
        # record_da_trajectories.sh is known to segfault in GGUI teardown AFTER
        # writing everything (exit 139), so the exit code is ignored and the
        # file is checked instead.
        in_container "DIFFTACTILE_VEIN=1 DIFFTACTILE_RECORD_TRAJECTORIES=slide \
            DIFFTACTILE_VIDEO_SCOPE=per-trajectory \
            DIFFTACTILE_SLIDE_HEADING_DEG=${heading} \
            DIFFTACTILE_VEIN_NORMAL_STIFFNESS=${kn} \
            DIFFTACTILE_VEIN_NORMAL_DAMPING=${cn} \
            DIFFTACTILE_CAMERA_VIEW=${camera} \
            ./docker/record_da_trajectories.sh" >&2 || true
        after=$(ls -1 difftactile/output/da_recordings)
        run_dir=$(comm -13 <(echo "${before}") <(echo "${after}") | tail -1)
        src="difftactile/output/da_recordings/${run_dir}/slide.mp4"
        if [ -z "${run_dir}" ] || [ ! -s "${src}" ]; then
            echo "ERROR: no slide.mp4 produced for ${name}" >&2
            exit 1
        fi
        cp "${src}" "${RAW_DIR}/dr_${name}.mp4"
        cp "difftactile/output/da_recordings/${run_dir}/recording.json" "${RAW_DIR}/dr_${name}.json"
    done
fi

# --- compress (same settings as record_videos.sh) ---------------------------------
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG=(ffmpeg)
else
    FFMPEG=(docker exec "${CONTAINER_NAME}" ffmpeg)
fi
echo "=== compressing -> ${OUT_DIR} and ${DOCS_DIR} ==="
for cfg in "${CONFIGS[@]}"; do
    read -r name _ _ _ _ <<< "${cfg}"
    raw="${RAW_DIR}/dr_${name}.mp4"
    [ -s "${raw}" ] || { echo "  missing ${raw}, skipped"; continue; }
    "${FFMPEG[@]}" -v error -y -i "${raw}" -an -c:v libx264 -preset slow -crf "${CRF}" \
        -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -movflags +faststart \
        "${OUT_DIR}/dr_${name}.mp4"
    cp "${OUT_DIR}/dr_${name}.mp4" "${DOCS_DIR}/dr_${name}.mp4"
    printf '  %-28s %6s -> %6s\n' "dr_${name}.mp4" \
        "$(du -h "${raw}" | cut -f1)" "$(du -h "${OUT_DIR}/dr_${name}.mp4" | cut -f1)"
done
echo "Done."
