#!/usr/bin/env bash
#
# Restore the Zenodo data bundle into this repository.
#
# The bundle mirrors the repository layout, so restoring is an unpack plus a
# verification pass. Run it once after cloning and downloading the archive.
#
# Usage:
#   ./data/restore_data.sh shallow-vessel-palpation-data.tar.gz     # from a downloaded tarball
#   ./data/restore_data.sh /path/to/shallow-vessel-palpation-data   # from an unpacked directory
#   ./data/restore_data.sh --verify                    # only check what is present
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit "${1:-0}"
}

# Paths the pipeline expects to exist once the bundle is restored.
EXPECTED=(
    "difftactile/output/training_data/pickle_20260814_191137_reordered_dense"
    "difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense"
    "difftactile/manual_or_experimental_data/meat_training_data/clean"
    "saved_models_sim/final_segmentation_model_gnn_sim.pt"
    "saved_models_meat/final_segmentation_model_gnn_meat.pt"
    "difftactile/output/test_loader_gnn_sim.pickle"
    "difftactile/output/test_loader_gnn_meat.pickle"
    # The published five-seed sweep (every seed's checkpoint + pickle); the
    # published checkpoints above are its best-of-five instances. Pinned by
    # files.published_sweep in system-params.json.
    "saved_models_sweeps/20260815-194045"
    # Legacy (pre-2026-08-15) models: only behind the accepted manuscript's
    # Fig. 8 / Table 4. See saved_models_legacy/README.md.
    "saved_models_legacy/sim/final_segmentation_model_gnn_sim.pt"
    "saved_models_legacy/sim/test_loader_gnn_sim.pickle"
    "saved_models_legacy/meat/final_segmentation_model_gnn_meat.pt"
    "saved_models_legacy/meat/test_loader_gnn_meat.pickle"
    # The one simulated slide (with poses) behind the Sim->Sim vessel map.
    "difftactile/output/vessel_map_sim/raw"
    "difftactile/output/vessel_map_sim/raw_reordered_dense"
    # The ten video trajectories with poses, behind the project page's Sim->Sim maps.
    "difftactile/output/vessel_map_sim/test_trajectories"
    # Sensor geometry / marker layout — prerequisites for the simulator.
    "difftactile/output/base-graph-connectivity.npz"
    "difftactile/output/marker_locations_ordered.npz"
    "difftactile/output/init-marker-positions.npz"
    "difftactile/output/init-marker-positions.pkl"
    "difftactile/output/gmsh_mesh_vitactip.pkl"
    "difftactile/output/gmsh_mesh_vein.pkl"
    "difftactile/output/vitactip_mesh.npz"
    "difftactile/output/edge_lengths.pkl"
    "difftactile/output/tactile_sensor.f2v.pkl"
    "difftactile/output/phantom_points.npz"
    "difftactile/output/vein_points.npz"
    "difftactile/output/is_fixed_layer.npz"
    "difftactile/output/grid_node_v0_mask.npz"
    "difftactile/output/initial_vertex_positions_undeformed.pkl"
    "difftactile/output/vitactip_points_E.pkl"
    "difftactile/output/marker_tracker/domain-adaptation-vascular-markers"
    # The published domain-adaptation BO run: alignment_figures.sh reads its
    # marker caches / MAEs, and it is the record behind the adopted parameters.
    "difftactile/output/domain_adaptation_published"
)

# Restored exactly like EXPECTED, but their absence is a WARNING rather than an
# error: stage 1 of docker/reproduce_analysis.sh regenerates them in about a
# minute. That stage is the only one that needs a GPU, so shipping its output is
# what lets a CPU-only machine rebuild every analysis figure and number (stages
# 2-12); a bundle downloaded before these were added simply needs that one stage
# run first. Listed file by file, not as the directory, because analysis/results/
# also holds git-tracked JSON and so exists in a fresh clone - verifying the
# directory would verify nothing.
REGENERABLE=(
    "analysis/results/frame_space_predictions_A-to-A.npz"
    "analysis/results/frame_space_predictions_A-to-B.npz"
    "analysis/results/frame_space_predictions_A-to-C.npz"
    "analysis/results/frame_space_predictions_C-to-B.npz"
)

# Restored when present, but not required for anything to run: the manuscript's
# figures and tables, shipped as the bundle's one "generable but included"
# exception (see data/MANIFEST.md). Lands in difftactile/output/manuscript_artifacts/.
OPTIONAL=(
    "manuscript_artifacts:difftactile/output/manuscript_artifacts"
)

verify() {
    local missing=0
    local absent=0
    echo
    echo "Verifying restored data in ${REPO_DIR}:"
    for rel in "${EXPECTED[@]}"; do
        local target="${REPO_DIR}/${rel}"
        # -e follows symlinks, which is what we want: the author's checkout wires
        # some of these up as links into an external data directory.
        if [ -e "${target}" ]; then
            printf '  OK      %s\n' "${rel}"
        else
            printf '  MISSING %s\n' "${rel}"
            missing=$((missing + 1))
        fi
    done
    for rel in "${REGENERABLE[@]}"; do
        if [ -e "${REPO_DIR}/${rel}" ]; then
            printf '  OK      %s\n' "${rel}"
        else
            printf '  ABSENT  %s  (regenerable)\n' "${rel}"
            absent=$((absent + 1))
        fi
    done
    echo
    if [ "${missing}" -gt 0 ]; then
        echo "${missing} expected path(s) missing — the ML entrypoints will fail with FileNotFoundError."
        return 1
    fi
    if [ "${absent}" -gt 0 ]; then
        echo "All required data present."
        echo "${absent} regenerable path(s) absent: run './docker/reproduce_analysis.sh --only 1'"
        echo "(needs a GPU, about a minute) before the CPU-only stages 2-12."
        return 0
    fi
    echo "All expected data present."
    return 0
}

case "${1:-}" in
    -h|--help) usage 0 ;;
    --verify)  verify; exit $? ;;
    "")        echo "ERROR: no bundle given." >&2; echo; usage 1 ;;
esac

SRC="$1"
if [ ! -e "${SRC}" ]; then
    echo "ERROR: not found: ${SRC}" >&2
    exit 1
fi

STAGE=""
# `return 0` matters: a trap's exit status becomes the script's, and the
# `[ -n "${STAGE}" ]` test is false (status 1) whenever we restored from an
# already-unpacked directory rather than a tarball. Without it, a successful
# directory restore exits 1 and breaks any caller using `&&` or `set -e`.
cleanup() { [ -n "${STAGE}" ] && rm -rf "${STAGE}"; return 0; }
trap cleanup EXIT

if [ -f "${SRC}" ]; then
    echo "Unpacking ${SRC} ..."
    STAGE="$(mktemp -d)"
    tar -xzf "${SRC}" -C "${STAGE}"
    # The archive contains a single top-level shallow-vessel-palpation-data/ directory
    # (older bundles used difftactile-data/; the find fallback below covers those).
    BUNDLE="${STAGE}/shallow-vessel-palpation-data"
    [ -d "${BUNDLE}" ] || BUNDLE="$(find "${STAGE}" -maxdepth 1 -mindepth 1 -type d | head -1)"
elif [ -d "${SRC}" ]; then
    BUNDLE="${SRC}"
else
    echo "ERROR: ${SRC} is neither a file nor a directory." >&2
    exit 1
fi

if [ ! -d "${BUNDLE}" ]; then
    echo "ERROR: could not locate bundle contents inside ${SRC}" >&2
    exit 1
fi

echo "Restoring from ${BUNDLE}"
echo "            to ${REPO_DIR}"
echo

# Copy each expected path into place. Existing files are replaced; anything the
# bundle does not carry is left untouched, so this is safe to re-run.
for rel in "${EXPECTED[@]}" "${REGENERABLE[@]}"; do
    src="${BUNDLE}/${rel}"
    dest="${REPO_DIR}/${rel}"
    if [ ! -e "${src}" ]; then
        echo "  not in bundle (skipped): ${rel}"
        continue
    fi
    mkdir -p "$(dirname "${dest}")"
    # If the destination is a symlink (the author's layout), remove it first so
    # we write real files rather than following the link outside the repo.
    [ -L "${dest}" ] && rm -f "${dest}"
    if [ -d "${src}" ]; then
        mkdir -p "${dest}"
        cp -r "${src}/." "${dest}/"
    else
        cp "${src}" "${dest}"
    fi
    echo "  restored: ${rel}"
done

# Optional extras: "<bundle path>:<destination>" pairs, skipped silently if absent.
for pair in "${OPTIONAL[@]}"; do
    src="${BUNDLE}/${pair%%:*}"
    dest="${REPO_DIR}/${pair#*:}"
    if [ -d "${src}" ]; then
        mkdir -p "${dest}"
        cp -r "${src}/." "${dest}/"
        echo "  restored (optional): ${pair%%:*} -> ${pair#*:}"
    fi
done

verify
