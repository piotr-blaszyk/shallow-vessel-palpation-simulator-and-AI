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
    "difftactile/output/training_data/pickle_20250901_220921_reordered_dense"
    "difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense"
    "difftactile/manual_or_experimental_data/meat_training_data/clean"
    "saved_models_sim/final_segmentation_model_gnn_sim.pt"
    "saved_models_meat/final_segmentation_model_gnn_meat.pt"
    "difftactile/output/test_loader_gnn_sim.pickle"
    "difftactile/output/test_loader_gnn_meat.pickle"
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
)

verify() {
    local missing=0
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
    echo
    if [ "${missing}" -gt 0 ]; then
        echo "${missing} expected path(s) missing — the ML entrypoints will fail with FileNotFoundError."
        return 1
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

# Where a path used to live before the naming cleanup. Bundles published prior
# to that rename -- including the current Zenodo record -- still carry the old
# layout, so each expected path falls back to its legacy name and is restored
# under the new one. Keep these entries as long as such a bundle is downloadable.
legacy_path_for() {
    case "$1" in
        "difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense")
            echo "difftactile/manual_or_experimental_data/endgame/20250901-131547_dense" ;;
        "difftactile/manual_or_experimental_data/meat_training_data/clean")
            echo "difftactile/manual_or_experimental_data/iros_training_data/clean" ;;
        "saved_models_sim/final_segmentation_model_gnn_sim.pt")
            echo "saved_models_icra/final_segmentation_model_gnn_icra.pt" ;;
        "saved_models_meat/final_segmentation_model_gnn_meat.pt")
            echo "saved_models_iros/final_segmentation_model_gnn_iros.pt" ;;
        "difftactile/output/test_loader_gnn_sim.pickle")
            echo "difftactile/output/test_loader_gnn_icra.pickle" ;;
        "difftactile/output/test_loader_gnn_meat.pickle")
            echo "difftactile/output/test_loader_gnn_iros.pickle" ;;
        "difftactile/output/marker_locations_ordered.npz")
            echo "difftactile/output/iros_marker_locations_ordered.npz" ;;
        *) echo "" ;;
    esac
}

# Copy each expected path into place. Existing files are replaced; anything the
# bundle does not carry is left untouched, so this is safe to re-run.
for rel in "${EXPECTED[@]}"; do
    src="${BUNDLE}/${rel}"
    dest="${REPO_DIR}/${rel}"
    if [ ! -e "${src}" ]; then
        legacy_rel="$(legacy_path_for "${rel}")"
        if [ -n "${legacy_rel}" ] && [ -e "${BUNDLE}/${legacy_rel}" ]; then
            src="${BUNDLE}/${legacy_rel}"
            echo "  (pre-rename bundle: ${legacy_rel} -> ${rel})"
        fi
    fi
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

verify
