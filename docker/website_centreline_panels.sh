#!/usr/bin/env bash
#
# The vessel-centreline/centroid panels for the PROJECT PAGE (docs/index.html),
# written as lossless WebP to docs/images/centrelines/ (committed).
#
# Same panels, same extraction and the same Hungarian matching as
# analysis/scripts/workflow_centreline_panels.py draws for the poster, with one
# difference: --all-axes, so all sixteen frame-space cells come out the same
# size. The poster wants axes on the left column and the bottom row only, so the
# cells butt up flush inside one figure; a web page scales every cell of its grid
# to one width, and unequal figure shapes would then render the plots at visibly
# different sizes.
#
#   frame space     16 panels, one per (ground-truth vessel count 0-3) x model
#   top-view space   4 panels, one per model
#
# docs/images/centrelines/manifest.md records what each image is and where the
# numbers beside it come from.
#
# Run INSIDE the container, after ./docker/reproduce_analysis.sh (it needs stage
# 1's per-marker predictions). ~1 minute.
#
# Usage:
#   ./docker/website_centreline_panels.sh
#
set -euo pipefail
REPO_DIR="${DIFFTACTILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

PY="${DIFFTACTILE_PYTHON:-python}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
DEST="docs/images/centrelines"

if [ ! -f analysis/results/frame_space_predictions_A-to-A.npz ]; then
    echo "analysis/results/frame_space_predictions_*.npz missing:"
    echo "run ./docker/reproduce_analysis.sh (or restore the Zenodo bundle) first." >&2
    exit 1
fi

"${PY}" analysis/scripts/workflow_centreline_panels.py --all-axes --out-dir "${STAGE}"

mkdir -p "${DEST}"
"${PY}" - "${STAGE}" "${DEST}" <<'PYEOF'
"""Convert the staged PNG panels to lossless WebP, which is what the page serves."""
import sys
from pathlib import Path
from PIL import Image

stage, dest = Path(sys.argv[1]), Path(sys.argv[2])
for src in sorted(stage.glob("cl_*.png")):
    if src.stem == "cl_legend":
        continue          # the page draws the legend in HTML, so it follows the reader's theme
    out = dest / (src.stem + ".webp")
    Image.open(src).convert("RGB").save(out, "WEBP", lossless=True, quality=100, method=6)
    print(f"  {out}  {out.stat().st_size // 1024} KB")
PYEOF

echo
echo "Written to ${DEST}/ - commit them together with docs/index.html."
