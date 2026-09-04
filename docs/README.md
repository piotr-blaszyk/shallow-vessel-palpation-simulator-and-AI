# Project page

`index.html` is the GitHub Pages site (served from `main`, `/docs`):
https://piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI/

`videos/` are **copies** of `../videos/` (this repository) plus the two data-collection files
from the robot-control repository, because raw.githubusercontent.com serves `.mp4` as
`application/octet-stream` with `nosniff`, which some browsers refuse to play. After
re-recording (`docker/record_videos.sh`) copy the new files here again. The seven `dr_*.mp4`
domain-randomisation slides are written here directly by
`docker/record_domain_randomisation_videos.sh` (heading ±15°/0° from a top-down camera, the 2 × 2
grid of sensor–vessel stiffness × damping range extremes from the side view).

`images/sensor_mesh/` holds the three screenshots of the ViTacTip mesh (side, 45° above, 45° below)
(lossless WebP, ~20 kB each), written directly here by `docker/sensor_mesh_screenshots.sh`.

`images/vessel_maps/` holds the 22 top-view vessel maps (lossless WebP, ~0.1–2 kB each) written
directly here by `docker/website_vessel_maps.sh`: Sim→Sim ×10 (the ten trajectories of the
Sim→Sim prediction video, in the video's order — `docker/vessel_map_sim_test_trajectories.sh`
must have re-simulated them with poses first), Sim→Silicone ×1, Sim→Meat ×10 (the video's trial
order), Meat→Silicone ×1. `manifest.md` beside them names the run each came from. After
re-recording the Sim→Sim video or retraining, re-run both scripts so the page's "same
trajectories, same order" statement stays true.

`images/centrelines/` holds the 20 vessel-centreline/centroid panels (lossless WebP, ~17–62 kB
each) written directly here by `docker/website_centreline_panels.sh`: 16 frame-space panels
(ground-truth vessel count 0–3 × the four models) and 4 top-view panels, one per model. They are
the same panels the poster's workflow diagram carries, drawn with `--all-axes` so that all
sixteen come out the same size — a web grid scales every cell to one width, and the poster's
flush layout (axes on the left column and bottom row only) would render them at visibly
different sizes here. `manifest.md` beside them records which frame each cell shows, why it was
chosen and the numbers quoted in its caption. The script needs
`analysis/results/frame_space_predictions_*.npz`, so run `docker/reproduce_analysis.sh` (or
restore the Zenodo bundle) first; re-run both after retraining, or the page's numbers go stale.

`poster-motivation-bibliography/` is a standalone LaTeXML page: the poster's
"Vessel-detection modalities" table with every claim cited. It is light-theme only.
