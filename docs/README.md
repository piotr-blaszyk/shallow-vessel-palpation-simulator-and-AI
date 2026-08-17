# Project page

`index.html` is the GitHub Pages site (served from `main`, `/docs`):
https://piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI/

`videos/` are **copies** of `../videos/` (this repository) plus the two data-collection files
from the robot-control repository, because raw.githubusercontent.com serves `.mp4` as
`application/octet-stream` with `nosniff`, which some browsers refuse to play. After
re-recording (`docker/record_videos.sh`) copy the new files here again.

`images/sensor_mesh/` holds the three screenshots of the ViTacTip mesh (side, 45° above, 45° below)
(lossless WebP, ~20 kB each), written directly here by `docker/sensor_mesh_screenshots.sh`.

`images/vessel_maps/` holds the 22 top-view vessel maps (lossless WebP, ~0.1–2 kB each) written
directly here by `docker/website_vessel_maps.sh`: Sim→Sim ×10 (the ten trajectories of the
Sim→Sim prediction video, in the video's order — `docker/vessel_map_sim_test_trajectories.sh`
must have re-simulated them with poses first), Sim→Silicone ×1, Sim→Meat ×10 (the video's trial
order), Meat→Silicone ×1. `manifest.md` beside them names the run each came from. After
re-recording the Sim→Sim video or retraining, re-run both scripts so the page's "same
trajectories, same order" statement stays true.
