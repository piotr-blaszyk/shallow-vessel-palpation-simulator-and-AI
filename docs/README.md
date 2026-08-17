# Project page

`index.html` is the GitHub Pages site (served from `main`, `/docs`):
https://piotr-blaszyk.github.io/shallow-vessel-palpation-simulator-and-AI/

`videos/` are **copies** of `../videos/` (this repository) plus the two data-collection files
from the robot-control repository, because raw.githubusercontent.com serves `.mp4` as
`application/octet-stream` with `nosniff`, which some browsers refuse to play. After
re-recording (`docker/record_videos.sh`) copy the new files here again.

`images/sensor_mesh/` holds the three screenshots of the ViTacTip mesh (side, 45° above, 45° below)
(lossless WebP, ~20 kB each), written directly here by `docker/sensor_mesh_screenshots.sh`.
