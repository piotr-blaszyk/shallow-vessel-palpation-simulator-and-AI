"""Bird's-eye vessel maps for the project page (docs/index.html).

Runs the vessel-map job (`vessel_map.run`) for the four configurations exactly
as the page shows them, then turns each map's raw confusion overlay into a
small lossless WebP under docs/images/vessel_maps/:

    Sim -> Sim          the TEN held-out trajectories of the Sim -> Sim
                        prediction video (DIFFTACTILE_SIM_MAP_TRAJECTORIES=test;
                        needs docker/vessel_map_sim_test_trajectories.sh once),
                        one map each, in the video's order   -> 10 files
    Sim -> Silicone     the one silicone map (video ground truth)  -> 1 file
    Sim -> Meat         one map per meat trial, sorted as the
                        prediction video plays them             -> 10 files
    Meat -> Silicone    the one silicone map (video ground truth)  -> 1 file

Every map is the r = 0 mm confusion overlay (ground truth NOT grown), on the
project colour scheme (Visualisation.CONFUSION_COLOURS_RGB): green = both say
vessel, red = missed vessel, blue = false alarm, black = neither. The maps are
1 px = 1 mm; they are upscaled by an integer factor (WEB_SCALE, nearest
neighbour, so every map pixel is a flat block) and written as LOSSLESS WebP -
flat colours and hard edges compress far better losslessly than lossy, and
lossy would smear the single-pixel marker points (a few kB per map).

`manifest.md` beside the images records which run each came from, the chosen
threshold, and the map order, so the page's captions can be checked against it.

Entrypoint: docker/website_vessel_maps.sh (inside the container).
"""

import json
import os
import shutil

import cv2
from PIL import Image

from difftactile.data_analysis.experiment import vessel_map
from difftactile.main.paths import repo_path

# Where the page reads the maps from.
WEB_DIR = "docs/images/vessel_maps"
# Integer nearest-neighbour upscale of the 1 px = 1 mm maps for the page.
WEB_SCALE = 5

# (configuration, ground-truth source, file-name prefix, DIFFTACTILE_SIM_MAP_TRAJECTORIES)
PAGE_RUNS = [
    ("A-to-A", "simulator", "sim_to_sim", "test"),
    ("A-to-B", "video", "sim_to_silicone", None),
    ("A-to-C", "video", "sim_to_meat", None),
    ("C-to-B", "video", "meat_to_silicone", None),
]


def _to_webp(png_path, webp_path):
    """Upscale a raw map PNG by WEB_SCALE (nearest) and save it as lossless WebP."""
    bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
    big = cv2.resize(bgr, None, fx=WEB_SCALE, fy=WEB_SCALE, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)).save(
        webp_path, "WEBP", lossless=True, method=6)
    return bgr.shape[1], bgr.shape[0]   # map width, height in px (= mm)


def main():
    web_dir = repo_path(WEB_DIR)
    if os.path.isdir(web_dir):
        shutil.rmtree(web_dir)
    os.makedirs(web_dir)
    manifest = [
        "# Bird's-eye vessel maps on the project page", "",
        f"Written by `docker/website_vessel_maps.sh` (`website_vessel_maps.py`). "
        f"Each image is a run's `confusion_r00.png` (1 px = 1 mm, ground truth not grown) "
        f"upscaled x{WEB_SCALE} nearest-neighbour and saved as lossless WebP. "
        "Colours: green = both say vessel, red = missed vessel (truth only), "
        "blue = false alarm (prediction only), black = neither.", "",
    ]
    for config, gt_source, prefix, sim_choice in PAGE_RUNS:
        if sim_choice is not None:
            os.environ["DIFFTACTILE_SIM_MAP_TRAJECTORIES"] = sim_choice
        else:
            os.environ.pop("DIFFTACTILE_SIM_MAP_TRAJECTORIES", None)
        print(f"\n==================== {config}, ground truth from {gt_source} ====================")
        run_dir = vessel_map.run(config, gt_source=gt_source)
        with open(os.path.join(run_dir, "run.json")) as f:
            run_json = json.load(f)
        maps = run_json["maps"]                # insertion order = map order of the run
        manifest += [f"## {config} ({run_json['train']} -> {run_json['test']}), "
                     f"ground truth from {gt_source}", "",
                     f"Run: `{os.path.relpath(run_dir, repo_path('.'))}`  ",
                     f"Model: {run_json['model']['description']}  ",
                     f"Threshold: {run_json['threshold']['note']}", "",
                     "| # | file | map (run subfolder) | description | size (mm, w x h) |",
                     "|---|---|---|---|---|"]
        for k, (name, info) in enumerate(maps.items()):
            src = os.path.join(run_dir, name if len(maps) > 1 else "", "confusion_r00.png")
            fname = f"{prefix}.webp" if len(maps) == 1 else f"{prefix}_{name}.webp"
            w, h = _to_webp(src, os.path.join(web_dir, fname))
            manifest.append(f"| {k + 1} | `{fname}` | `{name}` | {info['description']} | {w} x {h} |")
            print(f"  {fname}: {os.path.getsize(os.path.join(web_dir, fname)) / 1024:.1f} kB")
        manifest.append("")
    with open(os.path.join(web_dir, "manifest.md"), "w") as f:
        f.write("\n".join(manifest))
    print(f"\nProject-page maps written to {web_dir} (see manifest.md)")
