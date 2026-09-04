"""Build every image block of the poster's high-level workflow diagram.

Each function below produces ONE block image of the methods-column workflow
diagram (the poster source, "Workflow" figure) from the project-page sources of the
shallow-vessel-palpation-simulator-and-AI repository (its `docs/videos/`,
`docs/images/`) or from the manuscript's figure sources. Nothing is drawn by
hand: video frames are extracted with ffmpeg at fixed timestamps, black/white
padding is trimmed automatically, and arrows are drawn with OpenCV at
positions computed from the image content (e.g. the centroid of the green
sensor point cloud), so the whole set can be regenerated with one command:

    python scripts/make_workflow_images.py            # all blocks
    python scripts/make_workflow_images.py A1 B2 ...  # a subset

Outputs go to analysis/figures/workflow/<name>.png (RGB, no alpha).

Blocks (name -> source, edit):
  real_silicone   data_collection_silicone.mp4 @11 s, centre crop (phantom + robot
                  wrist, captions cut off), 5 double-headed arrows along the five
                  slide trajectories (4 green: vessel crossed, 1 red: vessel-free)
  real_meat       data_collection_meat.jpg, unchanged
  ann_silicone    dataset_annotations_silicone.mp4 @22 s, status bar cut, padding trimmed
  ann_meat        dataset_annotations_meat.mp4 @1:49, status bar cut, padding trimmed
  vitactip        ViTacTip photo cut out of the manuscript's framework figure, the
                  green "1" badge inpainted with the surrounding casing texture
  mesh            vitactip_mesh_side.webp, white padding trimmed
  A1, A2          slide videos @6 s, padding trimmed, red arrow pointing left
  B1              press video, last frame, red arrow pointing down
  B2              twist-about-x video, last frame, 30-degree anticlockwise arc arrow
                  (12 o'clock -> 11 o'clock)
  B3              twist-about-z video, last frame, 30-degree anticlockwise arc arrow
                  (6:30 -> 5:30) flattened to half its bulge (a rotation about the
                  vertical axis seen from the side)
  dr_heading      dr_heading_0.mp4 @1.00 s (top view), padding trimmed, three red
                  arrows from the sensor centre: straight up, +15 deg, -15 deg
  dr_stiffness    sim_slide_vessel_present.mp4 @8.00 s, padding trimmed, blue phantom
                  particles recoloured by their surroundings (black/green/yellow), the
                  two clock-arm keypoints (small yellow + magenta blobs) removed, and a
                  red double-headed arrow between the sensor and the vessel
"""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ANALYSIS, FIGURES, REPO  # noqa: E402

VIDEOS = os.path.join(REPO, "docs", "videos")
MESH = os.path.join(REPO, "docs", "images", "sensor_mesh", "vitactip_mesh_side.webp")
# The ViTacTip photograph is cropped out of the pipeline overview figure, which is kept
# beside these scripts so this file needs nothing outside the repository.
FRAMEWORK_FIG = os.path.join(ANALYSIS, "assets", "vessel_detection_framework.png")
OUT = os.path.join(FIGURES, "workflow")
WORK = os.path.join(tempfile.gettempdir(), "workflow-frames")

RED = (255, 0, 0)     # RGB
GREEN = (0, 200, 0)
UPSCALE = 2           # arrows are drawn on a 2x LANCZOS upscale for smoother edges


# --------------------------------------------------------------------------- helpers
def frame_at(video, seconds=None, last=False):
    """Extract one frame of `video` (name in docs/videos) as an RGB array."""
    os.makedirs(WORK, exist_ok=True)
    tag = "last" if last else f"{seconds:.2f}"
    png = os.path.join(WORK, f"{os.path.splitext(video)[0]}_{tag}.png")
    if not os.path.exists(png):
        src = os.path.join(VIDEOS, video)
        if last:
            cmd = ["ffmpeg", "-v", "error", "-y", "-sseof", "-0.05", "-i", src,
                   "-update", "1", "-frames:v", "1", png]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds:.3f}", "-i", src,
                   "-frames:v", "1", png]
        subprocess.run(cmd, check=True)
    return np.asarray(Image.open(png).convert("RGB"))


def trim(img, background="black", thresh=24, margin=6):
    """Crop `img` to the bounding box of its non-background pixels (+ margin)."""
    if background == "black":
        content = img.max(axis=2) > thresh
    else:
        content = img.min(axis=2) < 255 - thresh
    ys, xs = np.where(content)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def colour_masks(img):
    """Boolean masks of the five simulator colours (robust to video compression)."""
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    black = img.max(axis=2) < 60
    green = (g > 90) & (r < 90) & (b < 90)
    blue = (b > 35) & (b > r + 15) & (b > g + 15)
    yellow = (r > 90) & (g > 90) & (b < 90)
    magenta = (r > 90) & (b > 90) & (g < 90)
    # Looser, hue-based variants used when REMOVING things, so that the dim
    # anti-aliased fringe of a removed sphere or a blue/green mixed pixel of a
    # phantom particle seen through the dome goes with it.
    yellowish = (r > 60) & (g > 60) & (b < 0.75 * np.minimum(r, g))
    cyanish = (g > 60) & (b > 60) & (r < 60) & (b > 0.55 * g)
    return {"black": black, "green": green, "blue": blue, "yellow": yellow,
            "magenta": magenta, "yellowish": yellowish, "cyanish": cyanish}


def centroid(mask):
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def radius_inside(mask, centre, clock_from, clock_to, frac=0.8, closing=25):
    """Largest radius such that the arc clock_from..clock_to stays inside `mask`.

    The sensor is a cloud of particles, so the mask is first closed with a
    large structuring element to fill the gaps between them; the arc's angular
    range is then ray-marched from the centre and the shortest distance to the
    mask boundary, scaled by `frac`, is returned.
    """
    solid = ndimage.binary_closing(mask, structure=np.ones((closing, closing)))
    solid = ndimage.binary_fill_holes(solid)
    cx, cy = centre
    best = np.inf
    for hours in np.linspace(clock_from, clock_to, 25):
        a = np.deg2rad(hours * 30.0)
        for r in range(1, max(mask.shape)):
            x, y = int(round(cx + r * np.sin(a))), int(round(cy - r * np.cos(a)))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or not solid[y, x]:
                best = min(best, r)
                break
    return frac * best


def upscale(img):
    return np.asarray(Image.fromarray(img).resize(
        (img.shape[1] * UPSCALE, img.shape[0] * UPSCALE), Image.LANCZOS))


def arrow(img, p0, p1, colour=RED, thickness=8, tip=0.28, double=False):
    """Straight arrow p0 -> p1 (coordinates in the UPSCALED image)."""
    p0 = tuple(int(round(v)) for v in p0)
    p1 = tuple(int(round(v)) for v in p1)
    cv2.arrowedLine(img, p0, p1, colour, thickness, cv2.LINE_AA, tipLength=tip)
    if double:
        cv2.arrowedLine(img, p1, p0, colour, thickness, cv2.LINE_AA, tipLength=tip)


def arc_arrow(img, centre, radius, clock_from, clock_to, flatten=1.0,
              colour=RED, thickness=8, head_len=None):
    """Arc arrow around `centre` from clock position `clock_from` to `clock_to`.

    Clock positions are in HOURS on a 12-hour dial (12 = up, 3 = right); the
    arrow sweeps from clock_from to clock_to in the numeric direction (decreasing
    hours = anticlockwise), so write 10:30 as -1.5 to sweep 1:30 -> 10:30 the
    short way. `flatten` < 1 scales the
    arc's bulge (its deviation from the chord) while keeping both endpoints,
    which is how a rotation about the vertical axis looks from the side.
    """
    a0, a1 = np.deg2rad(clock_from * 30.0), np.deg2rad(clock_to * 30.0)
    angles = np.linspace(a0, a1, 60)
    cx, cy = centre
    pts = np.stack([cx + radius * np.sin(angles), cy - radius * np.cos(angles)], axis=1)
    if flatten != 1.0:
        chord0, chord1 = pts[0], pts[-1]
        d = chord1 - chord0
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        rel = pts - chord0
        along = rel @ d
        across = rel @ n
        pts = chord0 + np.outer(along, d) + np.outer(across * flatten, n)
    poly = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], False, colour, thickness, cv2.LINE_AA)
    # Arrow head: a short straight arrow along the final tangent.
    head_len = head_len or max(3.0 * thickness, 0.35 * radius)
    tangent = pts[-1] - pts[-4]
    tangent /= np.linalg.norm(tangent)
    tail = pts[-1] - tangent * head_len
    cv2.arrowedLine(img, tuple(int(v) for v in tail), tuple(int(v) for v in pts[-1]),
                    colour, thickness, cv2.LINE_AA, tipLength=1.0)


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.png")
    Image.fromarray(img).save(path, optimize=True)
    print(f"wrote {path} {img.shape[1]}x{img.shape[0]}")


# --------------------------------------------------------------------------- real datasets
def real_silicone():
    """Silicone data-collection frame: centre crop + 5 slide-trajectory arrows."""
    img = frame_at("data_collection_silicone.mp4", 11.0)
    x0, y0, x1, y1 = 345, 225, 905, 645      # captions on the left are outside this box
    crop = img[y0:y1, x0:x1]
    big = upscale(crop).copy()
    # Corners of the phantom's top face in full-frame coordinates (read off a
    # gridded enlargement of this frame): A left/near, B far-left, C far-right,
    # D near-right. AD and BC are one edge pair, AB and DC the other.
    #
    # ORIENTATION (user's decision, 2026-09-04, twice): each arrow runs from the
    # AB edge to the DC edge -- roughly left to right in the image -- and the five
    # are spread VERTICALLY up the phantom's face. An earlier version ran them
    # along the other axis; the user inspected the frame and asked for this one.
    A, B, C, D = (393, 505), (545, 355), (738, 402), (640, 550)
    to_big = lambda p: ((p[0] - x0) * UPSCALE, (p[1] - y0) * UPSCALE)
    A, B, C, D = map(np.array, map(to_big, (A, B, C, D)))
    # t walks up from the near edge to the far one, so t = 0.10 is the arrow lowest
    # in the image: that is the red one.
    for t, colour in zip((0.10, 0.30, 0.50, 0.70, 0.90), (RED, GREEN, GREEN, GREEN, GREEN)):
        left = A + t * (B - A)
        right = D + t * (C - D)
        p0 = left + 0.15 * (right - left)
        p1 = right - 0.15 * (right - left)
        arrow(big, p0, p1, colour, thickness=7, tip=0.2, double=True)
    save("real_silicone", big)


def real_meat():
    img = np.asarray(Image.open(os.path.join(VIDEOS, "data_collection_meat.jpg")).convert("RGB"))
    save("real_meat", img)


# --------------------------------------------------------------------------- annotated datasets
def _cut_status_bar(img):
    """Drop the light status bar the annotation viewer draws under the frame."""
    bright = img.mean(axis=(1, 2)) > 180
    rows = np.where(bright)[0]
    if rows.size and rows.min() > img.shape[0] * 0.7:
        return img[:rows.min()]
    return img


def _annotated(video, seconds):
    """Status bar cut, then a tight crop around the bright sensor disc."""
    img = _cut_status_bar(frame_at(video, seconds))
    return trim(img, thresh=70, margin=12)


def ann_silicone():
    save("ann_silicone", _annotated("dataset_annotations_silicone.mp4", 22.0))


def ann_meat():
    save("ann_meat", _annotated("dataset_annotations_meat.mp4", 109.0))


# --------------------------------------------------------------------------- sensor
def vitactip():
    """The ViTacTip photo cropped out of the pipeline overview figure, badge removed."""
    fig = np.asarray(Image.open(FRAMEWORK_FIG).convert("RGB"))
    photo = fig[705:940, 942:1232].copy()
    # The green "1" badge sits over the black casing; inpaint its disc from the
    # surrounding casing texture (Telea) after masking a slightly larger disc so
    # the anti-aliased rim goes too.
    r, g, b = [photo[..., i].astype(int) for i in range(3)]
    green = (g > 120) & (r < 200) & (b < 200) & (g - r > 30) & (g - b > 30)
    lab = ndimage.label(green)[0]
    sizes = ndimage.sum(green, lab, range(1, lab.max() + 1))
    badge = lab == (int(np.argmax(sizes)) + 1)
    cy, cx = ndimage.center_of_mass(badge)
    ys, xs = np.where(badge)
    rad = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max() + 4   # outer edge of the ring
    mask = np.zeros(photo.shape[:2], np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(rad)), 255, -1)
    bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    save("vitactip", cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))


def mesh():
    img = np.asarray(Image.open(MESH).convert("RGB"))
    # Crop to the coloured surfaces of the mesh; the black axis triad in the
    # corner of the Gmsh screenshot is grey-scale and therefore excluded.
    sat = img.max(axis=2).astype(int) - img.min(axis=2).astype(int)
    ys, xs = np.where(sat > 40)
    m = 6
    save("mesh", img[max(ys.min() - m, 0):ys.max() + m, max(xs.min() - m, 0):xs.max() + m])


# --------------------------------------------------------------------------- simulator interactions
def _sensor_view(video, seconds=None, last=False):
    """Trimmed frame + (upscaled image, green-sensor centroid, sensor bbox)."""
    img = trim(frame_at(video, seconds, last))
    big = upscale(img).copy()
    green = colour_masks(big)["green"]
    cx, cy = centroid(green)
    ys, xs = np.where(green)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    return big, (cx, cy), bbox


# Arrow geometry of the five simulator blocks: thick, long arrows that read at
# poster scale; the twist blocks use ARCS (rotations), the slide/press blocks lines.
ARROW_THICK = 16       # px in the upscaled image
ARC_HEAD = 60          # arrow-head length of the arc arrows, px


def _slide(name, video):
    big, (cx, cy), (x0, y0, x1, y1) = _sensor_view(video, 6.0)
    w = x1 - x0
    # from right of the centre almost to the left edge of the sensor: a long leftward slide
    arrow(big, (cx + 0.15 * w, cy), (cx - 0.42 * w, cy), thickness=ARROW_THICK, tip=0.3)
    save(name, big)


def A1():
    _slide("A1", "sim_slide_vessel_present.mp4")


def A2():
    _slide("A2", "sim_slide_vessel_absent.mp4")


def B1():
    big, (cx, cy), (x0, y0, x1, y1) = _sensor_view("sim_press.mp4", last=True)
    h = y1 - cy
    arrow(big, (cx, cy - 0.35 * h), (cx, cy + 0.85 * h), thickness=ARROW_THICK, tip=0.3)  # long downward press
    save("B1", big)


def B2():
    big, (cx, cy), _bbox = _sensor_view("sim_twist_x.mp4", last=True)
    green = colour_masks(big)["green"]
    # rotation about the x axis, seen side-on: a 120-degree anticlockwise arc (2 -> 10 o'clock)
    radius = radius_inside(green, (cx, cy), 2, -2, frac=0.9)  # large, but inside the sensor
    arc_arrow(big, (cx, cy), radius, clock_from=2, clock_to=-2, thickness=ARROW_THICK, head_len=ARC_HEAD)
    save("B2", big)


def B3():
    big, (cx, cy), _bbox = _sensor_view("sim_twist_z.mp4", last=True)
    green = colour_masks(big)["green"]
    # rotation about the vertical axis, seen side-on: a wide anticlockwise arc along the
    # bottom of the sensor (8 -> 4 o'clock), flattened into the ellipse a horizontal circle
    # makes from the side
    radius = radius_inside(green, (cx, cy), 8, 4, frac=0.9)
    arc_arrow(big, (cx, cy), radius, clock_from=8, clock_to=4, flatten=0.45,
              thickness=ARROW_THICK, head_len=ARC_HEAD)
    save("B3", big)


# --------------------------------------------------------------------------- domain randomisation
def dr_heading():
    """Top view of the slide: three approach-angle arrows from the sensor centre."""
    img = trim(frame_at("dr_heading_0.mp4", 1.0))
    big = upscale(img).copy()
    green = colour_masks(big)["green"]
    cx, cy = centroid(green)
    length = 0.85 * cy                     # up to just below the top edge
    for deg in (0, 15, -15):
        a = np.deg2rad(deg)
        tip = (cx + length * np.sin(a), cy - length * np.cos(a))
        arrow(big, (cx, cy), tip, thickness=8, tip=0.16)
    save("dr_heading", big)


def dr_stiffness():
    """Side view over the vessel with the phantom removed and a stiffness arrow."""
    img = trim(frame_at("sim_slide_vessel_present.mp4", 8.0)).copy()
    m = colour_masks(img)
    # 1. Every blue blob (phantom particle) takes the colour that dominates its
    #    immediate surroundings among black / green / yellow.
    remove = m["blue"] | m["cyanish"]
    lab, n = ndimage.label(remove)
    ring = ndimage.binary_dilation(remove, iterations=3) & ~remove
    palette = {"black": (0, 0, 0), "green": (0, 220, 0), "yellow": (230, 230, 0)}
    for i in range(1, n + 1):
        blob = lab == i
        halo = ndimage.binary_dilation(blob, iterations=3) & ring
        votes = {k: int((m[k] & halo).sum()) for k in ("black", "green", "yellow")}
        img[blob] = palette[max(votes, key=votes.get)]
    # 2. The two clock-arm keypoints: all magenta, plus any yellow that is not
    #    part of the vessel (the vessel is the one large cluster of yellow
    #    particles; the keypoint is an isolated yellow dot higher up).
    m = colour_masks(img)
    img[m["magenta"]] = 0
    yl = ndimage.binary_dilation(m["yellow"], iterations=10)
    lab, n = ndimage.label(yl)
    sizes = ndimage.sum(m["yellow"], lab, range(1, n + 1))
    vessel_cluster = lab == (int(np.argmax(sizes)) + 1)
    img[m["yellowish"] & ~vessel_cluster] = 0
    # Any leftover magenta/yellow anti-aliasing fringe (dim, low saturation).
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    fringe = ((r > 40) & (b > 40) & (g < 60)) | ((r > 40) & (g > 40) & (b < 60) & ~vessel_cluster)
    img[fringe] = 0
    # Anything that is neither clearly green nor clearly yellow and is dark is
    # a compression speckle left over from the removed particles: make it black.
    m = colour_masks(img)
    speckle = ~(m["green"] | m["yellow"]) & (img.max(axis=2) < 130)
    img[speckle] = 0
    # 3. Double-headed arrow between the sensor centre and the vessel centre.
    big = upscale(img).copy()
    m = colour_masks(big)
    sx, sy = centroid(m["green"])
    vx, vy = centroid(m["yellow"])
    arrow(big, (sx, sy), (vx, vy - 0.02 * big.shape[0]), thickness=9, tip=0.22, double=True)
    save("dr_stiffness", big)


# --------------------------------------------------------------------------- top-view maps
# The four "Top-view vessel maps" blocks and the one per-frame prediction block are
# not drawn here: they are files another entry point of the repository already wrote,
# copied in unchanged so the diagram cannot drift from the published run.
#
#   maps          the 5x-enlarged confusion map (confusion_r00_big.png) of the published
#                 vessel-map run of each configuration -- docker/vessel_map_all.sh, and
#                 docker/vessel_map_sim_test_trajectories.sh for the Sim->Sim panel, whose
#                 map is one held-out trajectory rather than the single dedicated slide.
#   pred          the per-frame confusion overlay chosen by select_sim_to_meat_frames.py:
#                 the frame of the Sim->Meat trial whose TP/FP/FN/TN mix is closest to the
#                 pooled test-set mix, i.e. a typical frame rather than a flattering one.
#                 The trial is the same one the Sim->Meat map panel shows.
SIM_TO_SIM_MAP = "map_04_trajectory_0430_vessel-present"   # 4th of the ten held-out trajectories
SIM_TO_MEAT_TRIAL = "trial_03_1-metal-straw-beneath-3-steaks-20260228-233031"


def _newest(d):
    runs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.endswith("-legacy"))
    if not runs:
        raise FileNotFoundError(f"no run under {d}")
    return runs[-1]


def maps():
    """Copy the four top-view confusion maps out of the published vessel-map runs."""
    from pathlib import Path
    vm = Path(REPO) / "difftactile/output/vessel_maps"
    sources = {
        "map_sim_to_sim": _newest(vm / "sim-to-sim-test-trajectories_gt-simulator")
                          / SIM_TO_SIM_MAP / "confusion_r00_big.png",
        "map_sim_to_meat": _newest(vm / "sim-to-meat_gt-video")
                           / SIM_TO_MEAT_TRIAL / "confusion_r00_big.png",
        "map_sim_to_silicone": _newest(vm / "sim-to-silicone_gt-video") / "confusion_r00_big.png",
        "map_meat_to_silicone": _newest(vm / "meat-to-silicone_gt-video") / "confusion_r00_big.png",
    }
    for name, src in sources.items():
        save(name, np.asarray(Image.open(src).convert("RGB")))


def pred_sim_to_meat():
    """Copy in the representative Sim->Meat frame chosen by select_sim_to_meat_frames.py."""
    from pathlib import Path
    root = Path(ANALYSIS) / "overlays/sim_to_meat_frame_choice"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist - run analysis/scripts/select_sim_to_meat_frames.py first")
    run = sorted(p for p in root.iterdir() if p.is_dir())[-1]
    hits = sorted(run.glob(f"{SIM_TO_MEAT_TRIAL}_frame_*_confusion.png"))
    if not hits:
        raise FileNotFoundError(f"no confusion overlay for {SIM_TO_MEAT_TRIAL} under {run}")
    save("pred_sim_to_meat", np.asarray(Image.open(hits[0]).convert("RGB")))


BLOCKS = {
    "real_silicone": real_silicone, "real_meat": real_meat,
    "ann_silicone": ann_silicone, "ann_meat": ann_meat,
    "vitactip": vitactip, "mesh": mesh,
    "A1": A1, "A2": A2, "B1": B1, "B2": B2, "B3": B3,
    "dr_heading": dr_heading, "dr_stiffness": dr_stiffness,
    "maps": maps, "pred_sim_to_meat": pred_sim_to_meat,
}

if __name__ == "__main__":
    for name in (sys.argv[1:] or BLOCKS):
        BLOCKS[name]()
