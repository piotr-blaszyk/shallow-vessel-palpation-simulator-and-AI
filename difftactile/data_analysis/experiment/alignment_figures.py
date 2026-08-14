"""Manuscript alignment figures: simulated vs real markers, on white.

Produces the four panels of

    "Alignment between simulated (red) and real (green) marker positions after
     domain adaptation for four canonical interactions: (a) press, (b) twist
     about the z-axis, (c) twist about the x-axis, and (d) slide."

Deliberately DIFFERENT from `Contact.generate_validation_img`, which draws the
same two point sets over the sensor photograph. That version answers "where are
the markers on the real sensor"; this one answers "how far apart are the two
sets", and for a small figure in a manuscript the photograph is texture that
competes with the data. Here the background is plain white, the dots are large
enough to read at print size, and a blue segment joins each corresponding pair
so the direction and magnitude of the misalignment are visible per marker rather
than inferred from two overlapping clouds.

Sizing is the whole difficulty: the markers sit ~55 px apart, so dots big enough
to see must still not touch their neighbours, and the connector must remain
visible without hiding the dots it joins. See DOT_RADIUS_PX / LINE_WIDTH_PX.

Reads the cached `markers_<name>.npz` written by `compute_da_loss`, so the
figures can be restyled without re-running the simulation.
"""

import os

import numpy as np

from difftactile.main.display import finish_plot
from difftactile.main.paths import repo_path

# The four interactions, in the order the manuscript's panels (a)-(d) use.
TRAJECTORY_ORDER = ["press", "twist_z", "twist_x", "slide"]
PANEL_LABELS = {"press": "(a)", "twist_z": "(b)", "twist_x": "(c)",
                "slide": "(d)"}

# Drawing sizes, in the DATA's own pixel units (1920x1080 marker space), so a
# radius here means the same thing regardless of output figure size or DPI.
#
# They are drawn as matplotlib Circles/Line2D in data coordinates rather than
# with `scatter`, whose `s` is in POINTS SQUARED - a screen unit that has no
# fixed relationship to the marker spacing. Sizing that way produced dots that
# merged into one solid mass and hid the connectors entirely.
#
# Inter-marker spacing is ~55 px. A red radius of 9 px is under a fifth of that,
# so a dot cannot reach its neighbour even when the pair is displaced; the green
# ring sits 3 px proud of it, enough to read as a ring at print size without
# closing the gap to the next marker.
DOT_RADIUS_PX = 9.0
REAL_DOT_RADIUS_PX = 12.0
# Half the red dot's diameter: clearly visible, but it cannot swallow the dots
# it joins.
LINE_WIDTH_PX = 9.0

COLOUR_SIM = "#ff0000"
COLOUR_REAL = "#00cc00"
COLOUR_LINK = "#0000ff"


def _panel(ax, sim, real):
    """Draw one interaction: green real dots, blue connectors, red sim dots.

    ORDER MATTERS and is the reason this is not three loops in the obvious
    sequence. Real dots go down first and largest, then the connectors, then the
    simulated dots on top: the link is thereby visible against the green but
    cannot cover the red centre, which is the point being compared. Drawing the
    lines last would hide exactly what the figure is about.
    """
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    # y is flipped: marker coordinates are image-space (y down), and a figure
    # reads naturally with y up. Flipping here rather than mutating the arrays
    # keeps the cached data untouched.
    real_xy = np.column_stack([real[:, 0], -real[:, 1]])
    sim_xy = np.column_stack([sim[:, 0], -sim[:, 1]])

    ax.add_collection(PatchCollection(
        [Circle(p, REAL_DOT_RADIUS_PX) for p in real_xy],
        facecolor=COLOUR_REAL, edgecolor="none", zorder=1))
    for (sx, sy), (rx, ry) in zip(sim_xy, real_xy):
        ax.plot([sx, rx], [sy, ry], color=COLOUR_LINK,
                linewidth=LINE_WIDTH_PX * 0.75, solid_capstyle="round",
                zorder=2)
    ax.add_collection(PatchCollection(
        [Circle(p, DOT_RADIUS_PX) for p in sim_xy],
        facecolor=COLOUR_SIM, edgecolor="none", zorder=3))

    # add_collection does not update the data limits, so set them explicitly.
    pts = np.vstack([real_xy, sim_xy])
    pad = REAL_DOT_RADIUS_PX * 2
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def generate_alignment_figures(source_dir, out_dir=None, combined=True):
    """Write one figure per trajectory from cached marker positions.

    `source_dir` holds `markers_<name>.npz` (written by `compute_da_loss`).
    Returns {name: mae_px} for the trajectories it found.
    """
    import matplotlib.pyplot as plt

    out_dir = out_dir or source_dir
    os.makedirs(out_dir, exist_ok=True)
    found, maes = {}, {}
    for name in TRAJECTORY_ORDER:
        path = os.path.join(source_dir, f"markers_{name}.npz")
        if not os.path.exists(path):
            print(f"  {name:9s} no cached markers ({path})")
            continue
        data = np.load(path)
        found[name] = (data["sim"], data["real"])
        maes[name] = float(data["mae_px"])

    if not found:
        raise FileNotFoundError(
            f"No markers_*.npz in {source_dir}. Run the four trajectories first "
            f"(./docker/alignment_figures.sh does this when the cache is empty)."
        )

    for name, (sim, real) in found.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        _panel(ax, sim, real)
        path = os.path.join(out_dir, f"alignment_{name}.png")
        finish_plot(plt, path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        print(f"  {name:9s} MAE {maes[name]:6.2f} px  -> {path}")

    if combined and len(found) > 1:
        # A 2x2 sheet in panel order, for dropping into the manuscript whole.
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, name in zip(axes.ravel(), TRAJECTORY_ORDER):
            if name not in found:
                ax.axis("off")
                continue
            sim, real = found[name]
            _panel(ax, sim, real)
            ax.set_title(f"{PANEL_LABELS[name]} {name.replace('_', ' ')}",
                         fontsize=13)
        path = os.path.join(out_dir, "alignment_all.png")
        finish_plot(plt, path, dpi=300, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white")
        print(f"  combined  -> {path}")
    return maes


def main():
    """Entrypoint: figures from the newest cache, or a named directory."""
    source = os.environ.get("DIFFTACTILE_ALIGNMENT_SOURCE")
    if not source:
        published = repo_path(
            "difftactile/output/domain_adaptation_published/joint_bo"
        )
        source = published
    out_dir = os.environ.get("DIFFTACTILE_ALIGNMENT_OUT", source)
    print(f"Marker cache: {source}")
    generate_alignment_figures(source, out_dir=out_dir)
