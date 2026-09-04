"""3D render of the ST-GNN input graph: 5 frames x (127 marker nodes + 1 global node).

Library choice (three candidates were considered, see NOTES.md):
  * NetworkX + Matplotlib (3D axes)  - graph-native data structure, ships with
    the project's stack, prints to PNG/PDF at any dpi, painter's-algorithm depth
    sorting is good enough for a layered, mostly planar graph like this one;
  * Plotly (Scatter3d)               - interactive, true depth, but static export
    needs the kaleido/Chrome toolchain and its typography is hard to match to LaTeX;
  * PyVista (VTK)                    - photoreal 3D with tubes/spheres, but it is
    a mesh library rather than a graph library and needs an OpenGL context.
  NetworkX + Matplotlib is used: the graph is built in NetworkX (so every edge
  type is a tagged, de-duplicated undirected edge) and drawn on a Matplotlib 3D
  axis with one colour per node/edge family.

Graph, as built by cnn/dataset.py (verified in the source, see NOTES.md):
  * per frame 127 marker nodes at the ViTacTip's hexagonal marker positions
    (base-graph-connectivity.npz: measured pixel layout, 1 centre + 6 rings) and
    one global node; 5 frames -> 640 nodes;
  * spatial edges: hexagonal neighbours within a frame (342 undirected pairs);
  * temporal edges: each marker to itself in the previous/next frame;
  * global-spatial edges: every marker to its frame's global node, both ways;
  * global-temporal edges: consecutive global nodes.
  All directed pairs A->B, B->A are drawn as ONE undirected line.

Layout: the five frames run left to right along the x (time) axis (frame 0 left,
frame 4 right). Each frame is a vertical slice PERPENDICULAR to the time axis: its
hexagonal marker grid lies in the y-z plane at the bottom and its global node
sits above the grid in the same plane, so the temporal edges are the straight
lines along x between neighbouring slices and the global-temporal chain runs
along the top.

Colours and weights (the poster legend in the poster source repeats them):
  marker node + spatial edge   blue      solid, spatial width 0.6
  global node + global-temporal edge   orange, width 2.0
  temporal edge                green,   thin/faint, except the frame 1 -> 2 set,
                               drawn at spatial weight to show one full set
  global-spatial edge          magenta, thin/faint, except frame 3's set,
                               drawn at spatial weight to show one full fan

Output: analysis/figures/gnn_graph.png (600 dpi, transparent, cropped to content).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGURES, OUTPUT, ensure_dirs  # noqa: E402

CONN = os.path.join(OUTPUT, "base-graph-connectivity.npz")
CLIP_LEN = 5
DX = 14.0          # frame-to-frame distance along x (time), in marker spacings (grid diameter ~ 12)
H_GLOBAL = 11.0    # z of the global node: above the grid, which spans z in [-6, 6]
HL_TEMPORAL = (1, 2)   # the temporal edge set drawn at full weight (frames 1 -> 2)
HL_GLOBAL_SPATIAL = 3  # the frame whose marker -> global fan is drawn at full weight

COL = {
    "marker": "#1f77b4", "spatial": "#1f77b4",           # blue
    "global": "#ff7f0e", "global_temporal": "#ff7f0e",   # orange
    "temporal": "#2ca02c",                               # green
    "global_spatial": "#d400d4",                         # magenta
}
# (line width, alpha, z-order) for faint edges and for the highlighted sets
WEIGHT = {
    "spatial": (0.6, 0.85, 3),
    "temporal": (0.35, 0.3, 2), "temporal_hl": (0.6, 0.85, 4),
    "global_spatial": (0.2, 0.12, 1), "global_spatial_hl": (0.6, 0.85, 4),
    "global_temporal": (2.0, 0.95, 5),
}


def build_graph():
    """NetworkX graph with 3D positions, an `etype` and a `hl` (highlight) flag per edge."""
    d = np.load(CONN)
    pts = d["points"].astype(float)              # (127, 2) pixel layout
    pts -= pts.mean(axis=0)
    spacing = np.median(np.linalg.norm(pts[d["adjacency_matrix"][:, 0]] - pts[d["adjacency_matrix"][:, 1]], axis=1))
    pts /= spacing                                # units of one marker spacing
    n = len(pts)
    g = nx.Graph()
    for t in range(CLIP_LEN):
        x0 = t * DX
        for i in range(n):
            g.add_node(("m", t, i), pos=(x0, pts[i, 0], pts[i, 1]), ntype="marker")   # grid in the y-z plane
        g.add_node(("g", t), pos=(x0, 0.0, H_GLOBAL), ntype="global")
        for a, b in d["adjacency_matrix"]:
            g.add_edge(("m", t, int(a)), ("m", t, int(b)), etype="spatial", hl=False)
        for i in range(n):
            g.add_edge(("m", t, i), ("g", t), etype="global_spatial", hl=(t == HL_GLOBAL_SPATIAL))
        if t > 0:
            for i in range(n):
                g.add_edge(("m", t - 1, i), ("m", t, i), etype="temporal", hl=((t - 1, t) == HL_TEMPORAL))
            g.add_edge(("g", t - 1), ("g", t), etype="global_temporal", hl=False)
    return g


def render(g, path):
    pos = nx.get_node_attributes(g, "pos")
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=14, azim=-55)   # oblique: x (time) runs left to right, the slices stay visible
    ax.set_proj_type("ortho")      # all five frames at the same size
    # faint families first, highlighted sets and the global chain on top
    for etype in ("global_spatial", "temporal", "spatial", "global_temporal"):
        for hl in (False, True):
            segs = [(pos[u], pos[v]) for u, v, e in g.edges(data=True) if e["etype"] == etype and e["hl"] == hl]
            if not segs:
                continue
            lw, alpha, zo = WEIGHT[etype + ("_hl" if hl else "")]
            ax.add_collection3d(Line3DCollection(segs, colors=COL[etype], linewidths=lw, alpha=alpha, zorder=zo))
    for ntype, size in (("marker", 10), ("global", 110)):
        xyz = np.array([pos[v] for v, t in g.nodes(data="ntype") if t == ntype])
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=COL[ntype],
                   edgecolors="white", linewidths=0.3, depthshade=False, zorder=10)
    for t in range(CLIP_LEN):    # frame labels under each grid
        ax.text(t * DX, 0.0, -8.0, f"frame {t}", fontsize=11, ha="center", va="top", color="0.25")
    xyz = np.array(list(pos.values()))
    ax.set_xlim(xyz[:, 0].min() - 1, xyz[:, 0].max() + 1)
    ax.set_ylim(xyz[:, 1].min() - 1, xyz[:, 1].max() + 1)
    ax.set_zlim(-9, H_GLOBAL + 1)
    ax.set_box_aspect((np.ptp(xyz[:, 0]) + 2, np.ptp(xyz[:, 1]) + 2, H_GLOBAL + 10))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=600, transparent=True)
    plt.close(fig)
    # crop to the drawn content (alpha bounding box) so LaTeX can scale it tightly
    im = Image.open(path)
    im.crop(im.getchannel("A").getbbox()).save(path, optimize=True)


if __name__ == "__main__":
    graph = build_graph()
    print(f"nodes {graph.number_of_nodes()}, undirected edges {graph.number_of_edges()}: " +
          ", ".join(f"{k} {sum(1 for _, _, e in graph.edges(data='etype') if e == k)}"
                    for k in ("spatial", "temporal", "global_spatial", "global_temporal")))
    ensure_dirs()
    out = os.path.join(FIGURES, "gnn_graph.png")
    render(graph, out)
    print("written", out, Image.open(out).size)
