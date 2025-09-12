import pyvista as pv
import numpy as np

# --- Example data ---

# Node coordinates (dict: node_id -> (x,y,z))
positions = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (0, 0, 1),
    4: (1, 1, 1),
}

# Edge list (pairs of node ids)
edges = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 4),
    (3, 4),
]

# --- Visualization setup ---

plotter = pv.Plotter()

# Add nodes (as spheres)
for node, coord in positions.items():
    sphere = pv.Sphere(radius=0.05, center=coord)
    plotter.add_mesh(sphere, color="skyblue", smooth_shading=True)

# Add edges (as tubes between nodes)
for u, v in edges:
    line = pv.Line(positions[u], positions[v])
    tube = line.tube(radius=0.01)
    plotter.add_mesh(tube, color="gray")

# Show interactive window
plotter.show()
