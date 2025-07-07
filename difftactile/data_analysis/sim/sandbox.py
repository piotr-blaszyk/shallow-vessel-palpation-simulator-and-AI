import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from difftactile.main.constants import *

# Define a set of points
points = np.array([
    [0, 0], [0, 1.1], [1, 0], [1, 1],
    [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]
])

# Create the Delaunay triangulation object
tri = Delaunay(points)

# Accessing triangulation information
print("Points:\n", points)
print("\nSimplices (triangles):\n", tri.simplices) # Indices of points forming each triangle
print("\nNeighbors of first simplex:\n", tri.neighbors[0]) # Indices of neighboring simplices