import numpy as np
import matplotlib.pyplot as plt

from difftactile.sensor_model.fisheye_model import *

# Create points along each axis
t = np.linspace(0, 1, 10)
x_axis_points = np.array([[x, 0, 1] for x in t])  # Points along X axis
y_axis_points = np.array([[0, y, 1] for y in t])  # Points along Y axis

fisheye_model = FisheyeModel()
# Project the points
x_proj = fisheye_model.project_points_to_pix(x_axis_points)
y_proj = fisheye_model.project_points_to_pix(y_axis_points)

# Create visualization
plt.figure(figsize=(10, 10))
plt.scatter(x_proj[:,0], x_proj[:,1], c='red', label='X+ axis')
plt.scatter(y_proj[:,0], y_proj[:,1], c='green', label='Y+ axis')

# Add arrows to show direction
plt.arrow(x_proj[0,0], x_proj[0,1], 
         x_proj[-1,0]-x_proj[0,0], x_proj[-1,1]-x_proj[0,1], 
         color='red', width=0.5)
plt.arrow(y_proj[0,0], y_proj[0,1], 
         y_proj[-1,0]-y_proj[0,0], y_proj[-1,1]-y_proj[0,1], 
         color='green', width=0.5)

plt.title('Fisheye Projection of 3D Axes')
plt.xlabel('Image X')
plt.ylabel('Image Y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()