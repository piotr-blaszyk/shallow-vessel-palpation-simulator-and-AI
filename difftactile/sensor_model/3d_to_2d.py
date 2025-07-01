import numpy as np
import cv2
import pickle
import os

from fisheye_model import *
from ..tasks.constants import *

fisheye_model = FisheyeModel()

# Load 3D nodes
with open(os.path.join(os.path.dirname(__file__), SYSTEM_PARAMS.files.vitactip_cam_3d_nodes), 'rb') as f:
    cam_3D_nodes = pickle.load(f)

# Load image
img_path = os.path.join(os.path.dirname(__file__), SYSTEM_PARAMS.files.vitactip_photo_default_state)
img = cv2.imread(img_path)

# Project using project_points_to_pix
points2d_pix = fisheye_model.project_points_to_pix(cam_3D_nodes.copy())

# Project using project_points_to_pix_cv2
points2d_cv2 = fisheye_model.project_points_to_pix_cv2(cam_3D_nodes.copy())

# Overlay function
def overlay_points(image, points, color=(0,255,0), radius=3):
    img_copy = image.copy()
    for pt in points:
        center = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(img_copy, center, radius=radius, color=color, thickness=-1)
    return img_copy

# Overlay and save
img_pix = overlay_points(img, points2d_pix, color=(0,255,0))
cv2.imwrite(SYSTEM_PARAMS.files.fisheye_model_project_points_to_pix, img_pix)

img_cv2 = overlay_points(img, points2d_cv2, color=(255,0,0))
cv2.imwrite(SYSTEM_PARAMS.files.fisheye_model_project_points_to_pix_cv2, img_cv2)
