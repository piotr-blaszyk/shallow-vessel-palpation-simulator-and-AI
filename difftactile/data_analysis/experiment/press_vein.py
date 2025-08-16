import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pickle
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.main.constants import *

img1 = cv2.imread(SYSTEM_PARAMS.files.press_no_vein)
img2 = cv2.imread(SYSTEM_PARAMS.files.press_horizontal_vein)
markers1, circle_center1, circle_radius1 = FisheyeModelNoTaichi.get_marker_image(img1)
markers2, circle_center2, circle_radius2 = FisheyeModelNoTaichi.get_marker_image(img2)
img1_with_markers = img1.copy()
img2_with_markers = img2.copy()

for marker in markers1:
    point = tuple(map(int, marker))
    cv2.circle(img1_with_markers, point, 5, (0, 255, 0), -1)
    cv2.circle(img2_with_markers, point, 5, (0, 255, 0), -1)
for marker in markers2:
    point = tuple(map(int, marker))
    cv2.circle(img1_with_markers, point, 5, (0, 0, 255), -1)
    cv2.circle(img2_with_markers, point, 5, (0, 0, 255), -1)

composite = cv2.addWeighted(img1_with_markers, 0.5, img2_with_markers, 0.5, 0)
circle_center = tuple(map(int, circle_center1))
circle_radius = int(circle_radius1)
cv2.circle(composite, circle_center, circle_radius, (255, 255, 255), 2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(composite, f'No Vein Markers: {len(markers1)}', (10, 30), font, 0.8, (0, 255, 0), 2)
cv2.putText(composite, f'Horizontal Vein Markers: {len(markers2)}', (10, 60), font, 0.8, (0, 0, 255), 2)
cv2.imshow('Composite Image with Markers', composite)
cv2.waitKey(0)
cv2.destroyAllWindows()

