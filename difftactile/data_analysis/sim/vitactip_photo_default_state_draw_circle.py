import cv2
import numpy as np

from difftactile.main.constants import *

image_path = SYSTEM_PARAMS.files.vitactip_photo_default_state
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

center = (1035, 580)
radius = 370
color = (0, 255, 0)
thickness = 2
cv2.circle(img, center, radius, color, thickness)

center_point_radius = 3
center_point_color = (0, 0, 255)
cv2.circle(img, center, center_point_radius, center_point_color, -1)

cv2.imshow('Image with Circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(SYSTEM_PARAMS.files.init_with_circle, img) 