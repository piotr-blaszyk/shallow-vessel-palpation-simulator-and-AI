import cv2
import numpy as np

from difftactile.main.constants import *

# Load the image
image_path = SYSTEM_PARAMS.files.vitactip_photo_default_state
img = cv2.imread(image_path)

if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Circle parameters
center = (359, 266)
radius = 189
color = (0, 255, 0)  # Green color in BGR
thickness = 2

# Draw the circle
cv2.circle(img, center, radius, color, thickness)

# Draw the center point
center_point_radius = 3
center_point_color = (0, 0, 255)  # Red color in BGR
cv2.circle(img, center, center_point_radius, center_point_color, -1)  # -1 fills the circle

# Display the image
cv2.imshow('Image with Circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the result
cv2.imwrite(SYSTEM_PARAMS.files.init_with_circle, img) 