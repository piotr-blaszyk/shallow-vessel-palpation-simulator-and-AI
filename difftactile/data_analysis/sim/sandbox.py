import numpy as np
import cv2

from difftactile.main.constants import *


w = int(SYSTEM_PARAMS.fisheye_model.target_image_width)
h = int(SYSTEM_PARAMS.fisheye_model.target_image_height)

cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
r = SYSTEM_PARAMS.fisheye_model.circle_radius
r_small = SYSTEM_PARAMS.fisheye_model.circle_small_radius

disp = np.array([cx + 1, cy])  # Changed to a more visible point
        
# Compute unit vector from camera center to displacement point
center = np.array([cx, cy])
direction = disp - center
unit_vector = direction / np.linalg.norm(direction)

c = 1/3

# Calculate chord points at r_small * (1/3) distance from center
chord_center = center + unit_vector * (r_small * c)
# Get perpendicular vector
perp_vector = np.array([-unit_vector[1], unit_vector[0]])
# Calculate chord endpoints using perpendicular vector
chord_half_length = np.sqrt(r_small**2 - (r_small * c)**2)  # Pythagorean theorem
chord_point1 = chord_center + perp_vector * chord_half_length
chord_point2 = chord_center - perp_vector * chord_half_length

# Create circle cap mask
cap_mask = np.zeros((h, w), dtype=np.uint8)
# Create polygon points for the cap
circle_points = []
num_points = 32  # Number of points to approximate circle arc
angle_start = np.arctan2(chord_point1[1] - cy, chord_point1[0] - cx)
angle_end = np.arctan2(chord_point2[1] - cy, chord_point2[0] - cx)
# Ensure we take the smaller arc
if abs(angle_end - angle_start) > np.pi:
    if angle_end > angle_start:
        angle_start += 2 * np.pi
    else:
        angle_end += 2 * np.pi
angles = np.linspace(angle_start, angle_end, num_points)
for angle in angles:
    x = cx + r_small * np.cos(angle)
    y = cy + r_small * np.sin(angle)
    circle_points.append([x, y])
# Add chord points to close the polygon
circle_points.append(chord_point2)
circle_points.append(chord_point1)
# Convert to numpy array and correct format for cv2
cap_points = np.array(circle_points, dtype=np.int32).reshape((-1, 1, 2))
cv2.fillPoly(cap_mask, [cap_points], color=255)

# Create a visualization image
vis_img = np.zeros((h, w, 3), dtype=np.uint8)
# Draw the cap mask in blue
vis_img[cap_mask > 0] = [255, 0, 0]  # Blue for the cap

# Draw helper elements
# Full circle in white
cv2.circle(vis_img, (int(cx), int(cy)), int(r_small), (255, 255, 255), 1)
# Center point in red
cv2.circle(vis_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
# Displacement point in green
cv2.circle(vis_img, (int(disp[0]), int(disp[1])), 5, (0, 255, 0), -1)
# Direction line in yellow
cv2.line(vis_img, (int(cx), int(cy)), (int(disp[0]), int(disp[1])), (0, 255, 255), 1)
# Chord line in red
cv2.line(vis_img, (int(chord_point1[0]), int(chord_point1[1])), 
         (int(chord_point2[0]), int(chord_point2[1])), (0, 0, 255), 2)

# Display the image
cv2.imshow('Cap Mask Visualization', vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

