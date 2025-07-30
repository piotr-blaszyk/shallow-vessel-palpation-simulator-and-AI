import json
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Load the VGG JSON annotation
with open("/Users/piotrblaszyk/Downloads/labels_my-project-name_2025-07-30-04-41-54.json") as f:
    data = json.load(f)

# Path to the image (you can update this accordingly)
image_filename = "/Users/piotrblaszyk/Downloads/phantom.jpg"
image = cv2.imread(image_filename)
height, width = image.shape[:2]

# Create a blank mask
mask = np.zeros((height, width), dtype=np.uint8)

# Draw polygons on the mask
for region in data["phantom.jpg"]["regions"]:
    shape = region["shape_attributes"]
    if shape["name"] == "polygon":
        pts = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1)

# Save or display the mask
cv2.imwrite("binary_mask.png", mask * 255)  # save as white (255) and black (0)

# Optional: visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Binary Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
