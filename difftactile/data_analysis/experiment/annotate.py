import json
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from difftactile.main.constants import *


def main():
    with open(SYSTEM_PARAMS.files.exp_phantom_labels) as f:
        data = json.load(f)
    image = cv2.imread(SYSTEM_PARAMS.files.phantom_uncropped_compressed_undistorted)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    regions = data["phantom-uncropped-compressed-undistorted.jpg"]["regions"]
    for region_id in regions:
        region = regions[region_id]
        shape = region["shape_attributes"]
        if shape["name"] == "polygon":
            pts = np.array(
                list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32
            )
            cv2.fillPoly(mask, [pts], color=1)
    cv2.imwrite("binary_mask.png", mask * 255)
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


if __name__ == "__main__":
    main()
