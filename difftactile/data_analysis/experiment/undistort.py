import cv2
import numpy as np

from difftactile.main.constants import *


def main():
    # Load the image
    image = cv2.imread(SYSTEM_PARAMS.files.phantom_uncropped_compressed)

    # Step 1: Define the 4 source points (your input - in image coordinates)
    # Replace with your actual points (x, y) in clockwise or counterclockwise order
    src_pts = np.array([
        [783, 832],  # Top-left
        [3148, 850],  # Top-right
        [3119, 2227],  # Bottom-right
        [772, 2206]   # Bottom-left
    ], dtype='float32')

    # Step 2: Define destination points — a perfect rectangle
    # You can calculate the width and height based on distances between the points
    width = int(max(
        np.linalg.norm(src_pts[0] - src_pts[1]),
        np.linalg.norm(src_pts[2] - src_pts[3])
    ))
    height = int(max(
        np.linalg.norm(src_pts[0] - src_pts[3]),
        np.linalg.norm(src_pts[1] - src_pts[2])
    ))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    # Step 3: Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Step 4: Apply the warp perspective
    warped = cv2.warpPerspective(image, M, (width, height))

    # Step 5: Save the result
    cv2.imwrite(SYSTEM_PARAMS.files.phantom_uncropped_compressed_undistorted, warped)

if __name__ == "__main__":
    main()
