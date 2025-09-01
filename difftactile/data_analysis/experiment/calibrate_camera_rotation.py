import math

import cv2
import numpy as np

from difftactile.main.constants import *


def calibrate_vertical():
    image_path = SYSTEM_PARAMS.files.calibration_vertical
    img = cv2.imread(image_path)
    points = np.array([
        [1106, 286],
        [1000, 884]
    ])

    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    print(f"Vertical line angle: {angle_deg:.2f} degrees")

    for x in points:
        center = (x[0], x[1])
        radius = 3
        color = (0, 0, 255)
        thickness = 2
        cv2.circle(img, center, radius, color, thickness)

    cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0, 255, 0), 2)
    cv2.imshow('image with point and line', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(SYSTEM_PARAMS.files.init_with_circle, img)

    return angle_deg

def calibrate_horizontal():
    image_path = SYSTEM_PARAMS.files.calibration_horizontal
    img = cv2.imread(image_path)
    points = np.array([
        [713, 500],
        [1365, 623]
    ])

    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    print(f"Horizontal line angle: {angle_deg:.2f} degrees")

    for x in points:
        center = (x[0], x[1])
        radius = 3
        color = (0, 0, 255)
        thickness = 2
        cv2.circle(img, center, radius, color, thickness)

    cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0, 255, 0), 2)
    cv2.imshow('image with point and line', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(SYSTEM_PARAMS.files.init_with_circle, img)

    return angle_deg

v = calibrate_vertical()
h = calibrate_horizontal()

res = (abs(v) + abs(h)) / 2 # 10.365
print(f"mean average angle: {res}")
