
import cv2
import numpy as np

from difftactile.main.constants import *


def go(frame_ix, x):
    image_path = SYSTEM_PARAMS.files.traj_out_snapshot.format(1, frame_ix)
    img = cv2.imread(image_path)
    center = (x[0], x[1])
    radius = 3
    color = (0, 0, 255)
    thickness = 2
    cv2.circle(img, center, radius, color, thickness)
    cv2.imshow(f'frame {frame_ix}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

go(103, np.array([935, 881]))
go(179, np.array([1197, 899]))
