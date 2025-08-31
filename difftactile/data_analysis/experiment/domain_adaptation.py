import cv2
import numpy as np

from difftactile.main.constants import SYSTEM_PARAMS
from difftactile.sensor_model.fisheye_model_no_taichi import FisheyeModelNoTaichi
from difftactile.data_analysis.experiment.adjacency import *


class DomainAdaptation:
    def __init__(self):
        pass

    @staticmethod
    def visualise_markers(file_path):
        img = cv2.imread(file_path)
        marker_centers, circle_center, circle_radius = (
            FisheyeModelNoTaichi.get_marker_image(img)
        )
        img_with_markers = img.copy()
        for center in marker_centers:
            x, y = center.astype(int)
            cv2.circle(img_with_markers, (x, y), 5, (0, 255, 0), -1)
        circle_center_int = circle_center.astype(int)
        cv2.circle(
            img_with_markers,
            tuple(circle_center_int),
            int(circle_radius),
            (255, 0, 0),
            2,
        )
        cv2.imshow("markers", img_with_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @staticmethod
    def extract_reorder_save_markers(path_img_in, path_npz_out):
        img = cv2.imread(path_img_in)
        marker_centers, circle_center, circle_radius = (
            FisheyeModelNoTaichi.get_marker_image(img)
        )
        _, points_reordered, _ = Adjacency.get_graph_connectivity(marker_centers)
        np.savez(
            path_npz_out,
            points=points_reordered,
        )
    
    @staticmethod
    def da_twist_x_add_manual_markers(marker_centers):
        manual = np.array([
            [762, 482],
            [845, 433],
            [1012, 430],
            [1067, 430],
            [1122, 430],
            [1197, 299],
        ], dtype=float)
        marker_centers = np.concatenate([marker_centers, manual], axis=0)
        return marker_centers


def main():
    dir = SYSTEM_PARAMS.files.da_dir
    img_in = SYSTEM_PARAMS.files.flat_sensor_default_state
    img_in = f'{dir}{img_in}'
    # DomainAdaptation.visualise_markers(img_in)
    npz_out = SYSTEM_PARAMS.files.flat_sensor_default_state_npz
    # DomainAdaptation.extract_reorder_save_markers(img_in, npz_out)

if __name__ == "__main__":
    main()
