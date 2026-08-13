import os

import cv2
import numpy as np

from difftactile.data_analysis.experiment.adjacency import *
from difftactile.main.constants import SYSTEM_PARAMS
from difftactile.main.display import destroy_windows, imshow, wait_key
from difftactile.sensor_model.fisheye_model_no_taichi import FisheyeModelNoTaichi


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
        # Preview only: shown briefly, and blocking on a key press is opt-in
        # via DIFFTACTILE_INTERACTIVE=1 so the script always terminates.
        imshow(cv2, "markers", img_with_markers)
        wait_key(cv2, 0)
        destroy_windows(cv2)
    
    @staticmethod
    def extract_reorder_save_markers(path_img_in, path_npz_out, add_manual=False):
        img = cv2.imread(path_img_in)
        marker_centers, circle_center, circle_radius = (
            FisheyeModelNoTaichi.get_marker_image(img)
        )
        if add_manual:
            # The twist_x photograph is the one the blob detector under-reads:
            # it finds 121 of the 127 markers, because six near the rim are
            # foreshortened past recognition by the twist. The grid reordering
            # in get_graph_connectivity() assumes the full 127 and raises
            # IndexError on a short list, so the six are supplied by hand.
            marker_centers = DomainAdaptation.da_twist_x_add_manual_markers(
                marker_centers
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


# The four canonical interactions, mapping each reference photograph of the REAL
# sensor at its apex configuration to the .npz of reordered marker positions that
# `Contact.compute_da_loss()` compares the simulation against.
# The trailing flag is whether the blob detector needs the hand-supplied markers
# (see extract_reorder_save_markers) - only twist_x does.
DA_INTERACTIONS = (
    ("press", "da_press", "da_press_npz", False),
    ("twist_z", "da_twist_z", "da_twist_z_npz", False),
    ("twist_x", "da_twist_x", "da_twist_x_npz", True),
    ("slide", "da_slide", "da_slide_npz", False),
)


def extract_real_marker_positions(force=False):
    """Extract real marker positions from the four DA reference photographs.

    `compute_da_loss()` loads difftactile/output/da_<name>.npz to get the REAL
    marker positions it measures the simulation against, but nothing generated
    those files - the photographs ship with the repository while the .npz did
    not, so domain adaptation died on a missing file. This closes that gap.

    Skips files that already exist unless `force`, since extraction is
    deterministic and the marker detector is the slow part.
    """
    da_dir = SYSTEM_PARAMS.files.da_dir
    written = []
    for name, img_key, npz_key, add_manual in DA_INTERACTIONS:
        img_in = f"{da_dir}{getattr(SYSTEM_PARAMS.files, img_key)}"
        npz_out = getattr(SYSTEM_PARAMS.files, npz_key)
        if os.path.exists(npz_out) and not force:
            print(f"  {name}: already extracted ({npz_out})")
            continue
        if not os.path.exists(img_in):
            raise FileNotFoundError(
                f"reference photograph for '{name}' is missing: {img_in}"
            )
        os.makedirs(os.path.dirname(npz_out), exist_ok=True)
        DomainAdaptation.extract_reorder_save_markers(
            img_in, npz_out, add_manual=add_manual
        )
        points = np.load(npz_out)["points"]
        print(f"  {name}: {len(points)} markers -> {npz_out}")
        written.append(npz_out)
    return written


def main():
    dir = SYSTEM_PARAMS.files.da_dir
    img_in = SYSTEM_PARAMS.files.flat_sensor_default_state
    img_in = f'{dir}{img_in}'
    DomainAdaptation.visualise_markers(img_in)
    npz_out = SYSTEM_PARAMS.files.flat_sensor_default_state_npz
    DomainAdaptation.extract_reorder_save_markers(img_in, npz_out)

if __name__ == "__main__":
    main()
