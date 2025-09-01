
import numpy as np

from difftactile.cnn.dataset import *
from difftactile.cnn.gnn import *
from difftactile.cnn.visualise import *
from difftactile.data_analysis.experiment.hungarian_exp import *
from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.main.constants import *
from difftactile.sensor_model.fisheye_model_no_taichi import *


def main():
    poses = np.array([
        [-382.1576,   85.3686,   24.0000,   180.0,   0.0,   0.0],  # target 1
        [-317.1576,   85.3686,   24.0000,   180.0,   0.0,   0.0],  # target 2
        [-317.1576,   65.3686,   24.0000,   180.0,   0.0,   0.0],  # target 3
        [-382.1576,   65.3686,   24.0000,   180.0,   0.0,   0.0],  # target 4
        [-382.1576,   45.3686,   24.0000,   180.0,   0.0,   0.0],  # target 5
        [-317.1576,   45.3686,   24.0000,   180.0,   0.0,   0.0],  # target 6
        [-317.1576,   25.3686,   24.0000,   180.0,   0.0,   0.0],  # target 7
        [-382.1576,   25.3686,   24.0000,   180.0,   0.0,   0.0],  # target 8
        [-382.1576,    5.3686,   24.0000,   180.0,   0.0,   0.0],  # target 9
        [-317.1576,    5.3686,   24.0000,   180.0,   0.0,   0.0],  # target 10
        [-317.1576,  -14.6314,   24.0000,   180.0,   0.0,   0.0],  # target 11
        [-382.1576,  -14.6314,   24.0000,   180.0,   0.0,   0.0],  # target 12
        [-382.1576,  -34.6314,   24.0000,   180.0,   0.0,   0.0],  # target 13
        [-317.1576,  -34.6314,   24.0000,   180.0,   0.0,   0.0],  # target 14
        [-317.1576,  -54.6314,   24.0000,   180.0,   0.0,   0.0],  # target 15
        [-382.1576,  -54.6314,   24.0000,   180.0,   0.0,   0.0],  # target 16
    ], dtype=float)
    frames = np.array([
        77,    # target 1
        169,   # target 2
        205,   # target 3
        297,   # target 4
        333,   # target 5
        425,   # target 6
        461,   # target 7
        554,   # target 8
        590,   # target 9
        683,   # target 10
        719,   # target 11
        812,   # target 12
        849,   # target 13
        941,   # target 14
        978,   # target 15
        1077,  # target 16
    ], dtype=int)
    frames_poses = np.zeros((len(frames), 7))
    frames_poses[:, 0] = frames
    frames_poses[:, 1:] = poses
    path = SYSTEM_PARAMS.files.experiment_og_frames_poses_npz
    np.savez(
        path,
        output=frames_poses,
    )

if __name__ == '__main__':
    main()
