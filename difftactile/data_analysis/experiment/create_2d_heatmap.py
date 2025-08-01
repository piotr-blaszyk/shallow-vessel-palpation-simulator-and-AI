import numpy as np
import cv2

from difftactile.data_analysis.experiment.marker_tracker import MarkerTracker
from difftactile.main.constants import *

def linear_interpolation():
    cap = cv2.VideoCapture(str(SYSTEM_PARAMS.files.vein_slide_across))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"total_frames: {total_frames}")

    poses = np.array([
        [-382.1576,   85.3686,   28.0000,   180.0,   0.0,   0.0],  # target 0
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
        [-382.1576,  -74.6314,   24.0000,   180.0,   0.0,   0.0],  # target 17
        [-317.1576,  -74.6314,   24.0000,   180.0,   0.0,   0.0],  # target 18
        [-317.1576,  -94.6314,   24.0000,   180.0,   0.0,   0.0],  # target 19
    ])

    video_frames = np.array([
        60,    # target 0
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
        1114,  # target 17
        1207,  # target 18
        1243,  # target 19
    ])

    positions = poses[:, :3]
    all_positions = np.zeros(shape=(video_frames[-1]+1, 3))
    all_positions[video_frames] = positions
    
    for i in range(len(video_frames) - 1):
        start_frame = video_frames[i]
        end_frame = video_frames[i + 1]
        start_pos = positions[i]
        end_pos = positions[i + 1]
        
        num_frames = end_frame - start_frame
        
        for j in range(1, num_frames):
            t = j / num_frames
            interpolated_pos = (1 - t) * start_pos + t * end_pos
            all_positions[start_frame + j] = interpolated_pos
    
    return all_positions


def main():
    marker_tracker = MarkerTracker()
    marker_tracker.extract_frames(
        SYSTEM_PARAMS.files.vein_slide_across
    )
    positions = linear_interpolation()
    foo = 7
