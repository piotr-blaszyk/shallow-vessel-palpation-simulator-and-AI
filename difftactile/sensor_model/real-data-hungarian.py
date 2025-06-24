import cv2
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def load_data():
    # Load experimental data
    with open('vascular-tumour-press-experimental-results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Load initial marker positions
    with open('init-marker-positions.pkl', 'rb') as f:
        init_marker_positions = pickle.load(f)
    
    marker_positions = data['marker_positions']  # Shape: (num_points, num_markers, 2)
    
    # Scale marker positions from 1920x1080 to 640x480
    scale_x = 640 / 1920
    scale_y = 480 / 1080
    marker_positions[..., 0] *= scale_x
    marker_positions[..., 1] *= scale_y
    
    return marker_positions, init_marker_positions

def match_markers(current_markers, init_markers):
    """Match markers using the Hungarian algorithm."""
    # Compute cost matrix (squared Euclidean distances)
    cost_matrix = cdist(current_markers, init_markers, metric='sqeuclidean')
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return row_ind, col_ind

def draw_frame(frame_idx, marker_positions, init_markers):
    """Draw current frame with markers and displacement arrows."""
    # Create a blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    current_markers = marker_positions[frame_idx]
    
    # Match markers
    row_ind, col_ind = match_markers(current_markers, init_markers)
    
    # Draw initial markers (red)
    for pos in init_markers:
        cv2.circle(img, tuple(pos.astype(int)), 3, (0, 0, 255), -1)
    
    # Draw current markers (green) and displacement arrows (blue)
    for curr_idx, init_idx in zip(row_ind, col_ind):
        curr_pos = tuple(current_markers[curr_idx].astype(int))
        init_pos = tuple(init_markers[init_idx].astype(int))
        
        # Draw current marker
        cv2.circle(img, curr_pos, 3, (0, 255, 0), -1)
        
        # Draw arrow
        cv2.arrowedLine(img, init_pos, curr_pos, (255, 0, 0), 1, tipLength=0.2)
    
    # Add frame number
    cv2.putText(img, f'Frame: {frame_idx}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def main():
    marker_positions, init_markers = load_data()
    num_frames = len(marker_positions)
    frame_idx = 0
    
    print("Controls:")
    print("j: Previous frame")
    print("l: Next frame")
    print("q: Quit")
    
    while True:
        img = draw_frame(frame_idx, marker_positions, init_markers)
        cv2.imshow('Marker Matching Visualization', img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('j'):  # Previous frame
            frame_idx = (frame_idx - 1) % num_frames
        elif key == ord('l'):  # Next frame
            frame_idx = (frame_idx + 1) % num_frames
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
