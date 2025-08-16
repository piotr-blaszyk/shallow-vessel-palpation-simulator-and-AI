import cv2
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from difftactile.main.constants import *

class VoronoiGenerator:
    def __init__(self):
        pass

    def go(self):
        # Load marker positions from npz file
        data = np.load(SYSTEM_PARAMS.files.init_marker_positions_npz)
        points = data['points']  # shape: (num_points, 2)

        # Get the central point (index 91)
        center_point = points[91]
        
        # Compute vectors from center to all points
        vectors = points - center_point  # Broadcasting will create vectors for all points
        
        # Compute magnitudes and angles
        magnitudes = np.sqrt(np.sum(vectors**2, axis=1))
        
        # Compute angles in degrees (clockwise from 12 o'clock)
        # In OpenCV coordinates:
        # - 12 o'clock is (0, -1) = 0 degrees
        # - 3 o'clock is (1, 0) = 90 degrees
        # - 6 o'clock is (0, 1) = 180 degrees
        # - 9 o'clock is (-1, 0) = 270 degrees
        angles = np.degrees(np.arctan2(vectors[:, 0], -vectors[:, 1]))  # Note: x/(-y) for clockwise from 12
        angles = np.mod(angles, 360)  # Ensure angles are in [0, 360)

        line_point_ixs = [
            126,
            102,
            124,
            10,
            2,
            31,
        ]
        line_point_ixs = np.array(line_point_ixs, dtype=int)
        angle_ranges = np.array([
            [angles[39], angles[106]],
            [angles[106], angles[116]],
            [angles[116], angles[12]],
            [angles[12], angles[40]],
            [angles[40], angles[55]],
            [angles[55], angles[39]],
        ], dtype=float)

        projected_magnitudes = np.zeros(shape=magnitudes.shape)

        # Compute line equations for each line from center to specified points
        # Each line equation will be stored as (unit_vector_x, unit_vector_y)
        line_equations = []
        for line_point_idx in line_point_ixs:
            # Get vector from center to point
            line_vector = vectors[line_point_idx]
            # Convert to unit vector
            line_magnitude = np.sqrt(np.sum(line_vector**2))
            unit_vector = line_vector / line_magnitude
            line_equations.append(unit_vector)
        line_equations = np.array(line_equations)

        # Project each point onto its corresponding line based on angle
        for i in range(len(points)):
            if i == 91:  # Skip center point
                projected_magnitudes[i] = 0
                continue
            
            # Find which angle range (and thus which line) this point belongs to
            point_angle = angles[i]
            line_idx = None
            for j, (start_angle, end_angle) in enumerate(angle_ranges):
                # Handle the case where the range crosses 360 degrees
                if start_angle > end_angle:
                    if point_angle >= start_angle or point_angle <= end_angle:
                        line_idx = j
                        break
                else:
                    if start_angle <= point_angle <= end_angle:
                        line_idx = j
                        break
            
            if line_idx is not None:
                # Project the point onto the corresponding line
                # The projection is (v · u)u where v is our vector and u is the unit vector
                vector = vectors[i]
                unit_vector = line_equations[line_idx]
                projection = np.dot(vector, unit_vector) * unit_vector
                
                # Compute magnitude of projection
                projected_magnitudes[i] = np.sqrt(np.sum(projection**2))

        rings = list(range(0, 7))
        rings = [VoronoiGenerator.num_markers_in_ring(x) for x in rings]
        
        # First sort by projected magnitudes to get rough ring ordering
        magnitude_sorted_indices = np.argsort(projected_magnitudes)
        
        # Process each ring separately
        new_indices = []
        start_idx = 0
        for ring_size in rings:
            # Get indices for current ring
            ring_indices = magnitude_sorted_indices[start_idx:start_idx + ring_size]
            
            if ring_size > 1:  # Only sort by angle if more than one point in ring
                # Sort ring points by angle
                ring_angles = angles[ring_indices]
                angle_sorted_indices = np.argsort(ring_angles)
                ring_indices = ring_indices[angle_sorted_indices]
            
            new_indices.extend(ring_indices)
            start_idx += ring_size
        
        new_indices = np.array(new_indices)
        # Create a mapping from old indices to new indices
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_indices)}

        # Load the default state image
        img = cv2.imread(SYSTEM_PARAMS.files.vitactip_photo_default_state)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {SYSTEM_PARAMS.files.vitactip_photo_default_state}")

        # Draw each point with its new index
        for i, (x, y) in enumerate(points):
            # Convert to integer coordinates
            x, y = int(x), int(y)
            
            # Draw a circle at the point
            cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # Red filled circle
            
            # Add text label with new index (if point has one)
            if i in index_mapping:
                new_idx = index_mapping[i]
                cv2.putText(img, str(new_idx), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           fontScale=0.5, color=(0, 255, 0), thickness=1)  # Green text

        # Display the image
        cv2.imshow('Marker Positions', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the image
        cv2.imwrite(SYSTEM_PARAMS.files.voronoi_image, img)
    
    @staticmethod
    def num_markers_in_ring(k):
        if k == 0:
            return 1
        else:
            return k * 6


def main():
    v = VoronoiGenerator()
    v.go()


if __name__ == '__main__':
    main()
