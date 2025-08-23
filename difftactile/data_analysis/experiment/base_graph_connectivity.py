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
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from difftactile.main.constants import *
from difftactile.cnn.dataset import *
from difftactile.data_analysis.experiment.adjacency import *

class ComputeEdges:
    def __init__(self):
        pass

    @staticmethod
    def compute_base_graph_connectivity():
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
        rings = [ComputeEdges.num_markers_in_ring(x) for x in rings]
        
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

        # Reorder points array according to the index mapping
        points = points[new_indices]

        ordered_indices = np.arange(127, dtype=int)

        ix_thresh_4_neighbours = 91

        ixs_3_neighbours = np.array([
            94, 100, 106, 112, 118, 124
        ], dtype=int)

        # Initialize with 6 neighbors for all points
        neighbour_counts = 6 * np.ones(shape=ordered_indices.shape, dtype=int)
        
        # Set 4 neighbors for points with index >= ix_thresh_4_neighbours
        neighbour_counts[ordered_indices >= ix_thresh_4_neighbours] = 4
        
        # Set 3 neighbors for specific points using vectorized operation
        mask = np.isin(ordered_indices, ixs_3_neighbours)
        neighbour_counts[mask] = 3

        # Find k nearest neighbors for each point
        k = 6  # maximum number of neighbors
        knn = KNeighborsClassifier(n_neighbors=k+1, metric='euclidean')  # k+1 because point itself is included
        knn.fit(points, np.zeros(len(points)))  # dummy labels
        distances, neighbors = knn.kneighbors(points)
        
        # Remove self-connections (first column)
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]
        
        # Create edges list with sorting by distance for each node
        edges = []
        for i in range(len(points)):
            # Get number of neighbors for this node
            n_neighbors = neighbour_counts[i]
            
            # Sort neighbors by distance
            node_distances = distances[i, :n_neighbors]
            node_neighbors = neighbors[i, :n_neighbors]
            
            # Sort indices by distance
            sort_idx = np.argsort(node_distances)
            node_neighbors = node_neighbors[sort_idx]
            
            # Add edges in both directions
            for neighbor in node_neighbors:
                edges.append([i, neighbor])
        
        # Convert to numpy array
        adjacency_matrix = np.array(edges, dtype=int)

        ring_ixs_all = np.zeros(shape=(127,), dtype=int)
        ix = 0
        for i in range(len(rings)):
            ring_size = rings[i]
            for j in range(ring_size):
                ring_ixs_all[ix] = i
                ix += 1
        
        # Compute angles relative to points[0]
        vectors = points - points[0]  # Get vectors from points[0] to each point
        angles = np.zeros((len(points), 2))  # Will store (cos(θ), sin(θ))
        
        # Skip points[0] as it should remain (0,0)
        norms = np.linalg.norm(vectors[1:], axis=1)
        normalized_vectors = vectors[1:] / norms[:, np.newaxis]
        
        # cos(θ) is x/r, sin(θ) is y/r after normalization
        angles[1:] = normalized_vectors  # Already normalized so x=cos(θ), y=sin(θ)

        # Save adjacency matrix and points to npz file
        np.savez(
            SYSTEM_PARAMS.files.base_graph_connectivity,
            adjacency_matrix=adjacency_matrix,
            points=points,
            ring_ixs=ring_ixs_all,
            angles=angles,
        )

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
            
            degree = neighbour_counts[i]
            cv2.putText(img, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=(0, 255, 0), thickness=1)  # Green text

        # Draw edges between connected nodes
        for edge in adjacency_matrix:
            start_idx, end_idx = edge
            start_point = tuple(map(int, points[start_idx]))
            end_point = tuple(map(int, points[end_idx]))
            cv2.line(img, start_point, end_point, color=(255, 255, 0), thickness=1)  # Yellow lines

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
    
    @staticmethod
    def validate_graph_connectivity_algorithm():
        dataset = MyDataset(mode='dummy', scheme='new')
        with open(SYSTEM_PARAMS.files.test_loader_gnn, 'rb') as f:
            test_data = pickle.load(f)
        stats = test_data['dataset_stats'][1.0]
        dataset.set_stats(stats)
        dataset.set_difficulty_level(1.0)

        n = len(dataset)
        k = 2
        
        # Create shuffled array of indices
        indices = np.arange(n, dtype=int)
        np.random.shuffle(indices)
        current_ix = 0
        
        while True:
            ix = indices[current_ix]
            points = dataset.get_points(ix)
            
            # Skip if any point has (0,0) coordinates
            # if np.any(np.all(points == 0, axis=1)):
            #     current_ix = (current_ix + 1) % n
            #     continue
                
            base_points, points, adjacency_matrix = Adjacency.get_graph_connectivity(points)

            base_points /= k
            points /= k
            
            # Create black image
            img = np.zeros((1080 // k, 1920 // k, 3), dtype=np.uint8)

            mode = 'adjacency_matrix'
            
            if mode == 'hungarian':
                # Draw correspondence edges in green
                for i in range(len(points)):
                    start_point = tuple(map(int, points[i]))
                    end_point = tuple(map(int, base_points[i]))
                    cv2.line(img, start_point, end_point, color=(0, 255, 0), thickness=1)
                
                # Draw input points in red
                for point in points:
                    x, y = map(int, point)
                    cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                
                # Draw base points in blue
                for point in base_points:
                    x, y = map(int, point)
                    cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
            elif mode == 'adjacency_matrix':
                # Draw edges from adjacency matrix in green
                for edge in adjacency_matrix:
                    start_idx, end_idx = edge
                    start_point = tuple(map(int, points[start_idx]))
                    end_point = tuple(map(int, points[end_idx]))
                    cv2.line(img, start_point, end_point, color=(0, 255, 0), thickness=1)
                
                # Draw points in red
                for point in points:
                    x, y = map(int, point)
                    cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            
            # Display image and index
            cv2.putText(img, f"Index: {ix} ({current_ix}/{n})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            cv2.imshow('Graph Connectivity Validation', img)
            key = cv2.waitKey(0)
            
            if key == ord('q'):
                break
            elif key == ord('j'):
                # Go back one image
                current_ix = (current_ix - 1) % n
            elif key == ord('k'):
                # Go forward one image
                current_ix = (current_ix + 1) % n
        
        cv2.destroyAllWindows()


def main():
    ComputeEdges.compute_base_graph_connectivity()
    # ComputeEdges.validate_graph_connectivity_algorithm()


if __name__ == '__main__':
    main()
