import pickle

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from difftactile.cnn.dataset import *
from difftactile.data_analysis.experiment.adjacency import *
from difftactile.main.constants import *
from difftactile.sensor_model.fisheye_model_no_taichi import FisheyeModelNoTaichi


class ComputeEdges:
    def __init__(self):
        pass

    @staticmethod
    def _find_closest_unique_indices(points, target_points):
        points = np.asarray(points, dtype=float)
        target_points = np.asarray(target_points, dtype=float)
        selected_indices = []
        used_indices = set()

        for target in target_points:
            distances = np.linalg.norm(points - target, axis=1)
            if used_indices:
                distances[list(used_indices)] = np.inf
            closest_idx = int(np.argmin(distances))
            selected_indices.append(closest_idx)
            used_indices.add(closest_idx)

        return np.array(selected_indices, dtype=int)

    @staticmethod
    def compute_base_graph_connectivity():
        img = cv2.imread(
            SYSTEM_PARAMS.files.iros_sensor_default_state,
            cv2.IMREAD_GRAYSCALE,
        )
        if img is None:
            raise FileNotFoundError(
                f"Could not load image from {SYSTEM_PARAMS.files.iros_sensor_default_state}"
            )
        marker_data = np.load(SYSTEM_PARAMS.files.init_marker_positions_npz)
        points = marker_data["points"]

        detected_markers_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for marker_idx, (x, y) in enumerate(points):
            x, y = int(x), int(y)
            cv2.circle(detected_markers_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                detected_markers_img,
                str(marker_idx),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1,
            )
        detected_markers_image_path = SYSTEM_PARAMS.files.voronoi_image.rsplit('.', 1)[0] + "-detected-markers.png"
        cv2.imwrite(detected_markers_image_path, detected_markers_img)

        if len(points) != 127:
            raise ValueError(f"Expected 127 markers, but found {len(points)}")

        key_marker_targets = np.array([
            [1039, 582],
            [1199, 300],
            [1366, 576],
            [1211, 858],
            [888, 868],
            [720, 592],
            [874, 301],
        ], dtype=float)
        key_marker_indices = ComputeEdges._find_closest_unique_indices(
            points,
            key_marker_targets,
        )
        center_idx = int(key_marker_indices[0])
        line_point_ixs = key_marker_indices[1:]

        key_marker_img = cv2.imread(SYSTEM_PARAMS.files.iros_sensor_default_state)
        if key_marker_img is None:
            raise FileNotFoundError(f"Could not load image from {SYSTEM_PARAMS.files.iros_sensor_default_state}")
        for i, marker_idx in enumerate(key_marker_indices):
            x, y = map(int, points[int(marker_idx)])
            cv2.circle(key_marker_img, (x, y), radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                key_marker_img,
                str(i),
                (x + 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=2,
            )
        key_markers_image_path = SYSTEM_PARAMS.files.voronoi_image.rsplit('.', 1)[0] + "-key-markers.png"
        cv2.imwrite(key_markers_image_path, key_marker_img)

        center_point = points[center_idx]
        vectors = points - center_point
        magnitudes = np.sqrt(np.sum(vectors**2, axis=1))
        angles = np.degrees(np.arctan2(vectors[:, 0], -vectors[:, 1]))
        angles = np.mod(angles, 360)

        line_point_ixs = np.array(line_point_ixs, dtype=int)
        corner_angles = angles[line_point_ixs]
        angle_ranges = np.array([
            [corner_angles[0], corner_angles[1]],
            [corner_angles[1], corner_angles[2]],
            [corner_angles[2], corner_angles[3]],
            [corner_angles[3], corner_angles[4]],
            [corner_angles[4], corner_angles[5]],
            [corner_angles[5], corner_angles[0]],
        ], dtype=float)

        projected_magnitudes = np.zeros(shape=magnitudes.shape)
        line_equations = []
        for line_point_idx in line_point_ixs:
            line_vector = vectors[line_point_idx]
            line_magnitude = np.sqrt(np.sum(line_vector**2))
            unit_vector = line_vector / line_magnitude
            line_equations.append(unit_vector)
        line_equations = np.array(line_equations)

        for i in range(len(points)):
            if i == center_idx:
                projected_magnitudes[i] = 0
                continue

            point_angle = angles[i]
            line_idx = None
            for j, (start_angle, end_angle) in enumerate(angle_ranges):
                if start_angle > end_angle:
                    if point_angle >= start_angle or point_angle <= end_angle:
                        line_idx = j
                        break
                else:
                    if start_angle <= point_angle <= end_angle:
                        line_idx = j
                        break

            if line_idx is not None:
                vector = vectors[i]
                unit_vector = line_equations[line_idx]
                projection = np.dot(vector, unit_vector) * unit_vector
                projected_magnitudes[i] = np.sqrt(np.sum(projection**2))

        rings = list(range(0, 7))
        rings = [ComputeEdges.num_markers_in_ring(x) for x in rings]

        remaining_indices = np.array([i for i in range(len(points)) if i != center_idx], dtype=int)
        radial_sorted_indices = remaining_indices[np.argsort(magnitudes[remaining_indices])]

        new_indices = [center_idx]
        start_idx = 0
        for ring_size in rings[1:]:
            ring_indices = radial_sorted_indices[start_idx:start_idx + ring_size]
            ring_angles = angles[ring_indices]
            angle_sorted_indices = np.argsort(ring_angles)
            ring_indices = ring_indices[angle_sorted_indices]
            new_indices.extend(ring_indices)
            start_idx += ring_size

        new_indices = np.array(new_indices)
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_indices)}
        points = points[new_indices]

        ordered_indices = np.arange(127, dtype=int)
        ix_thresh_4_neighbours = 91
        ixs_3_neighbours = np.sort(
            np.array([index_mapping[int(ix)] for ix in line_point_ixs], dtype=int)
        )

        neighbour_counts = 6 * np.ones(shape=ordered_indices.shape, dtype=int)
        neighbour_counts[ordered_indices >= ix_thresh_4_neighbours] = 4
        mask = np.isin(ordered_indices, ixs_3_neighbours)
        neighbour_counts[mask] = 3

        k = 6
        knn = KNeighborsClassifier(n_neighbors=k+1, metric='euclidean')
        knn.fit(points, np.zeros(len(points)))
        distances, neighbors = knn.kneighbors(points)

        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        edges = []
        for i in range(len(points)):
            n_neighbors = neighbour_counts[i]

            node_distances = distances[i, :n_neighbors]
            node_neighbors = neighbors[i, :n_neighbors]

            sort_idx = np.argsort(node_distances)
            node_neighbors = node_neighbors[sort_idx]

            for neighbor in node_neighbors:
                edges.append([i, neighbor])

        adjacency_matrix = np.array(edges, dtype=int)

        ring_ixs_all = np.zeros(shape=(127,), dtype=int)
        ix = 0
        for i in range(len(rings)):
            ring_size = rings[i]
            for j in range(ring_size):
                ring_ixs_all[ix] = i
                ix += 1

        vectors = points - points[0]
        angles = np.zeros((len(points), 2))
        norms = np.linalg.norm(vectors[1:], axis=1)
        normalized_vectors = vectors[1:] / norms[:, np.newaxis]
        angles[1:] = normalized_vectors

        norms_all = np.linalg.norm(vectors, axis=1)

        np.savez(
            SYSTEM_PARAMS.files.base_graph_connectivity,
            adjacency_matrix=adjacency_matrix,
            points=points,
            ring_ixs=ring_ixs_all,
            angles=angles,
            dist_from_centre=norms_all,
        )

        img = cv2.imread(SYSTEM_PARAMS.files.iros_sensor_default_state)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {SYSTEM_PARAMS.files.iros_sensor_default_state}")

        for i, (x, y) in enumerate(points):
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 0), thickness=1)

        for edge in adjacency_matrix:
            start_idx, end_idx = edge
            start_point = tuple(map(int, points[start_idx]))
            end_point = tuple(map(int, points[end_idx]))
            cv2.line(img, start_point, end_point, color=(255, 255, 0), thickness=1)

        cv2.imwrite(SYSTEM_PARAMS.files.voronoi_image, img)

        apple = np.load(SYSTEM_PARAMS.files.init_marker_positions_npz)
        peas = apple['points']
        conn = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        _, mango, _ = Adjacency.get_graph_connectivity_helper(conn, peas)
        np.savez(
            SYSTEM_PARAMS.files.iros_marker_locations_ordered,
            points=mango,
        )

    @staticmethod
    def visualise_flat_sensor():
        # Load the default state image
        dir = SYSTEM_PARAMS.files.da_dir
        file_path = SYSTEM_PARAMS.files.flat_sensor_default_state
        file_path = f'{dir}{file_path}'
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {SYSTEM_PARAMS.files.flat_sensor_default_state}")

        path = SYSTEM_PARAMS.files.iros_marker_locations_ordered
        data = np.load(path)
        points = data['points']

        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        adjacency_matrix = base_graph_data['adjacency_matrix']

        # Draw each point with its new index
        for i, (x, y) in enumerate(points):
            # Convert to integer coordinates
            x, y = int(x), int(y)
            
            # Draw a circle at the point
            cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # Red filled circle
            cv2.putText(img, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=(0, 0, 255), thickness=2)  # Green text

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
        cv2.imwrite('difftactile/output/flat_sensor_connectivity.png', img)

    @staticmethod
    def num_markers_in_ring(k):
        if k == 0:
            return 1
        else:
            return k * 6
    
    @staticmethod
    def validate_graph_connectivity_algorithm():
        dataset = MyDataset(mode='dummy', scheme='new')
        with open(SYSTEM_PARAMS.files.test_loader_gnn_icra, 'rb') as f:
            test_data = pickle.load(f)
        stats = test_data['dataset_stats'][1.0]
        dataset.set_stats(stats)
        dataset.set_difficulty_level(1.0)

        n = len(dataset)
        k = 2
        
        # Create shuffled array of indices
        indices = np.arange(n, dtype=int)
        NP_RNG.shuffle(indices)
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
    # ComputeEdges.visualise_flat_sensor()


if __name__ == '__main__':
    main()
