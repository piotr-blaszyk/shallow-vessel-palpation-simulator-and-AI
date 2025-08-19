import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import cv2

from difftactile.main.constants import *


class HungarianExp:
    @staticmethod
    def reorder_exp_points_complicated():
        return
        path = SYSTEM_PARAMS.files.exp_video_npz
        data = np.load(path)
        points = data['markers']
        points_mask = data['markers_mask']
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        base_points = base_graph_data['points']
        adjacency = base_graph_data['adjacency_matrix']
        points_reordered_list = []
        points_mask_reordered_list = []
        edge_attrs_list = []

        cost_matrix = cdist(points[0], base_points, metric='sqeuclidean')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        inverse_mapping = np.zeros_like(row_ind)
        inverse_mapping[col_ind] = row_ind
        points_reordered = points[0, inverse_mapping, :]
        points_mask_reordered = points_mask[0, inverse_mapping]
        points_reordered_list.append(points_reordered)
        points_mask_reordered_list.append(points_mask_reordered)

        me_points = points_reordered[adjacency[:, 0]]
        neighbor_points = points_reordered[adjacency[:, 1]]
        disp = neighbor_points - me_points
        edge_attrs_list.append(disp)

        for i in range(1, points.shape[0]):
            prev_points = points_reordered_list[-1]
            prev_point_mask = points_mask_reordered_list[-1]
            break
    
    @staticmethod
    def reorder_exp_points_simple():
        input_path = SYSTEM_PARAMS.files.exp_video_npz
        output_path = SYSTEM_PARAMS.files.exp_video_npz_reordered
        data = np.load(input_path)
        all_points = data['markers']
        all_points_mask = data['markers_mask']
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        base_points = base_graph_data['points']
        
        all_reordered_points_list = []
        all_reordered_points_mask_list = []

        for i in range(all_points.shape[0]):
            points = all_points[i]
            points_mask = all_points_mask[i]
            points = points[points_mask]  # Get only valid points for this frame

            if points.shape[0] < base_points.shape[0]:
                continue

            # Calculate cost matrix between detected points and base points
            cost_matrix = cdist(points, base_points, metric='sqeuclidean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Create reordered points array of correct size
            reordered_points = np.zeros((base_points.shape[0], 2))
            reordered_mask = np.zeros(base_points.shape[0], dtype=bool)
            
            # Only take the first base_points.shape[0] assignments
            for base_idx, detected_idx in zip(col_ind[:base_points.shape[0]], 
                                            row_ind[:base_points.shape[0]]):
                reordered_points[base_idx] = points[detected_idx]
                reordered_mask[base_idx] = True
            
            all_reordered_points_list.append(reordered_points)
            all_reordered_points_mask_list.append(reordered_mask)

        # Convert lists to numpy arrays
        all_reordered_points = np.array(all_reordered_points_list)
        all_reordered_points_mask = np.array(all_reordered_points_mask_list)

        np.savez(
            output_path,
            markers=all_reordered_points,
            markers_mask=all_reordered_points_mask
        )
    

    
    @staticmethod
    def visualise_reordered_point_connectivity():
        output_path = SYSTEM_PARAMS.files.exp_video_npz_reordered
        data = np.load(output_path)
        all_points = data['markers']
        all_points_mask = data['markers_mask']
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        adjacency = base_graph_data['adjacency_matrix']

        # Z-normalize x and y coordinates independently
        x_coords = all_points[..., 0]
        y_coords = all_points[..., 1]
        x_mean, x_std = np.mean(x_coords[all_points_mask]), np.std(x_coords[all_points_mask])
        y_mean, y_std = np.mean(y_coords[all_points_mask]), np.std(y_coords[all_points_mask])
        
        normalized_points = np.zeros_like(all_points)
        normalized_points[..., 0] = (x_coords - x_mean) / x_std
        normalized_points[..., 1] = (y_coords - y_mean) / y_std

        # Scale to 800x800 square and add padding
        canvas_size = 900
        display_size = 800
        padding = 50  # This ensures (1000-800)/2 = 100 pixels padding on each side
        
        # Scale normalized points to display size and center in canvas
        scaled_points = np.zeros_like(normalized_points)
        scaled_points[..., 0] = (normalized_points[..., 0] * (display_size/4)) + (canvas_size/2)
        scaled_points[..., 1] = (normalized_points[..., 1] * (display_size/4)) + (canvas_size/2)

        current_frame = 0
        num_frames = all_points.shape[0]

        while True:
            # Create black canvas
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

            # Draw edges (green lines)
            for edge in adjacency:
                if all_points_mask[current_frame, edge[0]] and all_points_mask[current_frame, edge[1]]:
                    pt1 = tuple(map(int, scaled_points[current_frame, edge[0]]))
                    pt2 = tuple(map(int, scaled_points[current_frame, edge[1]]))
                    cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

            # Draw nodes (red circles)
            for i in range(all_points.shape[1]):
                if all_points_mask[current_frame, i]:
                    pt = tuple(map(int, scaled_points[current_frame, i]))
                    cv2.circle(canvas, pt, 5, (0, 0, 255), -1)

            # Draw the display area border
            cv2.rectangle(canvas, 
                         (padding, padding), 
                         (canvas_size-padding, canvas_size-padding), 
                         (128, 128, 128), 2)

            # Display frame number
            cv2.putText(canvas, f'Frame: {current_frame}/{num_frames-1}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the image
            cv2.imshow('Graph Visualization', canvas)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('j'):  # Previous frame
                current_frame = (current_frame - 1) % num_frames
            elif key == ord('k'):  # Next frame
                current_frame = (current_frame + 1) % num_frames

        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main():
    # HungarianExp.reorder_exp_points_simple()
    HungarianExp.visualise_reordered_point_connectivity()
