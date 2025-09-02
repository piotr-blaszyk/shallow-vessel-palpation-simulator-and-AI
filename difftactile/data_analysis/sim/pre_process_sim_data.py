from difftactile.main.constants import *

if SYSTEM_PARAMS.meta.cnn_gnn == 0:
    pass

import os

import numpy as np
from tqdm import tqdm

from difftactile.data_analysis.experiment.adjacency import *
from difftactile.main.synthetic_image_generator import *
from difftactile.sensor_model.fisheye_model_no_taichi import *


class PreProcessSimData:
    @staticmethod
    def sim_marker_tracker():
        input_dir = "difftactile/output/training_data/pickle_20250901_220921"
        output_dir = "difftactile/output/training_data/pickle_20250901_220921_reordered"
        os.makedirs(output_dir, exist_ok=True)
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        base_points = base_graph_data['points']
        file_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        for file_path in tqdm(file_paths, desc="Processing marker data files"):
            data = np.load(file_path)
            points = data['markers']
            points_mask = data['markers_mask']
            labels = data['vein_polyline']
            labels_mask = data['vein_polyline_mask']
            target_id_array = data['target_id_array']

            file_base_points = points[0]
            cost_matrix = cdist(file_base_points, base_points, metric='sqeuclidean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            inverse_mapping = np.zeros_like(row_ind)
            inverse_mapping[col_ind] = row_ind

            points_reordered = points[:, inverse_mapping, :]
            points_mask_reordered = points_mask[:, inverse_mapping]
            file_name = os.path.basename(file_path)
            output_path = f'{output_dir}/{file_name}'
            np.savez(
                output_path,
                markers=points_reordered,
                markers_mask=points_mask_reordered,
                vein_polyline=labels,
                vein_polyline_mask=labels_mask,
                target_id_array=target_id_array
            )
    
    @staticmethod
    def smooth():
        input_dir = SYSTEM_PARAMS.files.dataset_root_yesterday_reordered
        output_dir = SYSTEM_PARAMS.files.dataset_root_yesterday_reordered_smoothed
        file_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path in tqdm(file_paths, desc="Processing marker data files"):
            data = np.load(file_path)
            points = data['markers']
            points_mask = data['markers_mask']
            labels = data['vein_polyline']
            labels_mask = data['vein_polyline_mask']
            target_id_array = data['target_id_array']
            
            num_frames, num_points, _ = points.shape
            k = min(11, num_frames)
            kernel = np.ones(k) / k
            smoothed_points = np.zeros((num_frames - k + 1, num_points, 2))  # Adjusted size for 'valid' mode
            
            for i in range(num_points):
                x_coords = points[:, i, 0]
                smoothed_points[:, i, 0] = np.convolve(x_coords, kernel, mode='valid')
                y_coords = points[:, i, 1]
                smoothed_points[:, i, 1] = np.convolve(y_coords, kernel, mode='valid')

            # Trim other arrays to match the smoothed points range
            valid_frames = slice(k - 1, num_frames)  # The range where valid convolution results exist
            points_mask = points_mask[valid_frames]
            labels = labels[valid_frames]
            labels_mask = labels_mask[valid_frames]

            output_path = os.path.join(output_dir, os.path.basename(file_path))
            np.savez(
                output_path,
                markers=smoothed_points,
                markers_mask=points_mask,
                vein_polyline=labels,
                vein_polyline_mask=labels_mask,
                target_id_array=target_id_array
            )
    
    def rename_npz():
        pass


def main():
    PreProcessSimData.sim_marker_tracker()


if __name__ == '__main__':
    main()
