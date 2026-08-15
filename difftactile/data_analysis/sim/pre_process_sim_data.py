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
        """Reorder each trajectory's markers into the base-graph ordering.

        The simulator emits markers in an arbitrary per-trajectory order; the
        GNN needs them in the fixed base-graph order, so frame 0 is matched to
        the base layout by Hungarian assignment and the whole trajectory is
        re-indexed with that mapping.

        DIFFTACTILE_SIM_RAW_DIR selects the raw collection directory (the
        `pickle_<timestamp>` folder `script_main` wrote); the reordered dataset
        is written beside it as `<input>_reordered_dense`, the layout
        `SYSTEM_PARAMS.files.sim_data` expects. The default is the raw
        directory behind the published dataset, which the Zenodo bundle does
        NOT ship (only the final `_reordered_dense` is published) - so a fresh
        clone must set the env var to a directory it has actually collected.
        """
        input_dir = os.environ.get(
            "DIFFTACTILE_SIM_RAW_DIR",
            "difftactile/output/training_data/pickle_20250901_220921",
        )
        if not os.path.isabs(input_dir):
            input_dir = repo_path(input_dir)
        input_dir = input_dir.rstrip("/")
        output_dir = f"{input_dir}_reordered_dense"
        print(f"reordering {input_dir} -> {output_dir}")
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

            # The dataset loader (cnn/dataset.py getitem, "single_dataset"
            # scheme) requires `vein_classification` and `vein_regression` keys,
            # matching the published dataset's schema. Neither is consumed by
            # any model or metric - they are carried through the PyG Data object
            # unused - so they are derived here rather than in the simulator:
            # classification is "does this frame contain a vein" (exactly what
            # the published files hold, verified equal to the per-frame OR of
            # vein_polyline_mask), and regression is zeros of the published
            # (frames, num_veins, 3) shape.
            vein_classification = labels_mask.any(axis=2).astype(np.int32)
            vein_regression = np.zeros(
                (labels.shape[0], labels.shape[1], 3), dtype=np.float32
            )

            file_name = os.path.basename(file_path)
            output_path = f'{output_dir}/{file_name}'
            np.savez(
                output_path,
                markers=points_reordered,
                markers_mask=points_mask_reordered,
                vein_polyline=labels,
                vein_polyline_mask=labels_mask,
                vein_classification=vein_classification,
                vein_regression=vein_regression,
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
