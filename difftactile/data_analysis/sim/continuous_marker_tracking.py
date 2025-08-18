from difftactile.main.constants import *
if SYSTEM_PARAMS.meta.cnn_gnn == 0:
    import cv2

import os
import torch
import numpy as np
import random
import math
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import time
from tqdm import tqdm

from difftactile.main.synthetic_image_generator import *
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.data_analysis.experiment.adjacency import *
import re

def main():
    input_dir = SYSTEM_PARAMS.files.dataset_root
    output_dir = SYSTEM_PARAMS.files.dataset_root_test
    base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
    base_points = base_graph_data['points']
    file_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
    for file_path in tqdm(file_paths, desc="Processing marker data files"):
        data = np.load(file_path)
        points = data['markers']
        points_mask = data['markers_mask']
        labels = data['vein_polyline']
        labels_mask = data['vein_polyline_mask']

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
            vein_polyline_mask=labels_mask
        )


if __name__ == '__main__':
    main()
