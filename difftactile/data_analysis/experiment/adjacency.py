import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

from difftactile.main.constants import *


class Adjacency:
    @staticmethod
    def get_graph_connectivity(points):
        # Load base graph connectivity data
        data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        return Adjacency.get_graph_connectivity_helper(data, points)
        
    @staticmethod
    def get_graph_connectivity_helper(data, points):
        base_points = data['points']
        base_adjacency_matrix = data['adjacency_matrix']

        # Compute cost matrix as squared euclidean distances between all pairs of points
        cost_matrix = cdist(points, base_points, metric='sqeuclidean')
        
        # Apply Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create inverse mapping: for each base point index, which input point maps to it
        inverse_mapping = np.zeros_like(row_ind)
        inverse_mapping[col_ind] = row_ind
        
        # Reorder input points to match base points ordering
        points_reordered = points[inverse_mapping]
        
        return base_points, points_reordered, base_adjacency_matrix

    @staticmethod
    def knn(points):
        _, points, _ = Adjacency.get_graph_connectivity(points)
        n = len(points)
        k = 6
        knn = KNeighborsClassifier(n_neighbors=k+1, metric='euclidean')
        knn.fit(points, np.zeros(n))
        _, neighbors = knn.kneighbors(points)
        nearest_neighbors = neighbors[:, 1:]
        source_nodes = np.repeat(np.arange(n), k)
        target_nodes = nearest_neighbors.flatten()
        adjacency_matrix = np.stack([source_nodes, target_nodes], axis=1)
        
        return None, points, adjacency_matrix
        
