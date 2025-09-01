import numpy as np


class Common:
    @staticmethod
    def compute_particle_spacing_helper(
        node_coordinates,
        all_tetrahedra,
    ):
        tet_vertices = node_coordinates[all_tetrahedra]
        edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        node_edge_lengths = [[] for _ in range(node_coordinates.shape[0])]
        tetra_edge_lengths = [[] for _ in range(all_tetrahedra.shape[0])]
        all_node_lengths = []
        for i, j in edge_pairs:
            edges = tet_vertices[:, i] - tet_vertices[:, j]
            lengths = np.sqrt(np.sum(edges**2, axis=1))
            if lengths.size > 0:
                all_node_lengths.append(lengths)
                for tet_idx, length in enumerate(lengths):
                    node_i = all_tetrahedra[tet_idx, i]
                    node_j = all_tetrahedra[tet_idx, j]
                    node_edge_lengths[node_i].append(length)
                    node_edge_lengths[node_j].append(length)
                    tetra_edge_lengths[tet_idx].append(length)
        all_node_lengths = np.concatenate(all_node_lengths)
        return (
            all_node_lengths,
            node_edge_lengths,
            tetra_edge_lengths,
        )
