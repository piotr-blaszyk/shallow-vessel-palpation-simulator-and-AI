import numpy as np

class Common:
    @staticmethod
    def compute_min_spacing_3d(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(len(points), dtype=bool)
        min_dist = np.min(distances[mask])
        return min_dist
