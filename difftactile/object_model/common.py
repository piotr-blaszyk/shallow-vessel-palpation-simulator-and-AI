import numpy as np
from scipy.spatial.transform import Rotation as R

class Common:
    @staticmethod
    def compute_min_spacing_3d(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(len(points), dtype=bool)
        min_dist = np.min(distances[mask])
        return min_dist
    
    @staticmethod
    def compute_transformation_matrix(pose):
        translation = pose[:3]
        quaternion = pose[3:]
        rotation_object = R.from_quat(quaternion)
        rotation_matrix = rotation_object.as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = translation
        return rotation_matrix, transformation_matrix
