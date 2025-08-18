import numpy as np
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import cv2

from difftactile.main.constants import *


class SyntheticImageGenerator:
    def __init__(self):
        pass

    @staticmethod
    def crop(points, top_left, target_resolution):
        # Create a copy and apply the offset to all points at once
        points -= top_left

        w = target_resolution[0]
        h = target_resolution[1]
        
        # Create mask for valid points using vectorized operations
        mask = (
            (points[..., 0] >= 0) & (points[..., 0] < w)
            & (points[..., 1] >= 0) & (points[..., 1] < h)
        )
        
        return points, mask

    @staticmethod
    def alpha_shape(points, alpha=1e-10):
        points = np.unique(points, axis=0)
        if len(points) < 4:
            return np.array([])

        tri = Delaunay(points)
        edges = set()

        for ia, ib, ic in tri.simplices:
            pa, pb, pc = points[ia], points[ib], points[ic]
            a = np.linalg.norm(pa - pb)
            b = np.linalg.norm(pb - pc)
            c = np.linalg.norm(pc - pa)
            s = (a + b + c) / 2.0
            area = max(s * (s - a) * (s - b) * (s - c), 1e-10) ** 0.5
            circum_r = a * b * c / (4.0 * area)

            if circum_r < 1.0 / alpha:
                edges.update([(ia, ib), (ib, ic), (ic, ia)])

        edge_segments = [ (points[i], points[j]) for i, j in edges ]
        m = MultiLineString(edge_segments)
        triangles = list(polygonize(m))
        concave = unary_union(triangles)

        if len(edge_segments) == 0:
            return np.array([])

        if isinstance(concave, Polygon):
            return np.array(concave.exterior.coords)
        else:
            return np.array(concave.geoms[0].exterior.coords)
    
    @staticmethod
    def filter_points(w, h, cx, cy, r, points):
        points_filtered = []
        for point in points:
            x, y = point
            if (0 <= x < w and 0 <= y < h and
                ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2):
                points_filtered.append([x, y])
        points_filtered = np.array(points_filtered, dtype=float)
        return points_filtered
    
    @staticmethod
    def filter_points_vectorised(cx, cy, r, points):
        # Extract x and y coordinates from the last dimension
        x = points[..., 0]
        y = points[..., 1]
        
        # Check circle condition
        in_circle = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
        
        # Combine conditions
        return in_circle

    @staticmethod
    def fit_polynomial(points, n=3, k=SYSTEM_PARAMS.meta.polyline_num):
        """
        Fit a polynomial of order n to 2D points and return evenly sampled points along the curve.
        
        Args:
            n: Order of the polynomial
            points: numpy array of shape (num_points, 2) containing x,y coordinates
            k: Number of points to sample along the curve (default: 100)
            
        Returns:
            numpy array of shape (k, 2) containing evenly sampled points along the polynomial
        """
        if len(points) < 3 * (n + 1):
            return -np.ones(shape=(0, 2))
            
        # Extract x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        # Get x range from input points
        x_min = np.min(x)
        x_max = np.max(x)
        
        # Fit polynomial using numpy's polyfit
        coefficients = np.polyfit(x, y, n)
        
        # Generate evenly spaced x coordinates
        x_sampled = np.linspace(x_min, x_max, k)
        
        # Evaluate polynomial at sampled points
        y_sampled = np.polyval(coefficients, x_sampled)
        
        # Combine x and y coordinates
        sampled_points = np.column_stack((x_sampled, y_sampled))
        
        return sampled_points

    @staticmethod
    def vector_point_to_polynomial(polynomial, target):
        """
        Compute the shortest vector from target point(s) to the closest point on a polynomial curve.
        
        Args:
            polynomial: numpy array of shape (num_points, 2) containing points along the polynomial curve
            target: numpy array of shape (2,) or (N, 2) containing the target point coordinates
            
        Returns:
            numpy array of shape (2,) or (N, 2) representing the vector(s) from target to closest point
        """
        # Handle both single point and batch inputs
        single_input = target.ndim == 1
        if single_input:
            target = target[np.newaxis, :]
        
        # Compute squared distances from each target to all points on polynomial
        # Using broadcasting: (x1-x2)^2 + (y1-y2)^2 for all points at once
        # Shape: (N, num_points)
        distances = np.sum((polynomial[np.newaxis, :, :] - target[:, np.newaxis, :]) ** 2, axis=2)
        
        # Find indices of closest points for each target
        # Shape: (N,)
        closest_idx = np.argmin(distances, axis=1)
        
        # Get closest points for each target
        # Shape: (N, 2)
        closest_points = polynomial[closest_idx]
        
        # Compute vectors from targets to closest points
        vectors = closest_points - target
        
        return vectors[0] if single_input else vectors

    @staticmethod
    def compute_mask(h, w, points):
        res = np.zeros((h, w), dtype=np.uint8)
        if len(points) > 0:
            contour_vein = SyntheticImageGenerator.alpha_shape(points).astype(np.int32)
            if len(contour_vein) > 0:
                contour_vein_cv = contour_vein.reshape((-1, 1, 2))
                cv2.fillPoly(res, [contour_vein_cv], color=255)
        
        return res

    @staticmethod
    def filter_using_mask(mask, points):
        res = []
        for point in points:
            x, y = int(point[0]), int(point[1])
            if mask[y, x] > 127:
                res.append(point)
        res = np.array(res)
        return res
    
    @staticmethod
    def create_padded_array_with_mask(data_list, k=None):
        if not data_list:
            return np.array([]), np.array([])
        n = len(data_list)
        if k is not None:
            num_points_max = k
        else:
            num_points_max = max(arr.shape[0] for arr in data_list)
        padded_array = np.zeros((n, num_points_max, 2), dtype=data_list[0].dtype)
        mask = np.zeros((n, num_points_max), dtype=bool)
        for i, arr in enumerate(data_list):
            num_points = arr.shape[0]
            if num_points > 0:
                padded_array[i, :num_points, :] = arr
                mask[i, :num_points] = True
        return padded_array, mask