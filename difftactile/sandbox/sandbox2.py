import numpy as np
import cv2
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import polygonize, unary_union

def alpha_shape(points, alpha=1.0):
    """
    Returns the exterior coordinates of the alpha shape (concave hull).
    """
    if len(points) < 4:
        return points

    tri = Delaunay(points)
    edges = set()

    # Loop through triangles
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

    if isinstance(concave, Polygon):
        return np.array(concave.exterior.coords)
    else:
        return np.array(concave.geoms[0].exterior.coords)

# Example points
points = np.random.randint(0, 400, (100, 2))
contour = alpha_shape(points, alpha=0.02).astype(np.int32)  # Lower alpha = tighter hull

# Create blank image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Reshape contour for OpenCV (needs shape Nx1x2)
contour_cv = contour.reshape((-1, 1, 2))

# Draw filled polygon
cv2.fillPoly(image, [contour_cv], color=(0, 255, 0))

# Optional: draw original points
for pt in points:
    cv2.circle(image, pt, 2, (0, 0, 255), -1)

# Show
cv2.imshow("Alpha Shape", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
