import numpy as np
from scipy.spatial import Delaunay

def alpha_shape_3d(points, alpha):
    if len(points) < 4:
        raise ValueError("Need at least 4 points")
    delaunay = Delaunay(points)
    tetrahedra = delaunay.simplices

    A = points[tetrahedra[:,0]]
    B = points[tetrahedra[:,1]]
    C = points[tetrahedra[:,2]]
    D = points[tetrahedra[:,3]]
    AB, AC, AD = B - A, C - A, D - A

    volume = np.einsum('ij,ij->i', np.cross(AB, AC), AD) / 6.0
    volume = np.abs(volume)
    AB2 = np.sum(AB**2, axis=1)
    AC2 = np.sum(AC**2, axis=1)
    AD2 = np.sum(AD**2, axis=1)
    M = np.stack([AB, AC, AD], axis=1)
    rhs = np.stack([AB2, AC2, AD2], axis=1) / 2.0

    try:
        centers = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        centers = np.full_like(rhs, np.inf)
    radii = np.linalg.norm(centers, axis=1)
    mask = radii < alpha
    
    return tetrahedra[mask]
