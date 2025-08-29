import taichi as ti
import numpy as np
import pickle

from difftactile.object_model.obj_loader import ObjLoader
from difftactile.main.constants import *
from difftactile.object_model.common import *

@ti.data_oriented
class Vein:
    def __init__(self):
        self.pose = np.array(SYSTEM_PARAMS_COMPUTED.vein_pose, dtype=float)
        self.dist_sf = SYSTEM_PARAMS.meta.distance_scaling_factor
        r = SYSTEM_PARAMS.gmsh_mm.vein.radius
        r *= 1/1_000*self.dist_sf
        self.r = r
        h = SYSTEM_PARAMS.gmsh_mm.vein.length
        h *= 1/1_000*self.dist_sf
        self.h = h
        self.load_obj()
        self.particles_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.particles_B.shape[0],),
            needs_grad=False,
        )
        self.transform_BA()
    
    def load_obj(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vein_pkl, "rb") as f:
            data = pickle.load(f)
        tetrahedra = data['all_tetrahedra']
        points = data['node_coordinates']
        triangles = data['surface_triangles']

        points *= 1/1_000*self.dist_sf

        self.particles_B = ti.Vector.field(
            3, dtype=float, shape=(points.shape[0],), needs_grad=False
        )
        self.particles_B.from_numpy(points)
        self.triangles = ti.Vector.field(
            3, 
            dtype=int, 
            shape=(triangles.shape[0],), 
            needs_grad=False,
        )
        self.triangles.from_numpy(triangles)

    def transform_BA(self):
        rotation_matrix, transformation_matrix = Common.compute_transformation_matrix(self.pose)
        self.T_BA_np = transformation_matrix.copy()
        self.T_BA = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_BA.from_numpy(transformation_matrix)
        self.initialise_point_cloud()
        self.compute_yz_centre()
    
    @ti.kernel
    def initialise_point_cloud(self):
        for i in range(self.particles_A.shape[0]):
            particle_B = self.particles_B[i]
            particle_A = self.T_BA[None] @ ti.Vector(
                [
                    particle_B[0],
                    particle_B[1],
                    particle_B[2],
                    1.0,
                ]
            )
            self.particles_A[i] = ti.Vector(
                [
                    particle_A[0],
                    particle_A[1],
                    particle_A[2],
                ]
            )
    
    def compute_yz_centre(self):
        centre_B = np.array([0, 0, 0], dtype=float)
        centre_B_h = np.append(centre_B, 1.0)
        centre_A_h = self.T_BA_np @ centre_B_h
        self.centre_A = centre_A_h[:3]
    
    def get_vein_mask(self, points):
        yz_distances = np.sqrt(
            (points[:, 1] - self.centre_A[1])**2 + 
            (points[:, 2] - self.centre_A[2])**2
        )
        radius_mask = yz_distances <= self.r
        length_mask = (points[:, 0] >= self.centre_A[0]) & (points[:, 0] <= self.centre_A[0] + self.h)
        return radius_mask & length_mask
