import taichi as ti
import numpy as np
import pickle

from difftactile.object_model.obj_loader import ObjLoader
from difftactile.main.constants import *
from difftactile.object_model.common import *

@ti.data_oriented
class RigidObj:
    def __init__(self):
        self.dist_sf = SYSTEM_PARAMS.meta.distance_scaling_factor
        self.load_obj()
        self.particles_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.particles_B.shape[0],),
            needs_grad=False,
        )
    
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

    def set_state_from_outside(self, pose):
        rotation_matrix, transformation_matrix = Common.compute_transformation_matrix(pose)
        self.T_BA = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_BA.from_numpy(transformation_matrix)
        self.initialise_point_cloud()
    
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
