import taichi as ti
import numpy as np
from difftactile.object_model.obj_loader import ObjLoader
from difftactile.main.constants import *
from difftactile.object_model.common import *

@ti.data_oriented
class RigidObj:
    def __init__(self):
        self.load_obj()
        self.set_up_physical_state()
    
    def load_obj(self):
        obj_loader = ObjLoader(
            SYSTEM_PARAMS.files.rigid_static_stl,
            num_particles_cube_1d=SYSTEM_PARAMS_COMPUTED.rigid_static.num_particles_cube_1d,
        )
        obj_loader.generate_particles()
        self.num_particles = len(obj_loader.particles)
        self.particles_B = ti.Vector.field(
            3, dtype=float, shape=(self.num_particles,), needs_grad=False
        )
        self.particles_B.from_numpy((obj_loader.particles).astype(np.float32))
    
    def set_up_physical_state(self):
        self.particles_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.num_particles,),
            needs_grad=False,
        )

    def set_pose_from_outside(self, pose):
        rotation_matrix, transformation_matrix = Common.compute_transformation_matrix(pose)
        self.T_BA = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_BA.from_numpy(transformation_matrix)
        self.initialise_point_cloud()
    
    @ti.kernel
    def initialise_point_cloud(self):
        for i in range(self.num_particles):
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
