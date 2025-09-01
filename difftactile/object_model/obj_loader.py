"""
load a obj/stl file and convert it to mpm particles
use millimeter as units
the space is 100 mm x 100 mm x 100 mm
"""

import numpy as np
import trimesh
from mesh_to_sdf import *

from difftactile.main.constants import *


class ObjLoader:
    def __init__(self, data_path, num_particles_cube_1d):
        '''
        Load an obj or stl mesh model and convert it to particles
        '''
        self.data_path = data_path # ending with obj or stl
        self.voxel_resolution = 128
        self.num_particles_1d = num_particles_cube_1d

    def generate_surface_particles(self, num_particles):
        self.raw_mesh = trimesh.load(self.data_path, force='mesh', skip_texture=True)
        self.normalized_mesh = self.cleanup_mesh(self.normalize_mesh(self.raw_mesh))
        self.point_cloud = get_surface_point_cloud(self.normalized_mesh)
        self.particles = self.point_cloud.get_random_surface_points(num_particles)

    def generate_particles(self):
        self.raw_mesh = trimesh.load(self.data_path, force='mesh', skip_texture=True)
        np.savetxt(SYSTEM_PARAMS.files.raw_mesh_vertices, self.raw_mesh.vertices, delimiter=",", fmt='%.2f')
        self.normalized_mesh = self.cleanup_mesh(self.normalize_mesh(self.raw_mesh))
        np.savetxt(SYSTEM_PARAMS.files.normalized_mesh_vertices, self.normalized_mesh.vertices, delimiter=",", fmt='%.2f')
        self.voxelized_mesh = self.normalized_mesh.voxelized(pitch=1.0/self.voxel_resolution).fill()
        cube_particles = self.sample_cube()
        self.particles = cube_particles[self.voxelized_mesh.is_filled(cube_particles)]
        self.particles *= SYSTEM_PARAMS_COMPUTED.object_scale
        # the center of the obj is [0.0, 0.0, 0.0]

    def normalize_mesh(self, mesh):
        '''
        Normalize mesh to [-0.5, 0.5].
        '''

        scale  = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
        center = (mesh.vertices.max(0) + mesh.vertices.min(0))/2.0

        normalized_mesh = mesh.copy()
        normalized_mesh.vertices -= center
        normalized_mesh.vertices /= scale
        return normalized_mesh

    def cleanup_mesh(self, mesh):
        '''
        Retain only mesh's vertices, faces, and normals.
        '''
        return trimesh.Trimesh(
            vertices       = mesh.vertices,
            faces          = mesh.faces,
            vertex_normals = mesh.vertex_normals,
            face_normals   = mesh.face_normals,
        )

    def sample_cube(self):
        '''
        Sample grid-like particles in a 3D cube space [-0.5, 0.5]
        '''
        dx = 1 / self.num_particles_1d
        x = np.linspace(-0.5, 0.5, self.num_particles_1d+1)
        particles = np.stack(np.meshgrid(x, x, x, indexing='ij'), -1).reshape((-1, 3))

        return particles
