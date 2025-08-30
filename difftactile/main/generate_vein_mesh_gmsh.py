import gmsh
import numpy as np

np.set_printoptions(formatter={"float_kind": "{:.2f}".format})
import math
import pickle
import os
from difftactile.main.constants import *
from difftactile.main.common import *


class MeshGenerator:
    def __init__(self):
        pass

    def cylinder_volume(self, radius_of_base, height):
        return math.pi * radius_of_base**2 * height

    def initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthFactor",
            SYSTEM_PARAMS.gmsh_mm.vein.characteristic_length_factor,
        )
        gmsh.model.add("vein")

    def load_system_params(self):
        self.r = SYSTEM_PARAMS.gmsh_mm.vein.radius
        self.h = SYSTEM_PARAMS.gmsh_mm.vein.length
        self.dist_eps = SYSTEM_PARAMS.gmsh_mm.dist_eps

    def generate_mesh(self):
        self.initialize_gmsh()
        self.load_system_params()
        vein = gmsh.model.occ.addCylinder(
            x=0, 
            y=0, 
            z=0, 

            dx=0, 
            dy=0, 
            dz=self.h, 

            r=self.r,
        )

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        self.get_difftactile_variables()
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        gmsh.finalize()
    
    def get_difftactile_variables(self):
        element_types, tetrahedra_tags, tetrahedra_vertex_tags = (
            gmsh.model.mesh.getElements(dim=3)
        )
        all_tetrahedra = tetrahedra_vertex_tags[0].reshape(-1, 4).astype(int)
        tetrahedra_tags = tetrahedra_tags[0]
        element_types_2d, triangle_tags, triangle_nodes = gmsh.model.mesh.getElements(
            dim=2
        )
        surface_triangles = triangle_nodes[0].reshape(-1, 3).astype(int)
        node_tags, node_coordinates, parametric_coord = (
            gmsh.model.mesh.getNodes()
        )
        node_coordinates = node_coordinates.reshape(-1, 3).astype(float)
        node_tags = np.array(node_tags)

        surface_node_tags = []
        surface_coords = []
        for i in range(len(node_tags)):
            tag = node_tags[i]
            node = node_coordinates[i]
            x, y, z = node
            xy = np.array([x, y])
            if (
                np.linalg.norm(xy) > self.r - self.dist_eps
                # or z < self.dist_eps
                # or z > self.h - self.dist_eps
            ):
                surface_node_tags.append(tag)
                surface_coords.append(node)
        surface_node_tags = np.array(surface_node_tags)
        surface_coords = np.array(surface_coords).reshape(-1, 3).astype(float)

        surface_triangles_mask = []
        for triangle in surface_triangles:
            all_nodes_surface = all(tag in surface_node_tags for tag in triangle)
            surface_triangles_mask.append(all_nodes_surface)
        surface_triangles_mask = np.array(surface_triangles_mask, dtype=bool)
        surface_triangles = surface_triangles[surface_triangles_mask]
        
        all_tetrahedra -= 1
        surface_node_tags -= 1
        surface_triangles -= 1
        node_tags -= 1

        mesh_data = {
            "all_tetrahedra": all_tetrahedra,
            "node_coordinates": node_coordinates,
            "surface_triangles": surface_triangles,
        }
        self.print_particle_spacing(
            node_coordinates,
            all_tetrahedra,
        )
        print(f"node_coordinates.shape[0]: {node_coordinates.shape[0]}")
        os.makedirs("output", exist_ok=True)
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vein_pkl, "wb") as f:
            pickle.dump(mesh_data, f)
        
    def print_particle_spacing(
        self,
        node_coordinates,
        all_tetrahedra,
    ):
        (
            all_node_lengths,
            node_edge_lengths,
            tetra_edge_lengths,
        ) = Common.compute_particle_spacing_helper(
            node_coordinates,
            all_tetrahedra,
        )

        self._min = all_node_lengths.min()
        self._max = all_node_lengths.max()
        self._mean = all_node_lengths.mean()

        print(f"min_particle_spacing (mm): {self._min}")
        print(f"max_particle_spacing (mm): {self._max}")
        print(f"mean_particle_spacing (mm): {self._mean}")


def main():
    mesh_generator = MeshGenerator()
    mesh_generator.generate_mesh()
