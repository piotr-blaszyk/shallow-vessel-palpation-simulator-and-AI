import gmsh
import numpy as np

np.set_printoptions(formatter={"float_kind": "{:.2f}".format})
import math
import pickle
import os
from difftactile.main.constants import *


class MeshGenerator:
    def __init__(self):
        pass

    def spherical_cap_volume_1(self, radius_of_curvature, cap_height):
        return 1 / 3 * math.pi * cap_height**2 * (3 * radius_of_curvature - cap_height)

    def spherical_cap_volume_2(self, radius_of_base, cap_height):
        return 1 / 6 * math.pi * cap_height * (3 * radius_of_base**2 + cap_height**2)

    def cylinder_volume(self, radius_of_base, height):
        return math.pi * radius_of_base**2 * height

    def dome_radius_of_curvature(self, cap_base_radius, cap_height):
        r = cap_base_radius
        h = cap_height
        return (r**2 + h**2) / (2 * h)

    def load_biomimetic_tip_data(self):
        with open(SYSTEM_PARAMS.files.biomimetic_tip_points, "rb") as f:
            self.biomimetic_tip_points = pickle.load(f)
        self.tips_A = self.biomimetic_tip_points["A_points"]
        self.tips_B = self.biomimetic_tip_points["B_points"]
        distances_A = np.sqrt(self.tips_A[:, 0] ** 2 + self.tips_A[:, 1] ** 2)
        closest_A_idx = np.argmin(distances_A)
        self.tips_A = self.tips_A[closest_A_idx : closest_A_idx + 1]
        self.tips_B = self.tips_B[closest_A_idx : closest_A_idx + 1]

    def initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthFactor",
            SYSTEM_PARAMS.gmsh_mm.characteristic_length_factor,
        )
        gmsh.model.add("ViTacTip")

    def load_system_params(self):
        r = SYSTEM_PARAMS.gmsh_mm.radii
        r = np.array(r, dtype=float)
        hs = SYSTEM_PARAMS.gmsh_mm.heights_top_to_bottom
        hs = np.array(hs, dtype=float)
        shell_thickness = SYSTEM_PARAMS.gmsh_mm.shell_thickness
        refinement_offset = SYSTEM_PARAMS.gmsh_mm.refinement_offset
        dist_eps = SYSTEM_PARAMS.gmsh_mm.dist_eps
        refine_mesh = SYSTEM_PARAMS.gmsh_mm.refine_mesh
        step_file_path = SYSTEM_PARAMS.files.sensor_geometry_step
        
        roc = self.dome_radius_of_curvature(
            r[0], hs[0]
        )
        bot_z = roc-hs.sum()

        self.r = r
        self.hs = hs
        self.roc = roc
        self.bz = bot_z
        self.st = shell_thickness
        self.ro = refinement_offset
        self.dist_eps = dist_eps
        self.should_refine_mesh = refine_mesh == 1
        self.step_file_path = step_file_path

    def generate_vitactip_mesh_1_material_no_tips(self):
        self.initialize_gmsh()
        self.load_system_params()
        outer_ball = gmsh.model.occ.addSphere(
            xc=0, 
            yc=0, 
            zc=0, 

            radius=self.roc,
        )
        cyl_helper = gmsh.model.occ.addCylinder(
            x=0, 
            y=0, 
            z=self.bz, 

            dx=0, 
            dy=0, 
            dz=self.hs[:].sum(), 

            r=self.r[0],
        )
        x = gmsh.model.occ.intersect([(3, outer_ball)], [(3, cyl_helper)])[0]
        cyl_helper = gmsh.model.occ.addCylinder(
            x=0, 
            y=0, 
            z=self.bz, 

            dx=0, 
            dy=0, 
            dz=self.hs[2:].sum(), 

            r=self.r[1],
        )
        x = gmsh.model.occ.fuse(x, [(3, cyl_helper)])[0]

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(SYSTEM_PARAMS.files.gmsh_debug_msh)
        self.get_difftactile_variables()
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        gmsh.finalize()

    def execute_refine_mesh(self):
        return
        z_poi = self.bz + self.rswh - self.ro
        f1 = f"""
        max(
            2.0, 
            min(
                6.0, 
                4.0 - (z-{z_poi}) * 2.0 / 2.0
            )
        )
        """
        gmsh.model.mesh.field.add("MathEval", 1)
        gmsh.model.mesh.field.setString(1, "F", f1)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)
    
    def debug_gmsh(self):
        element_types, tetrahedra_tags, tetrahedra_vertex_tags = (
            gmsh.model.mesh.getElements(dim=3)
        )
        all_tetrahedra = tetrahedra_vertex_tags[0].reshape(-1, 4).astype(int)
        node_tags, node_coordinates, parametric_coord = (
            gmsh.model.mesh.getNodes()
        )
        node_coordinates = node_coordinates.reshape(-1, 3).astype(float)
        all_tetrahedra -= 1
        mesh_data = {
            "all_tetrahedra": all_tetrahedra,
            "node_coordinates": node_coordinates,
        }
        with open(SYSTEM_PARAMS.files.gmsh_debug_pkl, "wb") as f:
            pickle.dump(mesh_data, f)

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
            if z > self.roc-self.hs[0]:
                if np.linalg.norm(node) > self.roc - self.dist_eps:
                    surface_node_tags.append(tag)
                    surface_coords.append(node)
            elif z > self.roc-self.hs[0:2].sum():
                xy = np.array([x, y])
                if np.linalg.norm(xy) > self.r[0] - self.dist_eps:
                    surface_node_tags.append(tag)
                    surface_coords.append(node)
        surface_node_tags = np.array(surface_node_tags)
        surface_coords = np.array(surface_coords).reshape(-1, 3).astype(float)
        surface_triangles_mask = []
        for triangle in surface_triangles:
            all_nodes_surface = all(tag in surface_node_tags for tag in triangle)
            surface_triangles_mask.append(all_nodes_surface)
        surface_triangles = surface_triangles[surface_triangles_mask]

        all_tetrahedra -= 1
        surface_node_tags -= 1
        surface_triangles -= 1
        node_tags -= 1
        dome_surface_node_tags = surface_node_tags[
            node_coordinates[surface_node_tags, 2] >= self.roc-self.hs[0]
        ]
        dome_surface_node_tags -= 1

        self.compute_particle_spacing(
            node_coordinates,
            all_tetrahedra,
        )

        z = node_coordinates[:, 2]
        xy = node_coordinates[:, :2]
        xy_magnitudes = np.linalg.norm(xy, axis=1)
        is_fixed_mask = (
            (z < self.roc-self.hs[:3].sum()) | 
            ((xy_magnitudes > self.r[0]) & (z < self.roc-self.hs[:2].sum()))
        )

        mesh_data = {
            "all_tetrahedra": all_tetrahedra,
            "node_coordinates": node_coordinates,
            "surface_node_tags": surface_node_tags,
            "surface_triangles": surface_triangles,
            "dome_surface_node_tags": dome_surface_node_tags,
            "z_bottom": self.bz,
            "radius_of_curvature_outer": self.roc,
            "mean_particle_spacing": float(self._mean),
            "is_fixed_mask": is_fixed_mask,
        }
        print(f"node_coordinates.shape[0]: {node_coordinates.shape[0]}")
        os.makedirs("output", exist_ok=True)
        with open(SYSTEM_PARAMS.files.gmsh_mesh, "wb") as f:
            pickle.dump(mesh_data, f)

    def compute_particle_spacing(
        self,
        node_coordinates,
        all_tetrahedra,
    ):
        tet_vertices = node_coordinates[all_tetrahedra]
        edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        node_edge_lengths = [[] for _ in range(node_coordinates.shape[0])]
        tetra_edge_lengths = [[] for _ in range(all_tetrahedra.shape[0])]
        all_node_lengths = []
        for i, j in edge_pairs:
            edges = tet_vertices[:, i] - tet_vertices[:, j]
            lengths = np.sqrt(np.sum(edges**2, axis=1))
            if lengths.size > 0:
                all_node_lengths.append(lengths)
                for tet_idx, length in enumerate(lengths):
                    node_i = all_tetrahedra[tet_idx, i]
                    node_j = all_tetrahedra[tet_idx, j]
                    node_edge_lengths[node_i].append(length)
                    node_edge_lengths[node_j].append(length)
                    tetra_edge_lengths[tet_idx].append(length)
        all_node_lengths = np.concatenate(all_node_lengths)
        
        pkl = {
            'node_edge_lengths': node_edge_lengths,
            'tetra_edge_lengths': tetra_edge_lengths,
        }
        path = SYSTEM_PARAMS.files.edge_lengths_pkl
        with open(path, "wb") as f:
            pickle.dump(pkl, f)
        
        self._min = all_node_lengths.min()
        self._max = all_node_lengths.max()
        self._mean = all_node_lengths.mean()

        print(f"min_particle_spacing (mm): {self._min}")
        print(f"max_particle_spacing (mm): {self._max}")
        print(f"mean_particle_spacing (mm): {self._mean}")


def main():
    mesh_generator = MeshGenerator()
    mesh_generator.generate_vitactip_mesh_1_material_no_tips()
