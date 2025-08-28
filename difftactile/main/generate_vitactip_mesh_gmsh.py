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
        
        roco = self.dome_radius_of_curvature(
            r[0], hs[0]
        )
        roci = roco-shell_thickness
        cap_base_z = roco-hs[0]
        bot_z = roco-hs.sum()

        self.r = r
        self.hs = hs
        self.roco = roco
        self.roci = roci
        self.cbz = cap_base_z
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

            radius=self.roco,
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
        # self.get_difftactile_variables()
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

    def get_difftactile_variables(self):
        element_types, tetrahedra_tags, tetrahedra_vertex_tags = (
            gmsh.model.mesh.getElements(dim=3)
        )
        self.all_tetrahedra = tetrahedra_vertex_tags[0].reshape(-1, 4).astype(int)
        tetrahedra_tags = tetrahedra_tags[0]
        element_types_2d, triangle_tags, triangle_nodes = gmsh.model.mesh.getElements(
            dim=2
        )
        self.surface_triangles = triangle_nodes[0].reshape(-1, 3).astype(int)
        self.node_tags, self.node_coordinates, parametric_coord = (
            gmsh.model.mesh.getNodes()
        )
        self.node_coordinates = self.node_coordinates.reshape(-1, 3).astype(float)
        self.node_tags = np.array(self.node_tags)
        node_tag_to_idx = {tag: idx for idx, tag in enumerate(self.node_tags)}
        self.node_labels = np.zeros((len(self.node_tags), 2), dtype=bool)
        physical_groups = gmsh.model.getPhysicalGroups()
        group_to_idx = {"shell": 0, "gel": 1}
        surface_node_tags = []
        surface_coords = []
        for i in range(len(self.node_tags)):
            tag = self.node_tags[i]
            node = self.node_coordinates[i]
            x, y, z = node
            if z > self.cbz:
                if np.linalg.norm(node) > self.roco - self.dist_eps:
                    surface_node_tags.append(tag)
                    surface_coords.append(node)
            else:
                xy = np.array([x, y])
                if np.linalg.norm(xy) > self.r[0] - self.dist_eps:
                    surface_node_tags.append(tag)
                    surface_coords.append(node)
        surface_node_tags = np.array(surface_node_tags)
        surface_coords = np.array(surface_coords).reshape(-1, 3).astype(float)
        surface_triangles_mask = []
        for triangle in self.surface_triangles:
            all_nodes_surface = all(tag in surface_node_tags for tag in triangle)
            surface_triangles_mask.append(all_nodes_surface)
        self.surface_triangles = self.surface_triangles[surface_triangles_mask]
        dome_surface_node_tags = surface_node_tags[
            self.node_coordinates[(surface_node_tags - 1), 2] >= self.cbz
        ]
        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            if name in group_to_idx:
                group_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
                node_tag_to_idx = {tag: idx for idx, tag in enumerate(self.node_tags)}
                for node_tag in group_node_tags:
                    if node_tag in node_tag_to_idx:
                        self.node_labels[
                            node_tag_to_idx[node_tag], group_to_idx[name]
                        ] = True
        marker_node_tags = np.array([0])
        self.all_tetrahedra -= 1
        surface_node_tags -= 1
        dome_surface_node_tags -= 1
        self.surface_triangles -= 1
        self.node_tags -= 1
        min_particle_spacing = self.compute_particle_spacing(mode="min")
        max_particle_spacing = self.compute_particle_spacing(mode="max")
        print(f"min_particle_spacing (mm): {min_particle_spacing}")
        print(f"max_particle_spacing (mm): {max_particle_spacing}")
        mesh_data = {
            "all_tetrahedra": self.all_tetrahedra,
            "node_coordinates": self.node_coordinates,
            "node_labels": self.node_labels,
            "surface_node_tags": surface_node_tags,
            "surface_triangles": self.surface_triangles,
            "node_tags": self.node_tags,
            "group_to_idx": group_to_idx,
            "marker_node_tags": marker_node_tags,
            "dome_surface_node_tags": dome_surface_node_tags,
            "z_bottom": self.bz,
            "radius_of_curvature_inner": self.roci,
            "radius_of_curvature_outer": self.roco,
            "min_particle_spacing": min_particle_spacing,
            "max_particle_spacing": max_particle_spacing,
        }
        print(f"self.node_coordinates.shape[0]: {self.node_coordinates.shape[0]}")
        os.makedirs("output", exist_ok=True)
        with open(SYSTEM_PARAMS.files.gmsh_mesh, "wb") as f:
            pickle.dump(mesh_data, f)

    def compute_particle_spacing(self, mode="min"):
        if mode not in ["min", "max"]:
            raise ValueError("mode must be either 'min' or 'max'")
        tet_vertices = self.node_coordinates[self.all_tetrahedra]
        tet_labels = self.node_labels[self.all_tetrahedra]
        edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        initial_value = -float("inf") if mode == "max" else float("inf")
        spacing_all = initial_value
        spacing_shell = initial_value
        spacing_gel = initial_value
        for i, j in edge_pairs:
            edges = tet_vertices[:, i] - tet_vertices[:, j]
            lengths = np.sqrt(np.sum(edges**2, axis=1))
            labels_i = tet_labels[:, i]
            labels_j = tet_labels[:, j]
            if lengths.size > 0:
                spacing_all = (
                    max(spacing_all, np.max(lengths))
                    if mode == "max"
                    else min(spacing_all, np.min(lengths))
                )
            shell_edges = lengths[labels_i[:, 0] & labels_j[:, 0]]
            if shell_edges.size > 0:
                spacing_shell = (
                    max(spacing_shell, np.max(shell_edges))
                    if mode == "max"
                    else min(spacing_shell, np.min(shell_edges))
                )
            gel_edges = lengths[labels_i[:, 1] & labels_j[:, 1]]
            if gel_edges.size > 0:
                spacing_gel = (
                    max(spacing_gel, np.max(gel_edges))
                    if mode == "max"
                    else min(spacing_gel, np.min(gel_edges))
                )
        return {"all": spacing_all, "shell": spacing_shell, "gel": spacing_gel}


def main():
    mesh_generator = MeshGenerator()
    mesh_generator.generate_vitactip_mesh_1_material_no_tips()
