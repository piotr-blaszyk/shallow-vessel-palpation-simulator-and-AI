import gmsh
import numpy as np
np.set_printoptions(formatter={'float_kind': '{:.2f}'.format})
import math
import sys
import pickle
import os
import itertools
import json

class MeshGenerator:
    def __init__(self):
        self.params = None
        self.mesh_params = None
        self.vitactip_params = None
        self.geometry_data = None
        self.biomimetic_tip_points = None
        self.A_points = None
        self.B_points = None
        self.node_tags = None
        self.node_coordinates = None
        self.all_tetrahedra = None
        self.surface_triangles = None
        
    def spherical_cap_volume_1(self, radius_of_curvature, cap_height):
        return 1/3 * math.pi * cap_height ** 2 * (3 * radius_of_curvature - cap_height)

    def spherical_cap_volume_2(self, radius_of_base, cap_height):
        return 1/6 * math.pi * cap_height * (3 * radius_of_base ** 2 + cap_height ** 2)

    def cylinder_volume(self, radius_of_base, height):
        return math.pi * radius_of_base ** 2 * height

    def dome_radius_of_curvature(self, radius_of_projection, height):
        r = radius_of_projection
        h = height
        return (r ** 2 + h ** 2) / (2 * h)

    def calculate_volumes_SI(self):
        stem_wall_radius_outer = self.geometry_data['stem_wall_radius_outer']
        stem_wall_radius_inner = self.geometry_data['stem_wall_radius_inner']
        radius_of_curvature_outer = self.geometry_data['radius_of_curvature_outer']
        radius_of_curvature_inner = self.geometry_data['radius_of_curvature_inner']
        stem_height = self.geometry_data['stem_height']
        cap_height = self.geometry_data['cap_height']
        
        outer_cylinder = self.cylinder_volume(stem_wall_radius_outer, stem_height)
        outer_cap = self.spherical_cap_volume_2(stem_wall_radius_outer, cap_height)
        outer_solid_volume = outer_cylinder + outer_cap

        inner_cylinder = self.cylinder_volume(stem_wall_radius_inner, stem_height)
        inner_cap = self.spherical_cap_volume_2(stem_wall_radius_inner, cap_height)
        inner_solid_volume = inner_cylinder + inner_cap

        shell_volume = outer_solid_volume - inner_solid_volume
        gel_volume = inner_solid_volume

        shell_volume /= 1e3 ** 3
        gel_volume /= 1e3 ** 3

        volumes = {
            'shell': shell_volume,
            'gel': gel_volume,
            'all': outer_solid_volume,
        }

        return volumes

    def load_parameters(self):
        # Load system parameters from JSON
        with open('../tasks/system-params.json', 'r') as f:
            self.params = json.load(f)
        self.mesh_params = self.params['gmsh_mm']
        self.vitactip_params = self.params['vitactip']

        self.number_of_materials = self.vitactip_params['number_of_materials']
        self.refine_mesh = self.mesh_params['refine_mesh'] == 1

        with open('biomimetic-tip-points.pkl', 'rb') as f:
            self.biomimetic_tip_points = pickle.load(f)
        self.A_points = self.biomimetic_tip_points['A_points']
        self.B_points = self.biomimetic_tip_points['B_points']
        self.A_points = self.A_points[:, [0, 2, 1]]
        self.B_points = self.B_points[:, [0, 2, 1]]
        
        # Find the point closest to (0,0) in the xz plane for A_points
        distances_A = np.sqrt(self.A_points[:, 0]**2 + self.A_points[:, 2]**2)  # xz plane distance
        closest_A_idx = np.argmin(distances_A)
        self.A_points = self.A_points[closest_A_idx:closest_A_idx+1]  # Keep only the closest point
        self.B_points = self.B_points[closest_A_idx:closest_A_idx+1]  # Keep only the closest point

    def initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", self.mesh_params['characteristic_length_factor'])
        gmsh.model.add("ViTacTip")

    def setup_geometry(self):
        stem_wall_radius_outer = self.mesh_params['stem_wall_radius_outer']
        stem_wall_radius_inner = self.mesh_params['stem_wall_radius_inner']
        cap_height = self.mesh_params['cap_height']
        stem_height = self.mesh_params['stem_height']

        radius_of_curvature_outer = self.dome_radius_of_curvature(stem_wall_radius_outer, cap_height)
        radius_of_curvature_inner = radius_of_curvature_outer - 1
        y_cap_base = radius_of_curvature_outer - cap_height
        y_bottom = y_cap_base - stem_height
        
        self.geometry_data = {
            'stem_wall_radius_outer': stem_wall_radius_outer,
            'stem_wall_radius_inner': stem_wall_radius_inner,
            'cap_height': cap_height,
            'stem_height': stem_height,
            'radius_of_curvature_outer': radius_of_curvature_outer,
            'radius_of_curvature_inner': radius_of_curvature_inner,
            'y_cap_base': y_cap_base,
            'y_bottom': y_bottom
        }

    def generate_vitactip_mesh(self):
        self.load_parameters()
        self.initialize_gmsh()
        self.setup_geometry()

        self.number_of_biomimetic_tips = self.vitactip_params['number_of_biomimetic_tips']
        radius_of_curvature_outer = self.geometry_data['radius_of_curvature_outer']
        radius_of_curvature_inner = self.geometry_data['radius_of_curvature_inner']
        y_cap_base = self.geometry_data['y_cap_base']
        y_bottom = self.geometry_data['y_bottom']
        stem_wall_radius_outer = self.geometry_data['stem_wall_radius_outer']
        stem_height = self.geometry_data['stem_height']

        if self.number_of_materials == 1:
            outer_ball = gmsh.model.occ.addSphere(0, 0, 0, radius_of_curvature_outer)
            cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
            all_volume = gmsh.model.occ.intersect([(3, outer_ball)], [(3, cyl_helper)])[0]
        elif self.number_of_materials == 2:
            outer_ball = gmsh.model.occ.addSphere(0, 0, 0, radius_of_curvature_outer)
            inner_ball = gmsh.model.occ.addSphere(0, 0, 0, radius_of_curvature_inner)
            cap = gmsh.model.occ.cut([(3, outer_ball)], [(3, inner_ball)])[0]
            cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
            cap = gmsh.model.occ.intersect(cap, [(3, cyl_helper)])[0]

            if self.number_of_biomimetic_tips == 0:
                pass
            elif self.number_of_biomimetic_tips == 1:
                A = self.A_points[0]
                B = self.B_points[0]
                AB = B-A
                tip_cylinder = gmsh.model.occ.addCylinder(0, A[1], 0, AB[0], AB[1], AB[2], 0.25)
                gmsh.model.occ.synchronize()
                cap = gmsh.model.occ.fragment(cap, [(3, tip_cylinder)])[0]
            else:
                raise Exception("Having more than 1 biomimetic tip is not supported yet")

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
            
            new_fragments = []
            for dim, tag in cap:
                node_tags, node_coordinates, parametric_coord = gmsh.model.mesh.getNodes(dim=dim, tag=tag, includeBoundary=True)
                node_coordinates = node_coordinates.reshape(-1, 3)
                assert node_coordinates.shape[0] > 0
                magnitudes = np.linalg.norm(node_coordinates, axis=1)
                max_magnitude_index = np.argmax(magnitudes)
                point_with_largest_magnitude = node_coordinates[max_magnitude_index]
                largest_magnitude = magnitudes[max_magnitude_index]
                magnitudes.sort()
                if largest_magnitude>radius_of_curvature_outer+1:
                    gmsh.model.occ.remove(dimTags=[(dim, tag)], recursive=True)
                    continue
                new_fragments.append((dim, tag))
            
            stem_wall_outer = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_outer)
            stem_wall_inner = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, self.geometry_data['stem_wall_radius_inner'])
            stem_wall = gmsh.model.occ.cut([(3, stem_wall_outer)], [(3, stem_wall_inner)])[0]
            shell = gmsh.model.occ.fuse(new_fragments, stem_wall)[0]

            outer_ball = gmsh.model.occ.addSphere(0, 0, 0, radius_of_curvature_outer)
            cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
            all_volume = gmsh.model.occ.intersect([(3, outer_ball)], [(3, cyl_helper)])[0]
            gel = gmsh.model.occ.cut(all_volume, shell, removeTool=False)[0]
            gmsh.model.occ.synchronize()
        else:
            raise Exception("number of materials must be 1 or 2")

        if self.refine_mesh:
            r_max = radius_of_curvature_outer
            r_min = radius_of_curvature_outer - 4
            f1 = f"6.0+max(0.0,min(4.0,{r_max}-sqrt(x^2+y^2+z^2)))"
            print(f1)

            gmsh.model.mesh.field.add("MathEval", 1)
            gmsh.model.mesh.field.setString(1, "F", f1)

            # gmsh.model.mesh.field.add("MathEval", 2)
            # gmsh.model.mesh.field.setString(2, "F", f"6.0")

            # gmsh.model.mesh.field.add("Min", 3)
            # gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

            gmsh.model.mesh.field.setAsBackgroundMesh(1)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        if self.number_of_materials == 2:
            gmsh.model.addPhysicalGroup(3, [x[1] for x in shell], name="shell")
            gmsh.model.addPhysicalGroup(3, [x[1] for x in gel], name="gel")
        
        volumes = self.calculate_volumes_SI()
        if self.number_of_materials == 1:
            print(f"Volume: {volumes['all']:0.3e} m³")
        else:
            print(f"Gel Volume: {volumes['gel']:0.3e} m³")
            print(f"Shell Volume: {volumes['shell']:0.3e} m³")
        self.get_difftactile_variables()
        gmsh.fltk.run()
        gmsh.finalize()

    def get_difftactile_variables(self):
        # Unpack all geometry variables
        stem_wall_radius_outer = self.geometry_data['stem_wall_radius_outer']
        stem_wall_radius_inner = self.geometry_data['stem_wall_radius_inner']
        cap_height = self.geometry_data['cap_height']
        stem_height = self.geometry_data['stem_height']
        radius_of_curvature_outer = self.geometry_data['radius_of_curvature_outer']
        radius_of_curvature_inner = self.geometry_data['radius_of_curvature_inner']
        y_cap_base = self.geometry_data['y_cap_base']
        y_bottom = self.geometry_data['y_bottom']

        # element_types is irrelevant
        # triangle_tags[0] is a 1d int numpy array of shape (num_triangles,)
        # triangle_vertex_tags[0] is a 1d int numpy array of shape (num_traingles * 3,)
        element_types, tetrahedra_tags, tetrahedra_vertex_tags = gmsh.model.mesh.getElements(dim=3)
        self.all_tetrahedra = tetrahedra_vertex_tags[0].reshape(-1, 4).astype(int)
        tetrahedra_tags = tetrahedra_tags[0]
        element_types_2d, triangle_tags, triangle_nodes = gmsh.model.mesh.getElements(dim=2)
        self.surface_triangles = triangle_nodes[0].reshape(-1, 3).astype(int)
        # node_tags is a 1d int python array of length num_particles
        # node_coordinates is a 1d float numpy array of shape (num_particles*3,)
        # parametric_coord is irrelevant
        self.node_tags, self.node_coordinates, parametric_coord = gmsh.model.mesh.getNodes()
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
            x,y,z = node
            if y > y_cap_base:
                if np.linalg.norm(node) > radius_of_curvature_outer - 0.1:
                    surface_node_tags.append(tag)
                    surface_coords.append(node)
            else:
                xz = np.array([x, z])
                if np.linalg.norm(xz) > stem_wall_radius_outer - 0.1:
                    surface_node_tags.append(tag)
                    surface_coords.append(node)
        surface_node_tags = np.array(surface_node_tags)
        surface_coords = np.array(surface_coords).reshape(-1, 3).astype(float)
        
        # Compute surface triangles - triangles where all nodes belong to surface_node_tags
        surface_triangles_mask = []
        for triangle in self.surface_triangles:
            all_nodes_surface = all(tag in surface_node_tags for tag in triangle)
            surface_triangles_mask.append(all_nodes_surface)
        self.surface_triangles = self.surface_triangles[surface_triangles_mask]

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            if name in group_to_idx:
                group_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
                node_tag_to_idx = {tag: idx for idx, tag in enumerate(self.node_tags)}
                for node_tag in group_node_tags:
                    if node_tag in node_tag_to_idx:
                        self.node_labels[node_tag_to_idx[node_tag], group_to_idx[name]] = True

        if self.number_of_biomimetic_tips not in [0, 1]:
            raise Exception("self.number_of_biomimetic_tips must be 0 or 1")

        min_dist = float('inf')
        A = self.A_points[0]
        min_node_tag = None
        for x in np.unique(self.all_tetrahedra.flatten()):
            dist = np.linalg.norm(self.node_coordinates[x-1]-A)
            if dist < min_dist:
                min_dist = dist
                min_node_tag = x
        assert min_node_tag is not None, "no marker node found"
        marker_node_tags = np.array([min_node_tag])-1

        self.node_coordinates = self.node_coordinates[1:]
        self.node_labels = self.node_labels[1:]
        self.all_tetrahedra -= 1
        self.surface_triangles -= 1
        marker_node_tags -= 1
        all_tetrahedra_temp = []
        for tetra in self.all_tetrahedra:
            if -1 not in tetra:
                all_tetrahedra_temp.append(tetra)
        self.all_tetrahedra = np.array(all_tetrahedra_temp)
        surface_triangles_temp = []
        for tetra in self.surface_triangles:
            if -1 not in tetra:
                surface_triangles_temp.append(tetra)
        self.surface_triangles = np.array(surface_triangles_temp)

        self.all_tetrahedra -= 1
        surface_node_tags -= 1
        self.surface_triangles -= 1
        self.node_tags -= 1

        min_particle_spacing = self.compute_particle_spacing(mode='min')
        max_particle_spacing = self.compute_particle_spacing(mode='max')
        print(f'min_particle_spacing (mm): {min_particle_spacing}')
        print(f'max_particle_spacing (mm): {max_particle_spacing}')

        mesh_data = {
            'all_tetrahedra': self.all_tetrahedra,
            'node_coordinates': self.node_coordinates,
            'node_labels': self.node_labels,
            'surface_node_tags': surface_node_tags,
            'surface_triangles': self.surface_triangles,
            'node_tags': self.node_tags,
            'group_to_idx': group_to_idx,
            'marker_node_tags': marker_node_tags,

            'y_bottom': y_bottom,
            'radius_of_curvature_inner': radius_of_curvature_inner,
            'radius_of_curvature_outer': radius_of_curvature_outer,
            'min_particle_spacing': min_particle_spacing,
            'max_particle_spacing': max_particle_spacing,
        }
        print(f'number of vertices generated: {self.node_coordinates.shape[0]}')
        os.makedirs('output', exist_ok=True)
        with open('output/gmsh-mesh.pkl', 'wb') as f:
            pickle.dump(mesh_data, f)
    
    def compute_particle_spacing(self, mode='min'):
        """
        Compute either minimum or maximum distance between vertices within tetrahedra,
        separately for shell nodes, gel nodes, and all nodes.
        
        Args:
            mode: String, either 'min' or 'max' to compute minimum or maximum spacing
        
        Returns:
            dict: Edge lengths for 'all', 'shell', and 'gel' groups
        """
        if mode not in ['min', 'max']:
            raise ValueError("mode must be either 'min' or 'max'")
            
        # Get coordinates of all vertices for each tetrahedron
        tet_vertices = self.node_coordinates[self.all_tetrahedra]  # Shape: (num_tets, 4, 3)
        tet_labels = self.node_labels[self.all_tetrahedra]  # Shape: (num_tets, 4, 2)
        
        # Edge pairs to check in each tetrahedron
        edge_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        # Initialize spacings with appropriate extreme values
        initial_value = -float('inf') if mode == 'max' else float('inf')
        spacing_all = initial_value
        spacing_shell = initial_value
        spacing_gel = initial_value
        
        for i, j in edge_pairs:
            # Calculate distances between vertex pairs for all tetrahedra at once
            edges = tet_vertices[:, i] - tet_vertices[:, j]  # Shape: (num_tets, 3)
            lengths = np.sqrt(np.sum(edges**2, axis=1))  # Shape: (num_tets,)
            
            # Get labels for both vertices of each edge
            labels_i = tet_labels[:, i]  # Shape: (num_tets, 2)
            labels_j = tet_labels[:, j]  # Shape: (num_tets, 2)
            
            # Update spacing for all edges
            if lengths.size > 0:
                spacing_all = max(spacing_all, np.max(lengths)) if mode == 'max' else min(spacing_all, np.min(lengths))
            
            # Update spacing for shell edges (both vertices must be shell)
            shell_edges = lengths[labels_i[:, 0] & labels_j[:, 0]]
            if shell_edges.size > 0:
                spacing_shell = max(spacing_shell, np.max(shell_edges)) if mode == 'max' else min(spacing_shell, np.min(shell_edges))
            
            # Update spacing for gel edges (both vertices must be gel)
            gel_edges = lengths[labels_i[:, 1] & labels_j[:, 1]]
            if gel_edges.size > 0:
                spacing_gel = max(spacing_gel, np.max(gel_edges)) if mode == 'max' else min(spacing_gel, np.min(gel_edges))
        
        return {
            'all': spacing_all,
            'shell': spacing_shell,
            'gel': spacing_gel
        }

if __name__ == "__main__":
    mesh_generator = MeshGenerator()
    mesh_generator.generate_vitactip_mesh()
