"""
a class to describe sensor elastomer with FEM
"""

import taichi as ti
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pickle
import sys

from difftactile.sensor_model.fisheye_model import * 
from difftactile.main.constants import *

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class ViTacTip:
    def __init__(self):
        self.fisheye_model = FisheyeModel()
        self.set_up_system_params()
        self.load_mesh()
        self.initialise_camera_model()
        self.set_up_physical_state()
    
    def set_up_system_params(self):
        # Material parameters for all materials
        self.mass_density = ti.field(dtype=ti.f32, shape=(SYSTEM_PARAMS.vitactip.number_of_materials,), needs_grad=False)
        self.youngs_modulus = ti.field(dtype=ti.f32, shape=(SYSTEM_PARAMS.vitactip.number_of_materials,), needs_grad=True)
        self.poissons_ratio = ti.field(dtype=ti.f32, shape=(SYSTEM_PARAMS.vitactip.number_of_materials,), needs_grad=False)
        self.mu = ti.field(dtype=ti.f32, shape=(SYSTEM_PARAMS.vitactip.number_of_materials,), needs_grad=True)
        self.lam = ti.field(dtype=ti.f32, shape=(SYSTEM_PARAMS.vitactip.number_of_materials,), needs_grad=True)

        if SYSTEM_PARAMS.vitactip.number_of_materials == 1:
            self.mass_density[0] = SYSTEM_PARAMS.vitactip.single_material.density
            self.youngs_modulus[0] = SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
            self.poissons_ratio[0] = SYSTEM_PARAMS.vitactip.single_material.poissons_ratio

            self.mass_density[1] = SYSTEM_PARAMS.vitactip.single_material.density
            self.youngs_modulus[1] = SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
            self.poissons_ratio[1] = SYSTEM_PARAMS.vitactip.single_material.poissons_ratio
        else:
            # Initialize material parameters
            # Shell (Vytaflex 60) - material index 0
            self.mass_density[0] = SYSTEM_PARAMS.vitactip.shell.density
            self.youngs_modulus[0] = SYSTEM_PARAMS.vitactip.shell.youngs_modulus
            self.poissons_ratio[0] = SYSTEM_PARAMS.vitactip.shell.poissons_ratio
            
            # Gel (RTV27905) - material index 1
            self.mass_density[1] = SYSTEM_PARAMS.vitactip.gel.density
            self.youngs_modulus[1] = SYSTEM_PARAMS.vitactip.gel.youngs_modulus
            self.poissons_ratio[1] = SYSTEM_PARAMS.vitactip.gel.poissons_ratio
        
        # Compute Lamé parameters for all materials
        for i in range(SYSTEM_PARAMS.vitactip.maximum_number_of_materials):
            self.mu[i] = self.youngs_modulus[i] / 2 / (1 + self.poissons_ratio[i])
            self.lam[i] = (self.youngs_modulus[i] * self.poissons_ratio[i] / 
                          ((1 + self.poissons_ratio[i]) * (1 - 2 * self.poissons_ratio[i])))
            
        # Add Rayleigh damping coefficients
        self.rayleigh_damping_alpha = ti.field(dtype=ti.f32, shape=(), needs_grad=False)  # mass damping coefficient
        self.rayleigh_damping_beta = ti.field(dtype=ti.f32, shape=(), needs_grad=False)   # stiffness damping coefficient
        
        # Set default values (these should be tuned based on your specific needs)
        self.rayleigh_damping_alpha[None] = SYSTEM_PARAMS.vitactip.rayleigh_damping_alpha  # typical values range from 0 to 0.1
        self.rayleigh_damping_beta[None] = SYSTEM_PARAMS.vitactip.rayleigh_damping_beta  # typical values range from 0.001 to 0.01

        # Add hourglass control parameters
        self.hourglass_enabled = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.hourglass_coefficient = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.hourglass_modulus_scale = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        
        # Set hourglass control parameters from config
        self.hourglass_enabled[None] = SYSTEM_PARAMS.vitactip.hourglass_control.enabled
        self.hourglass_coefficient[None] = SYSTEM_PARAMS.vitactip.hourglass_control.coefficient
        self.hourglass_modulus_scale[None] = SYSTEM_PARAMS.vitactip.hourglass_control.modulus_scale

    def load_mesh(self):
        # Load mesh data from gmsh
        with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
            mesh_data = pickle.load(f)
        
        # Unpack mesh data
        self.dome_surface_node_tags_npy = mesh_data['dome_surface_node_tags']
        self.dome_surface_node_tags = ti.field(dtype=int, shape=(self.dome_surface_node_tags_npy.shape[0],), needs_grad=False)
        self.dome_surface_node_tags.from_numpy(self.dome_surface_node_tags_npy)
        self.surface_node_tags_npy = mesh_data['surface_node_tags']
        all_tetrahedra = mesh_data['all_tetrahedra']
        node_coordinates = mesh_data['node_coordinates'] / 1_000
        node_labels = mesh_data['node_labels']
        surface_triangles = mesh_data['surface_triangles']
        group_to_idx = mesh_data['group_to_idx']
        z_bottom = mesh_data['z_bottom'] / 1_000
        
        # Compute fixed layer nodes (nodes at the bottom)
        is_fixed_layer = np.abs(node_coordinates[:, 2] - z_bottom) < SYSTEM_PARAMS.vitactip.fixed_layer_distance_from_bottom  # Check if y-coordinate is at bottom
        
        # Append is_fixed_layer to node_labels
        node_labels = np.column_stack([node_labels, is_fixed_layer])
        
        # Compute element materials and masses
        element_materials = np.full(len(all_tetrahedra), fill_value=-1, dtype=np.int32)
        vertex_masses = np.zeros(len(node_coordinates), dtype=np.float32)
        
        # Compute element volumes and assign materials
        for i, tetra in enumerate(all_tetrahedra):
            v1, v2, v3, v4 = tetra
            pos1, pos2, pos3, pos4 = node_coordinates[v1], node_coordinates[v2], node_coordinates[v3], node_coordinates[v4]
            
            # Compute tetrahedron volume
            matrix = np.vstack([pos1 - pos4, pos2 - pos4, pos3 - pos4]).T
            volume = abs(np.linalg.det(matrix)) / 6.0
            
            # Get the node labels for all nodes in this tetrahedron
            tetra_node_labels = node_labels[tetra]
            
            # Check if any node is part of the gel
            gel_count = np.sum(tetra_node_labels[:, group_to_idx['gel']])
            # Check if all nodes are part of the shell
            shell_count = np.sum(tetra_node_labels[:, group_to_idx['shell']])
            
            # Assign material based on node composition
            if gel_count == 4 and shell_count <= 3:
                element_materials[i] = group_to_idx['gel']  # gel material
            else:
                element_materials[i] = group_to_idx['shell']  # shell material
            
            # Determine density based on material
            material_density = self.mass_density[element_materials[i]]
            element_mass = volume * material_density
            
            # Distribute element mass to vertices
            for vertex_idx in tetra:
                vertex_masses[vertex_idx] += element_mass / 4.0  # Equal distribution
        
        max_z = np.max(node_coordinates[:, 2])
        z_translation = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface - max_z
        translation_vector = np.array([0, 0, z_translation])
        node_coordinates = node_coordinates + translation_vector

        self.node_coordinates = node_coordinates
        self.tetrahedra_npy = all_tetrahedra
        self.outer_surface_triangles = surface_triangles
        self.node_labels = node_labels
        self.element_materials_npy = element_materials
        self.vertex_masses_npy = vertex_masses
        
        self.is_fixed_layer = ti.field(int, len(self.node_coordinates))
        is_fixed_layer_data = self.node_labels[:,-1].astype(np.int32)
        self.is_fixed_layer.from_numpy(is_fixed_layer_data)
        
        self.num_vertices = len(self.node_coordinates)
        self.num_tetrahedra = len(self.tetrahedra_npy)
        self.num_contact_surface_triangles = len(self.outer_surface_triangles)

        self.element_materials = ti.field(dtype=ti.i32, shape=(self.num_tetrahedra,), needs_grad=False)
        self.vertex_mass = ti.field(dtype=ti.f32, shape=(self.num_vertices,), needs_grad=False)
        self.element_materials.from_numpy(self.element_materials_npy)
        self.vertex_mass.from_numpy(self.vertex_masses_npy)

        self.vertex_positions_local_initial = ti.Vector.field(3, float, self.num_vertices, needs_grad=False)
        self.vertex_positions_local_initial.from_numpy(self.node_coordinates.astype(np.float32))

        self.tetrahedra = ti.Vector.field(4, int, self.num_tetrahedra)
        self.tetrahedra.from_numpy(self.tetrahedra_npy.astype(np.int32))
        self.contact_surface = ti.Vector.field(3, int, self.num_contact_surface_triangles) # surface triangle mesh
        self.contact_surface.from_numpy(self.outer_surface_triangles.astype(np.int32))

        self.projection_2d_dome_surface_nodes_deformed = ti.Vector.field(2, float, self.surface_node_tags_npy.shape[0], needs_grad=False)
        self.projection_2d_dome_surface_nodes_undeformed = ti.Vector.field(2, float, self.surface_node_tags_npy.shape[0], needs_grad=False)

        self.clock_arms_node_idxs = ti.field(int, (2,), needs_grad=False)
        self.projection_2d_clock_arms = ti.Vector.field(2, float, (2,), needs_grad=False)

    @ti.kernel
    def project_surface_nodes_to_2d(self, surface_nodes: ti.types.ndarray(), output_projections: ti.types.ndarray()):
        for i in range(surface_nodes.shape[0]):
            pos = ti.Vector([surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2]])
            proj = self.fisheye_model.project_3d_2d(pos)
            output_projections[i, 0] = proj[0]
            output_projections[i, 1] = proj[1]

    def initialise_camera_model(self):
        self.marker_interpolation_knn_k = 5

        initial_camera_image = cv2.imread(SYSTEM_PARAMS.files.vitactip_photo_default_state)
        initial_marker_positions, _, _ = self.fisheye_model.get_marker_image(initial_camera_image)
        marker_visualization_image = initial_camera_image.copy()
        for marker_position in initial_marker_positions:
            marker_center = (int(round(marker_position[0])), int(round(marker_position[1])))
            cv2.circle(marker_visualization_image, marker_center, radius=5, color=(0, 0, 255), thickness=2)
        cv2.imwrite(SYSTEM_PARAMS.files.vitactip_photo_default_state_detected_markers, marker_visualization_image)

        # OpenCV has camera.position(0,0,0), camera.lookat(0,0,1), camera.up(0,-1,0)
        surface_nodes_z_up = self.node_coordinates[self.dome_surface_node_tags_npy]
        surface_node_projections_2d = np.zeros((len(surface_nodes_z_up), 2), dtype=np.float32)
        self.project_surface_nodes_to_2d(surface_nodes_z_up, surface_node_projections_2d)

        surface_node_visualization = initial_camera_image.copy()
        for projected_point in surface_node_projections_2d:
            point_center = (int(round(projected_point[0])), int(round(projected_point[1])))
            cv2.circle(surface_node_visualization, point_center, radius=3, color=(0, 255, 0), thickness=2)
        cv2.imwrite(SYSTEM_PARAMS.files.vitactip_photo_default_state_dome_surface_3d_vertices_projected_to_2d, surface_node_visualization)

        marker_interpolation_indices = []
        marker_interpolation_weights = []
        interpolated_marker_positions_2d = []
        for marker_idx in range(initial_marker_positions.shape[0]):
            distances_to_projections = np.linalg.norm(initial_marker_positions[marker_idx,0:2] - surface_node_projections_2d, axis=1)
            neighbor_indices = np.argpartition(distances_to_projections, self.marker_interpolation_knn_k)
            k_nearest_indices = neighbor_indices[:self.marker_interpolation_knn_k]
            inverse_distances = 1/distances_to_projections[k_nearest_indices]
            total_inverse_distance = np.sum(inverse_distances)
            interpolation_weights = inverse_distances / total_inverse_distance
            interpolated_position = np.matmul(surface_node_projections_2d[k_nearest_indices].T, interpolation_weights).T
            interpolated_marker_positions_2d.append(interpolated_position)
            marker_interpolation_indices.append(k_nearest_indices)
            marker_interpolation_weights.append(interpolation_weights)
        interpolated_marker_positions_2d = np.array(interpolated_marker_positions_2d)
        marker_interpolation_indices = np.array(marker_interpolation_indices)
        marker_interpolation_weights = np.array(marker_interpolation_weights)

        surface_node_visualization = initial_camera_image.copy()
        for projected_point in interpolated_marker_positions_2d:
            point_center = (int(round(projected_point[0])), int(round(projected_point[1])))
            cv2.circle(surface_node_visualization, point_center, radius=3, color=(0, 255, 0), thickness=2)
        cv2.imwrite(SYSTEM_PARAMS.files.vitactip_photo_default_state_interpolated_marker_positions_2d, surface_node_visualization)

        self.num_markers = len(interpolated_marker_positions_2d)

        self.deformed_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=True)
        self.undeformed_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=True)
        self.initial_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=True)
        self.initial_undeformed_vertices_after_applying_transformation_matrix = ti.Vector.field(3, float, self.num_markers, needs_grad=True)

        self.marker_interpolation_weights = ti.Vector.field(self.marker_interpolation_knn_k, float, self.num_markers, needs_grad=True)
        self.marker_interpolation_weights.from_numpy(marker_interpolation_weights.astype(np.float32))
        self.marker_interpolation_indices = ti.Vector.field(self.marker_interpolation_knn_k, int, self.num_markers)
        self.marker_interpolation_indices.from_numpy(marker_interpolation_indices.astype(np.int32))

    def set_up_physical_state(self):
        # vertex_positions_ideal: ideal/undeformed vertex positions in m
        self.vertices_undeformed_A = ti.Vector.field(3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices), needs_grad=True)
        self.vertices_B = ti.Vector.field(3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices), needs_grad=False)
        # vertex_positions_deformed: current deformed vertex positions in m
        self.vertices_deformed_A = ti.Vector.field(3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices), needs_grad=True)
        # vertex_velocities: vertex velocities in m/s
        self.vertex_velocities = ti.Vector.field(3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices), needs_grad=True)

        # deformation_gradient_inverse: inverse of deformation gradient matrix (dimensionless)
        self.deformation_gradient_inverse = ti.Matrix.field(3, 3, float, self.num_tetrahedra, needs_grad=True)
        # element_potential_energy: potential energy of each tetrahedral element in J
        self.element_potential_energy = ti.field(float, self.num_tetrahedra, needs_grad=True)  # potential energy of each face (Neo-Hookean)

        # contact_forces_on_vertices: external contact forces applied to vertices in N
        self.contact_forces_on_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices), needs_grad=True) # contact force between FEM node to the closest particle
        # surface_force_resultant: total surface force vector in N
        self.total_surface_force = ti.Vector.field(3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames), needs_grad=True) # surface aggreated 3-axis forces

        # contact model parameters (default)
        # sensor_outward_normal: outward normal direction vector (dimensionless)
        self.sensor_outward_normal = ti.Vector.field(3, float, (), needs_grad=True)

        ## control parameters
        # global_translational_velocity: global translational velocity in m/s
        self.translation_A = ti.Vector.field(3, ti.f32, shape = (), needs_grad=True)
        self.R_BA_quat = ti.Vector.field(4, ti.f32, shape = (), needs_grad=True)

        # local_translational_velocity: local translational velocity in m/s
        self.translation_CD = ti.Vector.field(3, ti.f32, shape = (), needs_grad=True)
        # local_angular_velocity: local angular velocity in degrees/s
        self.R_CD = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=True)

        # homogeneous_rotation_matrix: 3x3 rotation matrix (dimensionless)
        self.R_BA = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=True)
        # world_rotation_matrix: world coordinate rotation matrix (dimensionless)
        self.R_CA = ti.Matrix.field(3, 3, ti.f32, shape = ())
        # local_rotation_matrix: local coordinate rotation matrix (dimensionless)
        self.R_BC = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=True)
        # inverse_rotation_matrix: inverse of rotation matrix (dimensionless)
        self.R_AB = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=True)

        # homogeneous_transformation_matrix: 4x4 homogeneous transformation matrix (dimensionless)
        self.T_BA = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=True)
        # world_transformation_matrix: world coordinate transformation matrix (dimensionless)
        self.T_CA = ti.Matrix.field(4, 4, ti.f32, shape = ())
        # local_transformation_matrix: local coordinate transformation matrix (dimensionless)
        self.T_BC = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=True)
        # inverse_transformation_matrix: inverse of transformation matrix (dimensionless)
        self.T_AB = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=True)
        # delta_transformation_matrix: incremental transformation matrix (dimensionless)
        self.T_CD = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=True)
        self.T_A = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=True)
        # vertex_control_velocities: prescribed control velocities for vertices in m/s
        self.vertex_control_velocities = ti.Vector.field(3, float, shape = (self.num_vertices), needs_grad=True)
        # signed_distance_function: signed distance to surface in m
        self.signed_distance_function = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        # simulation_cache: cache for gradient computation (dimensionless)
        self.simulation_cache = dict() # for grad backward

        # translation_vector: translation vector in m
        self.translation_CD = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.R_A = ti.Matrix.field(3, 3, dtype=float, shape=(), needs_grad=False)
        self.R_A_quat = ti.Vector.field(4, ti.f32, shape = (), needs_grad=True)

    def set_up_pose(self, pose):
        rotation_object = R.from_quat(pose[3:])
        initial_rotation_matrix = rotation_object.as_matrix()
        initial_transformation_matrix = np.eye(4)
        initial_transformation_matrix[0:3,0:3] = initial_rotation_matrix
        initial_transformation_matrix[0,3] = pose[0]; initial_transformation_matrix[1,3] = pose[1]; initial_transformation_matrix[2,3] = pose[2]

        # abbreviations
        # A - global
        # B - local initial
        # C - local current time step
        # D - local next time step
        # T_XY - transformation from X to Y
        # R_XY - rotation from X to Y

        self.R_BC[None] = np.eye(3)
        self.R_CA[None] = initial_rotation_matrix
        self.R_BA[None] = self.R_CA[None] @ self.R_BC[None]
        self.R_AB[None] = self.R_BA[None].transpose()

        self.T_BC[None] = np.eye(4)
        self.T_CA[None] = initial_transformation_matrix
        self.T_BA[None] = self.T_CA[None] @ self.T_BC[None]
        self.T_AB[None] = self.T_BA[None].inverse()

        self.T_CD[None] = np.eye(4)
        self.set_up_pose_helper()
    
    def set_up_pose_print(self):
        print()
        print("\n=== Class Fields and Variables Used in set_up_pose ===")
        print("\nClass Fields:")
        print(f"local_rotation_matrix:\n{self.R_BC[None]}")
        print(f"world_rotation_matrix:\n{self.R_AB[None]}")
        print(f"homogeneous_rotation_matrix:\n{self.R_BA[None]}")
        print(f"inverse_rotation_matrix:\n{self.R_AB[None]}")
        print(f"local_transformation_matrix:\n{self.T_BC[None]}")
        print(f"world_transformation_matrix:\n{self.T_AB[None]}")
        print(f"homogeneous_transformation_matrix:\n{self.T_BA[None]}")
        print(f"inverse_transformation_matrix:\n{self.T_AB[None]}")
        print()

    @ti.kernel
    def extract_markers(self, frame_idx: ti.i32):
        for i in range(self.dome_surface_node_tags.shape[0]):
            surface_node_idx = self.dome_surface_node_tags[i]
            deformed_pos_global = self.vertices_deformed_A[frame_idx, surface_node_idx]
            undeformed_pos_global = self.vertices_undeformed_A[frame_idx, surface_node_idx]
            homogeneous_deformed_pos_global = ti.Vector([deformed_pos_global[0], deformed_pos_global[1], deformed_pos_global[2], 1.0])
            homogeneous_udeformed_pos_global = ti.Vector([undeformed_pos_global[0], undeformed_pos_global[1], undeformed_pos_global[2], 1.0])
            deformed_pos_local_initial = self.T_AB[None] @ homogeneous_deformed_pos_global
            undeformed_pos_local_initial = self.T_AB[None] @ homogeneous_udeformed_pos_global
            deformed_pos_local_initial_ti = ti.Vector([deformed_pos_local_initial[0], deformed_pos_local_initial[1], deformed_pos_local_initial[2]])
            undeformed_pos_local_initial_ti = ti.Vector([undeformed_pos_local_initial[0], undeformed_pos_local_initial[1], undeformed_pos_local_initial[2]])
            camera_space_deformed_2d = self.fisheye_model.project_3d_2d(deformed_pos_local_initial_ti)
            camera_space_undeformed_2d = self.fisheye_model.project_3d_2d(undeformed_pos_local_initial_ti)
            self.projection_2d_dome_surface_nodes_deformed[i] += camera_space_deformed_2d
            self.projection_2d_dome_surface_nodes_undeformed[i] = camera_space_undeformed_2d

        for i in range(self.num_markers):
            nearest_surface_indices = self.marker_interpolation_indices[i]
            interpolation_weights = self.marker_interpolation_weights[i]
            interpolated_deformed_pos_2d = ti.Vector([0.0, 0.0])
            interpolated_undeformed_pos_2d = ti.Vector([0.0, 0.0])
            for neighbor_idx in range(self.marker_interpolation_knn_k):
                interpolated_deformed_pos_2d += interpolation_weights[neighbor_idx] * self.projection_2d_dome_surface_nodes_deformed[nearest_surface_indices[neighbor_idx]]
                interpolated_undeformed_pos_2d += interpolation_weights[neighbor_idx] * self.projection_2d_dome_surface_nodes_undeformed[nearest_surface_indices[neighbor_idx]]
            self.deformed_markers[i] += interpolated_deformed_pos_2d
            self.undeformed_markers[i] = interpolated_undeformed_pos_2d

    @ti.kernel
    def copy_markers_to_initial_markers_for_drift_correction(self):
        for marker_idx in range(self.num_markers):
            self.initial_markers[marker_idx] = self.undeformed_markers[marker_idx]
    
    def get_keypoint_idxs(self):
        # Convert the field to numpy array for easier max operations
        points = self.projection_2d_dome_surface_nodes_deformed.to_numpy()
        # Find index of point with maximum x-coordinate
        idx = np.argmax(points[:, 0])
        self.clock_arms_node_idxs[0] = self.dome_surface_node_tags[idx]
        # Find index of point with maximum y-coordinate
        idx = np.argmax(points[:, 1])
        self.clock_arms_node_idxs[1] = self.dome_surface_node_tags[idx]

    def save_predicted_markers_to_image(self):
        initial_camera_image = cv2.imread(SYSTEM_PARAMS.files.vitactip_photo_default_state)
        surface_node_visualization = initial_camera_image.copy()
        with open(SYSTEM_PARAMS.files.initial_vertex_positions_undeformed, 'wb') as f:
            pickle.dump(self.vertices_B.to_numpy()[0], f)
        for projected_point in self.initial_markers.to_numpy():
            point_center = (int(round(projected_point[0])), int(round(projected_point[1])))
            cv2.circle(surface_node_visualization, point_center, radius=3, color=(0, 255, 0), thickness=2)
        cv2.imwrite(SYSTEM_PARAMS.files.vitactip_photo_default_state_predicted_markers, surface_node_visualization)

    @ti.kernel
    def test_mapping_from_global_space_to_camera_space(self):
        for i in range(self.vertices_undeformed_A.shape[0]):
            initial_vertex_pos = self.vertices_undeformed_A[0, i]
            homogeneous_initial_pos = ti.Vector([initial_vertex_pos[0], initial_vertex_pos[1], initial_vertex_pos[2], 1.0])
            transformed_initial_pos = self.T_AB[None] @ homogeneous_initial_pos
            camera_space_initial_pos = ti.Vector([transformed_initial_pos[0], transformed_initial_pos[1], transformed_initial_pos[2]])
            self.vertices_B[0, i] = camera_space_initial_pos
    
    def debug_marker_drift(self, ts):
        print(f'ts: {ts}; marker coords: {self.deformed_markers[0]}')
    
    @ti.kernel
    def extract_clock_arm_2d_projections(self, frame_idx: ti.i32):
        for i in range(self.clock_arms_node_idxs.shape[0]):
            node_idx = self.clock_arms_node_idxs[i]
            undeformed_vertex_pos = self.vertices_undeformed_A[frame_idx, node_idx]
            homogeneous_undeformed_pos = ti.Vector([undeformed_vertex_pos[0], undeformed_vertex_pos[1], undeformed_vertex_pos[2], 1.0])
            transformed_undeformed_pos = self.T_AB[None] @ homogeneous_undeformed_pos
            camera_space_undeformed_pos = ti.Vector([transformed_undeformed_pos[0], transformed_undeformed_pos[1], transformed_undeformed_pos[2]])
            camera_space_undeformed_2d = self.fisheye_model.project_3d_2d(camera_space_undeformed_pos)
            self.projection_2d_clock_arms[i] = camera_space_undeformed_2d

    @ti.kernel
    def set_vel(self, f:ti.i32):
        for p in range(self.num_vertices):
            self.vertex_velocities[f, p] = self.vertex_control_velocities[p]

    @ti.kernel
    def set_pose_control_1(self):
        self.R_CD[None] = self.R_CA[None].transpose() @ self.R_A[None] @ self.R_CA[None]
        self.translation_CD[None] = self.R_CA[None].transpose() @ self.translation_A[None]

        self.translation_CD[None] = self.translation_CD[None] * SYSTEM_PARAMS.contact.dt * (SYSTEM_PARAMS.contact.num_sub_frames -1)

        self.T_CD[None] = ti.Matrix.identity(float, 4)
    
    @ti.kernel
    def set_pose_control_2(self):
        for i, j in ti.ndrange(3, 3):
            self.T_CD[None][i, j] = self.R_CD[None][i, j]
        for i in range(3):
            self.T_CD[None][i, 3] = self.translation_CD[None][i]

    @ti.kernel
    def set_pose_control_3(self):
        self.T_A[None] = self.T_CA[None] @ self.T_CD[None] @ self.T_CA[None].inverse()

        self.T_BC[None] = self.T_CD[None] @ self.T_BC[None]
        self.T_BA[None] = self.T_CA[None] @ self.T_BC[None]
        self.T_AB[None] = self.T_BA[None].inverse()

        self.R_BC[None] = self.R_CD[None] @ self.R_BC[None]
        self.R_BA[None] = self.R_CA[None] @ self.R_BC[None]
        self.R_AB[None] = self.R_BA[None].transpose()

    def set_pose_control_print(self):
        print()
        print("\n=== Class Fields and Variables Used in set_pose_control ===")
        print("Class Fields:")
        print(f"local_rotation_over_big_step_matrix:\n{self.R_CD[None]}")
        print(f"inverse_rotation_matrix:\n{self.R_AB[None]}")
        print(f"global_rotation_over_big_step_matrix:\n{self.R_A[None]}")
        print(f"local_translational_velocity:\n{self.translation_CD[None]}")
        print(f"global_translational_velocity:\n{self.translation_A[None]}")
        print(f"translation_vector:\n{self.translation_CD[None]}")
        print(f"transformation_matrix:\n{self.T_CD[None]}")
        print(f"delta_transformation_matrix:\n{self.T_CD[None]}")
        print(f"world_transformation_matrix:\n{self.T_AB[None]}")
        print(f"local_transformation_matrix:\n{self.T_BC[None]}")
        print(f"homogeneous_transformation_matrix:\n{self.T_BA[None]}")
        print(f"inverse_transformation_matrix:\n{self.T_AB[None]}")
        print(f"local_rotation_matrix:\n{self.R_BC[None]}")
        print(f"homogeneous_rotation_matrix:\n{self.R_BA[None]}")
        print(f"world_rotation_matrix:\n{self.R_AB[None]}")
        
        print("\nSystem Parameters:")
        print(f"SYSTEM_PARAMS.contact.dt:\n{SYSTEM_PARAMS.contact.dt}")
        print(f"SYSTEM_PARAMS.contact.num_sub_frames:\n{SYSTEM_PARAMS.contact.num_sub_frames}")
        print()

    @ti.kernel
    def set_pose_control_2_bp_unused(self):
        return
        rot_v = self.R_CD[None] * SYSTEM_PARAMS.contact.dt * (SYSTEM_PARAMS.contact.num_sub_frames -1)
        trans_v = self.translation_CD[None] * SYSTEM_PARAMS.contact.dt * (SYSTEM_PARAMS.contact.num_sub_frames -1)
        trans_mat, rot_mat = self.eul2mat_unused(rot_v, trans_v)
        self.T_CD[None] = self.T_AB[None] @ trans_mat @ (self.T_AB[None].inverse())

        self.T_AB[None] = self.T_BA[None].inverse()
        self.R_AB[None] = self.R_BA[None].inverse()

    @ti.kernel
    def set_control_vel(self, f:ti.i32):
        for i in range(self.num_vertices):
            current_vertex_positions_undeformed = self.vertices_undeformed_A[f, i]
            target_vertex_positions_undeformed = self.T_A[None] @ ti.Vector([current_vertex_positions_undeformed[0], current_vertex_positions_undeformed[1], current_vertex_positions_undeformed[2], 1.0]) # 4 x 1 homogeneous
            self.vertex_control_velocities[i][0] = (target_vertex_positions_undeformed[0] - current_vertex_positions_undeformed[0]) / (SYSTEM_PARAMS.contact.dt * (SYSTEM_PARAMS.contact.num_sub_frames -1))
            self.vertex_control_velocities[i][1] = (target_vertex_positions_undeformed[1] - current_vertex_positions_undeformed[1]) / (SYSTEM_PARAMS.contact.dt * (SYSTEM_PARAMS.contact.num_sub_frames -1))
            self.vertex_control_velocities[i][2] = (target_vertex_positions_undeformed[2] - current_vertex_positions_undeformed[2]) / (SYSTEM_PARAMS.contact.dt * (SYSTEM_PARAMS.contact.num_sub_frames -1))

    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for k in range(self.num_contact_surface_triangles):
            a, b, c = self.contact_surface[k]
            self.total_surface_force[f] += 1/3 * self.contact_forces_on_vertices[f,a] * self.dx
            self.total_surface_force[f] += 1/3 * self.contact_forces_on_vertices[f,b] * self.dx
            self.total_surface_force[f] += 1/3 * self.contact_forces_on_vertices[f,c] * self.dx

    def compute_current_orientation(self):
        rot_mat = self.R_BA.to_numpy()
        rotation_object = R.from_matrix(rot_mat)
        self.R_BA_quat.from_numpy(rotation_object.as_quat())

    @ti.kernel
    def set_up_pose_helper(self):
        for idx in range(self.num_vertices):
            vertex_position_local_initial = self.vertex_positions_local_initial[idx]
            vertex_position_global = self.T_CA[None] @ ti.Vector([vertex_position_local_initial[0], vertex_position_local_initial[1], vertex_position_local_initial[2], 1.0])

            self.vertices_deformed_A[0, idx] = ti.Vector([vertex_position_global[0], vertex_position_global[1], vertex_position_global[2]])
            self.vertices_undeformed_A[0, idx] = self.vertices_deformed_A[0, idx]

        for i in range(self.num_tetrahedra):
            ia, ib, ic, id = self.tetrahedra[i]
            a, b, c, d = self.vertices_deformed_A[0, ia], self.vertices_deformed_A[0, ib], self.vertices_deformed_A[0, ic], self.vertices_deformed_A[0, id]
            deformation_gradient = ti.Matrix.cols([a - d, b - d, c - d])
            self.deformation_gradient_inverse[i] = deformation_gradient.inverse()

        self.sensor_outward_normal[None] = self.R_BA[None] @ ti.Vector([0.0, 0.0, 1.0])

    @ti.func
    def find_closest(self, grid_p, f):
        """
        Find the closest triangle segment to a given grid point.
        
        Args:
            grid_p: grid point position vector in m
            f: current simulation frame (dimensionless)
            
        Returns:
            cur_min_idx: index of closest triangle segment (dimensionless)
        """
        # cur_min_offset: distance in m
        cur_min_offset = SYSTEM_PARAMS.vitactip.collision_search_distance # arbitrary large value
        # cur_min_idx: triangle index (dimensionless)
        cur_min_idx = -1
        for k in range(self.num_contact_surface_triangles):
            a, b, c = self.contact_surface[k]
            # p_1, p_2, p_3: triangle vertex positions in m
            p_1 = self.vertices_deformed_A[f, a] # triangle's 1st node
            p_2 = self.vertices_deformed_A[f, b] # triangle's 2nd node
            p_3 = self.vertices_deformed_A[f, c] # triangle's 3rd node
            # p_c: triangle centroid position in m
            p_c = 1/3 * (p_1 + p_2 + p_3) # center of the segment
            # offset_p: distance in m
            offset_p = (p_c - grid_p).norm(SYSTEM_PARAMS.contact.norm_eps) # distance to the center point of the segment

            if (offset_p < cur_min_offset):
                cur_min_offset = offset_p
                cur_min_idx = k

        return cur_min_idx

    @ti.func
    def find_sdf(self, point_position, point_velocity, triangle_index, frame, collision_offset = 0.0):
        """
        Calculate signed distance function (SDF) for a point relative to a triangle.
        
        Args:
            point_position: point position vector in m
            point_velocity: point velocity vector in m/s
            triangle_index: triangle segment index (dimensionless)
            frame: current simulation frame (dimensionless)
            collision_offset: collision offset distance in m (default: 0.0)
            
        Returns:
            signed_distance: signed distance to triangle surface in m
            surface_normal: unit normal vector (dimensionless)
            relative_velocity: relative velocity vector in m/s
            is_contact: boolean indicating contact (dimensionless)
        """
        vertex1_idx, vertex2_idx, vertex3_idx = self.contact_surface[triangle_index]
        # vertex1_pos, vertex2_pos, vertex3_pos: triangle vertex positions in m
        vertex1_pos = self.vertices_deformed_A[frame, vertex1_idx]
        vertex2_pos = self.vertices_deformed_A[frame, vertex2_idx]
        vertex3_pos = self.vertices_deformed_A[frame, vertex3_idx]

        # triangle_normal: unit normal vector (dimensionless)
        triangle_normal = ti.math.cross(vertex2_pos-vertex1_pos, vertex3_pos-vertex1_pos) # plane's norm
        triangle_normal = triangle_normal.normalized(SYSTEM_PARAMS.contact.norm_eps)
        # normal_direction: sign value (dimensionless)
        normal_direction = ti.math.sign(triangle_normal.dot(self.sensor_outward_normal[None]))
        triangle_normal = normal_direction * triangle_normal # facing up

        # point_to_vertex1: position vector in m
        point_to_vertex1 = point_position - vertex1_pos # vector from the first node to the particle
        # signed_distance: distance in m
        signed_distance = point_to_vertex1.dot(triangle_normal) # distance to the plane
        # point_projected: position vector in m
        point_projected = point_position - signed_distance * triangle_normal # projection of point on the segment
        # surface_normal: unit normal vector (dimensionless)
        surface_normal = -1* triangle_normal

        # edge1, edge2: edge vectors in m
        edge1 = vertex3_pos - vertex1_pos
        edge2 = vertex2_pos - vertex1_pos
        # point_projected_rel: relative position vector in m
        point_projected_rel = point_projected - vertex1_pos
        # dot_edge1_edge1, dot_edge1_edge2, dot_edge1_point, dot_edge2_edge2, dot_edge2_point: dot products in m^2
        dot_edge1_edge1 = edge1.dot(edge1)
        dot_edge1_edge2 = edge1.dot(edge2)
        dot_edge1_point = edge1.dot(point_projected_rel)
        dot_edge2_edge2 = edge2.dot(edge2)
        dot_edge2_point = edge2.dot(point_projected_rel)
        # inv_denominator: inverse area factor in m^-2
        inv_denominator = 1 / (dot_edge1_edge1 * dot_edge2_edge2 - dot_edge1_edge2 * dot_edge1_edge2)
        # barycentric_u, barycentric_v: barycentric coordinates (dimensionless)
        barycentric_u = (dot_edge2_edge2 * dot_edge1_point - dot_edge1_edge2 * dot_edge2_point) * inv_denominator
        barycentric_v = (dot_edge1_edge1 * dot_edge2_point - dot_edge1_edge2 * dot_edge1_point) * inv_denominator

        ### correct with an offset for pbd collision
        signed_distance -= collision_offset
        # relative_velocity: velocity vector in m/s
        relative_velocity = point_velocity - 1/3 * (self.vertex_velocities[frame, vertex1_idx] + self.vertex_velocities[frame, vertex2_idx] + self.vertex_velocities[frame, vertex3_idx])
        # is_contact: boolean (dimensionless)
        is_contact = signed_distance < 0 and barycentric_u >= 0 and barycentric_v >= 0.0 and (barycentric_u + barycentric_v <= 1)
        return signed_distance, surface_normal, relative_velocity, is_contact

    def reset(self):
        self.contact_forces_on_vertices.fill(0.0)
        self.total_surface_force.fill(0.0)

        self.deformed_markers.fill(0.0)
        self.projection_2d_dome_surface_nodes_deformed.fill(0.0)
        for i in range(1, self.vertices_deformed_A.shape[0]):
            for j in range(self.vertices_deformed_A.shape[1]):
                self.vertices_deformed_A[i, j] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def update_contact_force(self, triangle_index, contact_force, frame):
        """
        Distribute contact force among triangle vertices.
        
        Args:
            triangle_index: triangle segment index (dimensionless)
            contact_force: contact force vector in N
            frame: current simulation frame (dimensionless)
        """
        vertex1_idx, vertex2_idx, vertex3_idx = self.contact_surface[triangle_index]
        # Distribute contact force equally among triangle vertices
        self.contact_forces_on_vertices[frame, vertex1_idx] += 1/3 * contact_force
        self.contact_forces_on_vertices[frame, vertex2_idx] += 1/3 * contact_force
        self.contact_forces_on_vertices[frame, vertex3_idx] += 1/3 * contact_force

    @ti.func
    def compute_hourglass_forces(self, vertex1_pos, vertex2_pos, vertex3_pos, vertex4_pos,
                               vertex1_vel, vertex2_vel, vertex3_vel, vertex4_vel,
                               tetra_volume, mu, lam):
        """
        Compute hourglass control forces for a tetrahedral element.
        
        Args:
            vertex1_pos, vertex2_pos, vertex3_pos, vertex4_pos: vertex positions in m
            vertex1_vel, vertex2_vel, vertex3_vel, vertex4_vel: vertex velocities in m/s
            tetra_volume: tetrahedral volume in m^3
            mu: shear modulus in Pa
            lam: Lamé's first parameter in Pa
            
        Returns:
            hourglass_force_matrix: hourglass force matrix in N
        """
        # Compute element center and average velocity
        center_pos = (vertex1_pos + vertex2_pos + vertex3_pos + vertex4_pos) / 4.0
        center_vel = (vertex1_vel + vertex2_vel + vertex3_vel + vertex4_vel) / 4.0
        
        # Compute relative positions and velocities
        rel_pos1 = vertex1_pos - center_pos
        rel_pos2 = vertex2_pos - center_pos
        rel_pos3 = vertex3_pos - center_pos
        rel_pos4 = vertex4_pos - center_pos
        
        rel_vel1 = vertex1_vel - center_vel
        rel_vel2 = vertex2_vel - center_vel
        rel_vel3 = vertex3_vel - center_vel
        rel_vel4 = vertex4_vel - center_vel
        
        # Compute hourglass modes (non-physical deformation patterns)
        # Using strain-rate based approach
        strain_rate1 = rel_vel1.outer_product(rel_pos1)
        strain_rate2 = rel_vel2.outer_product(rel_pos2)
        strain_rate3 = rel_vel3.outer_product(rel_pos3)
        strain_rate4 = rel_vel4.outer_product(rel_pos4)
        
        # Average strain rate
        avg_strain_rate = (strain_rate1 + strain_rate2 + strain_rate3 + strain_rate4) / 4.0
        
        # Compute hourglass mode contributions
        hg1 = strain_rate1 - avg_strain_rate
        hg2 = strain_rate2 - avg_strain_rate
        hg3 = strain_rate3 - avg_strain_rate
        
        # Compute hourglass stiffness (based on material properties)
        hg_stiffness = self.hourglass_coefficient[None] * self.hourglass_modulus_scale[None] * (mu + lam)
        
        # Compute hourglass forces
        hg_force_matrix = -hg_stiffness * tetra_volume * ti.Matrix.cols([
            hg1.transpose() @ rel_pos1,
            hg2.transpose() @ rel_pos2,
            hg3.transpose() @ rel_pos3
        ])
        
        return hg_force_matrix

    @ti.kernel
    def update_internal_forces(self, frame:ti.i32):
        """
        Update internal forces for all tetrahedral elements using finite element method.
        Computes deformation gradient, stress tensor, and applies forces to vertices.
        Includes Rayleigh damping and hourglass control forces.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for tetra_idx in range(self.num_tetrahedra):
            vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx = self.tetrahedra[tetra_idx]
            # Get vertex positions and velocities
            vertex1_pos = self.vertices_deformed_A[frame, vertex1_idx]
            vertex2_pos = self.vertices_deformed_A[frame, vertex2_idx]
            vertex3_pos = self.vertices_deformed_A[frame, vertex3_idx]
            vertex4_pos = self.vertices_deformed_A[frame, vertex4_idx]
            
            vertex1_vel = self.vertex_velocities[frame, vertex1_idx]
            vertex2_vel = self.vertex_velocities[frame, vertex2_idx]
            vertex3_vel = self.vertex_velocities[frame, vertex3_idx]
            vertex4_vel = self.vertex_velocities[frame, vertex4_idx]

            # Calculate elastic forces
            deformation_matrix = ti.Matrix.cols([vertex1_pos - vertex4_pos, vertex2_pos - vertex4_pos, vertex3_pos - vertex4_pos])
            tetra_volume = ti.abs(deformation_matrix.determinant()) / 6
            deformation_gradient = deformation_matrix @ self.deformation_gradient_inverse[tetra_idx]

            mu = self.mu[self.element_materials[tetra_idx]]
            lam = self.lam[self.element_materials[tetra_idx]]

            # Neo-hookean calculations
            jacobian = deformation_gradient.determinant()
            first_invariant = (deformation_gradient.transpose() @ deformation_gradient).trace()
            dJ_dF0 = deformation_gradient[:,1].cross(deformation_gradient[:,2])
            dJ_dF1 = deformation_gradient[:,2].cross(deformation_gradient[:,0])
            dJ_dF2 = deformation_gradient[:,0].cross(deformation_gradient[:,1])
            jacobian_derivative = ti.Matrix.cols([dJ_dF0, dJ_dF1, dJ_dF2])
            alpha = 1 + 0.75 * mu/lam
            stress_tensor = mu * (1 - 1/(first_invariant+1)) * deformation_gradient + lam * (jacobian - alpha) * jacobian_derivative

            # Calculate elastic force matrix
            elastic_force_matrix = -tetra_volume * stress_tensor @ self.deformation_gradient_inverse[tetra_idx].transpose()
            
            # Calculate velocity-based damping forces (Rayleigh damping)
            relative_vel1 = vertex1_vel - vertex4_vel
            relative_vel2 = vertex2_vel - vertex4_vel
            relative_vel3 = vertex3_vel - vertex4_vel
            
            # Combine mass and stiffness damping
            damping_force_matrix = -self.rayleigh_damping_alpha[None] * ti.Matrix.cols([relative_vel1, relative_vel2, relative_vel3])
            damping_force_matrix -= self.rayleigh_damping_beta[None] * elastic_force_matrix
            
            # Calculate hourglass control forces if enabled
            force_matrix = elastic_force_matrix + damping_force_matrix
            if self.hourglass_enabled[None] == 1:
                hourglass_force_matrix = self.compute_hourglass_forces(
                    vertex1_pos, vertex2_pos, vertex3_pos, vertex4_pos,
                    vertex1_vel, vertex2_vel, vertex3_vel, vertex4_vel,
                    tetra_volume, mu, lam
                )
                force_matrix += hourglass_force_matrix
            
            # Apply forces to vertices
            vertex_indices = ti.Vector([vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx])
            for k in ti.static(range(3)):
                vertex_force = ti.Vector([force_matrix[j,k] for j in range(3)])
                self.vertex_velocities[frame,vertex_indices[k]] += SYSTEM_PARAMS.contact.dt * vertex_force / self.vertex_mass[vertex_indices[k]]
                self.vertex_velocities[frame,vertex_indices[3]] += -1*SYSTEM_PARAMS.contact.dt * vertex_force / self.vertex_mass[vertex_indices[3]]


    @ti.kernel
    def update_external_forces(self, frame:ti.i32):
        """
        Update external forces and advance vertex positions and velocities.
        Applies external forces, handles fixed boundary conditions, and updates positions.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for vertex_idx in range(self.num_vertices):
            # updated_velocity: velocity vector in m/s
            updated_velocity = ti.Vector([0.0, 0.0, 0.0])
            updated_velocity += self.vertex_velocities[frame,vertex_idx]
            updated_velocity += SYSTEM_PARAMS.contact.dt * self.contact_forces_on_vertices[frame,vertex_idx] / self.vertex_mass[vertex_idx]

            ### stick the bottom layer to be fixed using node_labels information
            # is_fixed_layer: boolean flag (dimensionless)
            is_fixed_layer = self.is_fixed_layer[vertex_idx] == 1
            if is_fixed_layer:
                updated_velocity = self.vertex_control_velocities[vertex_idx]
            self.vertex_velocities[frame+1, vertex_idx] = updated_velocity
            self.vertices_deformed_A[frame+1, vertex_idx] += self.vertices_deformed_A[frame, vertex_idx]
            self.vertices_deformed_A[frame+1, vertex_idx] += SYSTEM_PARAMS.contact.dt * updated_velocity
            # update virtual pos
            self.vertices_undeformed_A[frame+1, vertex_idx] = self.vertices_undeformed_A[frame, vertex_idx] + SYSTEM_PARAMS.contact.dt * self.vertex_control_velocities[vertex_idx]


    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.num_vertices):
            self.vertices_deformed_A[target, p] = self.vertices_deformed_A[source, p]
            self.vertex_velocities[target, p] = self.vertex_velocities[source, p]
            self.vertices_undeformed_A[target, p] = self.vertices_undeformed_A[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), 
                            cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), 
                            cache_rot: ti.types.ndarray(),
                            cache_R_CD: ti.types.ndarray(), cache_R_CA: ti.types.ndarray(), 
                            cache_R_A: ti.types.ndarray(), cache_translation_A: ti.types.ndarray(),
                            cache_T_CD: ti.types.ndarray(), cache_T_CA: ti.types.ndarray(),
                            cache_T_BC: ti.types.ndarray(), cache_T_AB: ti.types.ndarray(),
                            cache_R_BC: ti.types.ndarray(), cache_R_AB: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                self.T_BA[None][j,k] = cache_trans[j,k]
                self.T_CD[None][j,k] = cache_T_CD[j,k]
                self.T_CA[None][j,k] = cache_T_CA[j,k]
                self.T_BC[None][j,k] = cache_T_BC[j,k]
                self.T_AB[None][j,k] = cache_T_AB[j,k]
        for j in range(3):
            for k in range(3):
                self.R_BA[None][j,k] = cache_rot[j,k]
                self.R_CD[None][j,k] = cache_R_CD[j,k]
                self.R_CA[None][j,k] = cache_R_CA[j,k]
                self.R_A[None][j,k] = cache_R_A[j,k]
                self.R_BC[None][j,k] = cache_R_BC[j,k]
                self.R_AB[None][j,k] = cache_R_AB[j,k]
            self.translation_A[None][j] = cache_translation_A[j]
        for p in range(self.num_vertices):
            for i in ti.static(range(3)):
                self.vertices_deformed_A[f, p][i] = cache_pos[p,i]
                self.vertices_undeformed_A[f, p][i] = cache_virtual_pos[p, i]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), 
                         cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), 
                         cache_rot: ti.types.ndarray(),
                         cache_R_CD: ti.types.ndarray(), cache_R_CA: ti.types.ndarray(), 
                         cache_R_A: ti.types.ndarray(), cache_translation_A: ti.types.ndarray(),
                         cache_T_CD: ti.types.ndarray(), cache_T_CA: ti.types.ndarray(),
                         cache_T_BC: ti.types.ndarray(), cache_T_AB: ti.types.ndarray(),
                         cache_R_BC: ti.types.ndarray(), cache_R_AB: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                cache_trans[j,k] = self.T_BA[None][j,k]
                cache_T_CD[j,k] = self.T_CD[None][j,k]
                cache_T_CA[j,k] = self.T_CA[None][j,k]
                cache_T_BC[j,k] = self.T_BC[None][j,k]
                cache_T_AB[j,k] = self.T_AB[None][j,k]
        for j in range(3):
            for k in range(3):
                cache_rot[j,k] = self.R_BA[None][j,k]
                cache_R_CD[j,k] = self.R_CD[None][j,k]
                cache_R_CA[j,k] = self.R_CA[None][j,k]
                cache_R_A[j,k] = self.R_A[None][j,k]
                cache_R_BC[j,k] = self.R_BC[None][j,k]
                cache_R_AB[j,k] = self.R_AB[None][j,k]
            cache_translation_A[j] = self.translation_A[None][j]
        for p in range(self.num_vertices):
            for i in ti.static(range(3)):
                cache_pos[p,i] = self.vertices_deformed_A[f, p][i]
                cache_virtual_pos[p, i] = self.vertices_undeformed_A[f, p][i]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.simulation_cache[cur_step_name] = dict()

        # Original fields
        self.simulation_cache[cur_step_name]['pos'] = torch.zeros((self.num_vertices, 3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['trans_h'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['rot_h'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['virtual_pos'] = torch.zeros((self.num_vertices, 3), dtype=TC_TYPE, device=device)

        # Additional fields
        self.simulation_cache[cur_step_name]['R_CD'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['R_CA'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['R_A'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['translation_A'] = torch.zeros(3, dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['T_CD'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['T_CA'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['T_BC'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['T_AB'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['R_BC'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['R_AB'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)

        self.add_step_to_cache(0, 
            self.simulation_cache[cur_step_name]['pos'],
            self.simulation_cache[cur_step_name]['trans_h'],
            self.simulation_cache[cur_step_name]['virtual_pos'],
            self.simulation_cache[cur_step_name]['rot_h'],
            self.simulation_cache[cur_step_name]['R_CD'],
            self.simulation_cache[cur_step_name]['R_CA'],
            self.simulation_cache[cur_step_name]['R_A'],
            self.simulation_cache[cur_step_name]['translation_A'],
            self.simulation_cache[cur_step_name]['T_CD'],
            self.simulation_cache[cur_step_name]['T_CA'],
            self.simulation_cache[cur_step_name]['T_BC'],
            self.simulation_cache[cur_step_name]['T_AB'],
            self.simulation_cache[cur_step_name]['R_BC'],
            self.simulation_cache[cur_step_name]['R_AB']
        )
        self.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        self.load_step_from_cache(0,
            self.simulation_cache[cur_step_name]['pos'],
            self.simulation_cache[cur_step_name]['trans_h'],
            self.simulation_cache[cur_step_name]['virtual_pos'],
            self.simulation_cache[cur_step_name]['rot_h'],
            self.simulation_cache[cur_step_name]['R_CD'],
            self.simulation_cache[cur_step_name]['R_CA'],
            self.simulation_cache[cur_step_name]['R_A'],
            self.simulation_cache[cur_step_name]['translation_A'],
            self.simulation_cache[cur_step_name]['T_CD'],
            self.simulation_cache[cur_step_name]['T_CA'],
            self.simulation_cache[cur_step_name]['T_BC'],
            self.simulation_cache[cur_step_name]['T_AB'],
            self.simulation_cache[cur_step_name]['R_BC'],
            self.simulation_cache[cur_step_name]['R_AB']
        )

    def get_keypoint_indices(self, f: ti.i32):
        # Convert positions to numpy array
        positions = self.vertices_deformed_A.to_numpy()[f]
        
        # Point A: minimum z coordinate
        z_coords = positions[:, 2]
        point_a_idx = int(np.argmin(z_coords))
        
        # Points B and C: high z coordinate points
        max_z = float(np.max(z_coords))
        z_mask = (z_coords >= (max_z - SYSTEM_PARAMS.vitactip.keypoint_search_z_threshold))
        
        # Point B: max x coordinate among high z points
        x_coords = positions[:, 0]
        x_coords_filtered = x_coords.copy()
        x_coords_filtered[~z_mask] = float('-inf')
        point_b_idx = int(np.argmax(x_coords_filtered))
        
        # Point C: max y coordinate among high z points
        y_coords = positions[:, 1]
        y_coords_filtered = y_coords.copy()
        y_coords_filtered[~z_mask] = float('-inf')
        point_c_idx = int(np.argmax(y_coords_filtered))
        
        return np.array([point_a_idx, point_b_idx, point_c_idx])

    def get_keypoint_indices_numpy_point_a(self):
        # Convert positions to numpy array
        positions = self.node_coordinates
        
        # Point A: max y coordinate
        z_coords = positions[:, 2]
        point_a_idx = int(np.argmax(z_coords))

        return point_a_idx

    def get_keypoint_coordinates(self, f: int, keypoint_indices: np.ndarray) -> np.ndarray:
        """
        Get coordinates for specified keypoint indices at a given frame.
        
        Args:
            f: Frame index
            keypoint_indices: Array of keypoint indices to get coordinates for
            
        Returns:
            numpy array of shape (num_points, 3) containing the coordinates
        """
        # Convert positions to numpy array for the given frame
        positions = self.vertices_undeformed_A.to_numpy()[f]
        
        # Extract coordinates for the specified indices
        coordinates = positions[keypoint_indices]
        
        return coordinates

    @ti.kernel
    def clear_loss_grad(self):
        """
        Clear gradients of all loss-related fields.
        """
        self.youngs_modulus.grad.fill(0.0)
        self.mu.grad.fill(0.0)
        self.lam.grad.fill(0.0)

        self.deformed_markers.grad.fill(0.0)
        self.translation_A.grad[None].fill(0.0)
        self.R_BA.grad[None].fill(0.0)
        self.R_BC.grad[None].fill(0.0)
        self.R_AB.grad[None].fill(0.0)
        self.T_BA.grad[None].fill(0.0)
        self.T_BC.grad[None].fill(0.0)
        self.T_AB.grad[None].fill(0.0)
        self.T_CD.grad[None].fill(0.0)
        self.vertex_control_velocities.grad.fill(0.0)

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        self.total_surface_force.grad.fill(0.0)
        self.contact_forces_on_vertices.grad.fill(0.0)
        for p in range(self.num_vertices):
            for t in range(f):
                self.vertices_deformed_A.grad[t, p].fill(0.0)
                self.vertex_velocities.grad[t, p].fill(0.0)
                self.vertices_undeformed_A.grad[t, p].fill(0.0)
    
###############################################################################

    @ti.func
    def eul2mat_unused(self, rot_v, trans_v):
        return
        # rot_v: euler angles (degrees) for rotation (x,y,z)
        # trans_v: translation (x,y,z)
        rot_v_r = ti.math.radians(rot_v)
        rot_x = rot_v_r[0]
        rot_y = rot_v_r[1]
        rot_z = rot_v_r[2]
        mat_x = ti.Matrix([[1.0, 0.0, 0.0],[0.0, ti.cos(rot_x), -ti.sin(rot_x)],[0.0, ti.sin(rot_x), ti.cos(rot_x)]])
        mat_y = ti.Matrix([[ti.cos(rot_y), 0.0, ti.sin(rot_y)],[0.0, 1.0, 0.0],[-ti.sin(rot_y), 0.0, ti.cos(rot_y)]])
        mat_z = ti.Matrix([[ti.cos(rot_z), -ti.sin(rot_z), 0.0],[ti.sin(rot_z), ti.cos(rot_z), 0.0],[0.0, 0.0, 1.0]])
        mat_R = mat_z @ mat_y @ mat_x
        trans_h = ti.Matrix.identity(float, 4)
        trans_h[0:3, 0:3] = mat_R
        trans_h[0:3, 3] = trans_v
        return trans_h, mat_R

    @ti.kernel
    def copy_grad_unused(self, source: ti.i32, target: ti.i32):
        return
        for p in range(self.num_vertices):
            self.vertices_deformed_A.grad[target, p] = self.vertices_deformed_A.grad[source, p]
            self.vertex_velocities.grad[target, p] = self.vertex_velocities.grad[source, p]
            self.vertices_undeformed_A.grad[target, p] = self.vertices_undeformed_A.grad[source, p]