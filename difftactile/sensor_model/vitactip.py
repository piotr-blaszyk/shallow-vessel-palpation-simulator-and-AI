import taichi as ti
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
from difftactile.sensor_model.fisheye_model import *
from difftactile.main.constants import *

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32


@ti.data_oriented
class ViTacTip:
    def __init__(self):
        self.fisheye_model = FisheyeModel()
        self.set_up_system_params_1()
        self.set_up_system_params_2()
        self.load_mesh()
        self.initialise_camera_model()
        self.set_up_physical_state_1()

    def set_up_system_params_1(self):
        self.dt = ti.field(dtype=float, shape=(), needs_grad=False)
        self.dt[None] = SYSTEM_PARAMS.contact.dt
        self.rayleigh_damping_alpha = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.rayleigh_damping_beta = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.rayleigh_damping_alpha[None] = (
            SYSTEM_PARAMS.vitactip.rayleigh_damping_alpha
        )
        self.rayleigh_damping_beta[None] = SYSTEM_PARAMS.vitactip.rayleigh_damping_beta
        self.mass_density = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.youngs_modulus = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.poissons_ratio = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.lam = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mass_density[None] += SYSTEM_PARAMS.vitactip.single_material.density
        self.youngs_modulus[None] += (
            SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
        )
        self.poissons_ratio[None] += (
            SYSTEM_PARAMS.vitactip.single_material.poissons_ratio
        )

    @ti.kernel
    def set_up_system_params_2(self):
        self.lam[None] += (self.youngs_modulus[None] * self.poissons_ratio[None]) / (
            (1 + self.poissons_ratio[None]) * (1 - 2 * self.poissons_ratio[None])
        )
        self.mu[None] += self.youngs_modulus[None] / (
            2 * (1 + self.poissons_ratio[None])
        )

    def load_mesh(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh, "rb") as f:
            mesh_data = pickle.load(f)
        self.dome_surface_node_tags_npy = mesh_data["dome_surface_node_tags"]
        self.dome_surface_node_tags = ti.field(
            dtype=int,
            shape=(self.dome_surface_node_tags_npy.shape[0],),
            needs_grad=False,
        )
        self.dome_surface_node_tags.from_numpy(self.dome_surface_node_tags_npy)
        self.dome_surface_node_contact_mask = ti.field(
            dtype=int,
            shape=(self.dome_surface_node_tags.shape[0],),
            needs_grad=False,
        )
        if False:
            self.surface_node_tags_npy = mesh_data["surface_node_tags"]
        all_tetrahedra = mesh_data["all_tetrahedra"]
        node_coordinates = mesh_data["node_coordinates"] / 1_000
        node_labels = mesh_data["node_labels"]
        surface_triangles = mesh_data["surface_triangles"]
        group_to_idx = mesh_data["group_to_idx"]
        z_bottom = mesh_data["z_bottom"] / 1_000
        is_fixed_layer = (
            np.abs(node_coordinates[:, 2] - z_bottom)
            < SYSTEM_PARAMS.vitactip.fixed_layer_distance_from_bottom
        )
        node_labels = np.column_stack([node_labels, is_fixed_layer])
        element_materials = np.full(len(all_tetrahedra), fill_value=-1, dtype=np.int32)
        vertex_masses = np.zeros(len(node_coordinates), dtype=np.float32)
        for i, tetra in enumerate(all_tetrahedra):
            v1, v2, v3, v4 = tetra
            pos1, pos2, pos3, pos4 = (
                node_coordinates[v1],
                node_coordinates[v2],
                node_coordinates[v3],
                node_coordinates[v4],
            )
            matrix = np.vstack([pos1 - pos4, pos2 - pos4, pos3 - pos4]).T
            volume = abs(np.linalg.det(matrix)) / 6.0
            tetra_node_labels = node_labels[tetra]
            gel_count = np.sum(tetra_node_labels[:, group_to_idx["gel"]])
            shell_count = np.sum(tetra_node_labels[:, group_to_idx["shell"]])
            if gel_count == 4 and shell_count <= 3:
                element_materials[i] = group_to_idx["gel"]
            else:
                element_materials[i] = group_to_idx["shell"]
            material_density = self.mass_density[None]
            element_mass = volume * material_density
            for vertex_idx in tetra:
                vertex_masses[vertex_idx] += element_mass / 4.0
        
        max_z = node_coordinates[:, 2].max()
        translation_vector = np.array([0, 0, -max_z])
        node_coordinates = node_coordinates + translation_vector
        self.node_coordinates = node_coordinates
        self.tetrahedra_npy = all_tetrahedra
        self.outer_surface_triangles = surface_triangles
        self.node_labels = node_labels
        self.element_materials_npy = element_materials
        self.vertex_masses_npy = vertex_masses
        self.is_fixed_layer = ti.field(
            int, len(self.node_coordinates), needs_grad=False
        )
        is_fixed_layer_data = self.node_labels[:, -1].astype(np.int32)
        self.is_fixed_layer.from_numpy(is_fixed_layer_data)
        with open(SYSTEM_PARAMS.files.is_fixed_layer, "wb") as f:
            pickle.dump(is_fixed_layer_data, f)
        self.num_vertices = len(self.node_coordinates)
        self.num_tetrahedra = len(self.tetrahedra_npy)
        self.num_contact_surface_triangles = len(self.outer_surface_triangles)
        self.element_materials = ti.field(
            dtype=ti.i32, shape=(self.num_tetrahedra,), needs_grad=False
        )
        self.vertex_mass = ti.field(
            dtype=ti.f32, shape=(self.num_vertices,), needs_grad=False
        )
        self.element_materials.from_numpy(self.element_materials_npy)
        self.vertex_mass.from_numpy(self.vertex_masses_npy)
        self.vertices_B = ti.Vector.field(
            3, float, self.num_vertices, needs_grad=False
        )
        self.vertices_B.from_numpy(
            self.node_coordinates.astype(np.float32)
        )
        self.tetrahedra = ti.Vector.field(4, int, self.num_tetrahedra, needs_grad=False)
        self.tetrahedra.from_numpy(self.tetrahedra_npy.astype(np.int32))
        self.contact_surface = ti.Vector.field(
            3, int, self.num_contact_surface_triangles, needs_grad=False
        )
        self.contact_surface.from_numpy(self.outer_surface_triangles.astype(np.int32))
        self.projection_2d_dome_surface_nodes_deformed = ti.Vector.field(
            2, float, self.dome_surface_node_tags.shape[0], needs_grad=True
        )
        self.projection_2d_dome_surface_nodes_undeformed = ti.Vector.field(
            2, float, self.dome_surface_node_tags.shape[0], needs_grad=False
        )
        self.clock_arms_node_idxs = ti.field(int, (2,), needs_grad=False)
        self.projection_2d_clock_arms = ti.Vector.field(
            2, float, (2,), needs_grad=False
        )

    @ti.kernel
    def project_surface_nodes_to_2d(
        self, surface_nodes: ti.types.ndarray(), output_projections: ti.types.ndarray()
    ):
        for i in range(surface_nodes.shape[0]):
            pos = ti.Vector(
                [surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2]]
            )
            proj = self.fisheye_model.project_3d_2d_ti(pos)
            output_projections[i, 0] = proj[0]
            output_projections[i, 1] = proj[1]

    def initialise_camera_model(self):
        self.marker_interpolation_knn_k = 5
        initial_camera_image = cv2.imread(
            SYSTEM_PARAMS.files.vitactip_photo_default_state
        )
        initial_marker_positions, _, _ = self.fisheye_model.get_marker_image(
            initial_camera_image
        )
        marker_visualization_image = initial_camera_image.copy()
        for marker_position in initial_marker_positions:
            marker_center = (
                int(round(marker_position[0])),
                int(round(marker_position[1])),
            )
            cv2.circle(
                marker_visualization_image,
                marker_center,
                radius=5,
                color=(0, 0, 255),
                thickness=2,
            )
        cv2.imwrite(
            SYSTEM_PARAMS.files.vitactip_photo_default_state_detected_markers,
            marker_visualization_image,
        )

        dome_surface_nodes = self.node_coordinates[self.dome_surface_node_tags_npy]
        max_z = np.max(dome_surface_nodes[:, 2])
        z_translation = (
            SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface
            - max_z
        )
        translation_vector = np.array([0, 0, z_translation])
        dome_surface_nodes += translation_vector

        surface_node_projections_2d = np.zeros(
            (len(dome_surface_nodes), 2), dtype=np.float32
        )
        self.project_surface_nodes_to_2d(
            dome_surface_nodes, surface_node_projections_2d
        )
        surface_node_visualization = initial_camera_image.copy()
        for projected_point in surface_node_projections_2d:
            point_center = (
                int(round(projected_point[0])),
                int(round(projected_point[1])),
            )
            cv2.circle(
                surface_node_visualization,
                point_center,
                radius=3,
                color=(0, 255, 0),
                thickness=2,
            )
        cv2.imwrite(
            SYSTEM_PARAMS.files.vitactip_photo_default_state_dome_surface_3d_vertices_projected_to_2d,
            surface_node_visualization,
        )
        marker_interpolation_indices = []
        marker_interpolation_weights = []
        interpolated_marker_positions_2d = []
        for marker_idx in range(initial_marker_positions.shape[0]):
            distances_to_projections = np.linalg.norm(
                initial_marker_positions[marker_idx, 0:2] - surface_node_projections_2d,
                axis=1,
            )
            neighbor_indices = np.argpartition(
                distances_to_projections, self.marker_interpolation_knn_k
            )
            k_nearest_indices = neighbor_indices[: self.marker_interpolation_knn_k]
            inverse_distances = 1 / distances_to_projections[k_nearest_indices]
            total_inverse_distance = np.sum(inverse_distances)
            interpolation_weights = inverse_distances / total_inverse_distance
            interpolated_position = np.matmul(
                surface_node_projections_2d[k_nearest_indices].T, interpolation_weights
            ).T
            interpolated_marker_positions_2d.append(interpolated_position)
            marker_interpolation_indices.append(k_nearest_indices)
            marker_interpolation_weights.append(interpolation_weights)
        interpolated_marker_positions_2d = np.array(interpolated_marker_positions_2d)
        marker_interpolation_indices = np.array(marker_interpolation_indices)
        marker_interpolation_weights = np.array(marker_interpolation_weights)
        surface_node_visualization = initial_camera_image.copy()
        for projected_point in interpolated_marker_positions_2d:
            point_center = (
                int(round(projected_point[0])),
                int(round(projected_point[1])),
            )
            cv2.circle(
                surface_node_visualization,
                point_center,
                radius=3,
                color=(0, 255, 0),
                thickness=2,
            )
        cv2.imwrite(
            SYSTEM_PARAMS.files.vitactip_photo_default_state_interpolated_marker_positions_2d,
            surface_node_visualization,
        )
        self.num_markers = len(interpolated_marker_positions_2d)
        self.deformed_markers = ti.Vector.field(
            2, float, self.num_markers, needs_grad=True
        )
        self.undeformed_markers = ti.Vector.field(
            2, float, self.num_markers, needs_grad=False
        )
        self.initial_markers_unused = ti.Vector.field(
            2, float, self.num_markers, needs_grad=False
        )
        self.initial_undeformed_vertices_after_applying_transformation_matrix = (
            ti.Vector.field(3, float, self.num_markers, needs_grad=False)
        )
        self.marker_interpolation_weights = ti.Vector.field(
            self.marker_interpolation_knn_k, float, self.num_markers, needs_grad=False
        )
        self.marker_interpolation_weights.from_numpy(
            marker_interpolation_weights.astype(np.float32)
        )
        self.marker_interpolation_indices = ti.Vector.field(
            self.marker_interpolation_knn_k, int, self.num_markers, needs_grad=False
        )
        self.marker_interpolation_indices.from_numpy(
            marker_interpolation_indices.astype(np.int32)
        )

    def set_up_physical_state_1(self):
        self.vertices_undeformed_A = ti.Vector.field(
            3,
            float,
            shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices),
            needs_grad=False,
        )
        self.vertices_E = ti.Vector.field(
            3,
            float,
            shape=(self.num_vertices),
            needs_grad=False,
        )
        self.vertices_B_testing = ti.Vector.field(
            3,
            float,
            shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices),
            needs_grad=False,
        )
        self.vertices_deformed_A = ti.Vector.field(
            3,
            float,
            shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices),
            needs_grad=True,
        )
        self.vertex_velocities = ti.Vector.field(
            3,
            float,
            shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices),
            needs_grad=True,
        )
        self.initial_deformation_gradient_inverse = ti.Matrix.field(
            3, 3, float, self.num_tetrahedra, needs_grad=False
        )
        self.contact_forces_on_vertices = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_sub_frames, self.num_vertices),
            needs_grad=True,
        )
        self.total_surface_force = ti.Vector.field(
            3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames), needs_grad=False
        )
        self.sensor_outward_normal = ti.Vector.field(3, float, (), needs_grad=False)
        self.translation_A = ti.Vector.field(3, ti.f32, shape=(), needs_grad=False)
        self.R_BA_quat = ti.Vector.field(4, ti.f32, shape=(), needs_grad=False)
        self.translation_CD = ti.Vector.field(3, ti.f32, shape=(), needs_grad=False)
        self.R_CD = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        self.R_BA = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        self.R_CA = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        self.R_BC = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        self.R_AB = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        self.T_BA = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_CA = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_BC = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_AB = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_CD = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_A = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_BE = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_EB = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.R_EB = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        T_BE_np = np.eye(4)
        T_BE_np[2, 3] = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface
        self.T_BE.from_numpy(T_BE_np)
        self.T_EB[None] = self.T_BE[None].inverse()
        
        self.vertex_control_velocities = ti.Vector.field(
            3, float, shape=(self.num_vertices), needs_grad=False
        )
        self.signed_distance_function = ti.field(
            dtype=ti.f32, shape=(), needs_grad=False
        )
        self.simulation_cache = dict()
        self.translation_CD = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.R_A = ti.Matrix.field(3, 3, dtype=float, shape=(), needs_grad=False)
        self.R_A_quat = ti.Vector.field(4, ti.f32, shape=(), needs_grad=False)
    
    @ti.kernel
    def set_up_physical_state_2(self):
        for i in range(3):
            for j in range(3):
                self.R_EB[None][i, j] = self.T_EB[None][i, j]

    def set_up_pose(self, pose):
        rotation_object = R.from_quat(pose[3:])
        initial_rotation_matrix = rotation_object.as_matrix()
        initial_transformation_matrix = np.eye(4)
        initial_transformation_matrix[0:3, 0:3] = initial_rotation_matrix
        initial_transformation_matrix[0, 3] = pose[0]
        initial_transformation_matrix[1, 3] = pose[1]
        initial_transformation_matrix[2, 3] = pose[2]
        self.R_BC.from_numpy(np.eye(3))
        self.R_CA.from_numpy(initial_rotation_matrix)
        self.R_BA[None] = self.R_CA[None] @ self.R_BC[None]
        self.R_AB[None] = self.R_BA[None].transpose()
        self.T_BC.from_numpy(np.eye(4))
        self.T_CA.from_numpy(initial_transformation_matrix)
        self.T_BA[None] = self.T_CA[None] @ self.T_BC[None]
        self.T_AB[None] = self.T_BA[None].inverse()
        self.T_CD.from_numpy(np.eye(4))
        self.sensor_outward_normal[None] = self.R_BA[None] @ ti.Vector([0.0, 0.0, 1.0])
        self.set_up_pose_helper()

    @ti.func
    def rotate_E_to_A(self, inhomogeneous_velocity_E):
        inhomogeneous_velocity_A = (
            self.R_BA[None] @ self.R_EB[None] @ inhomogeneous_velocity_E
        )
        return inhomogeneous_velocity_A

    @ti.kernel
    def debug_project_E_to_A(self, inhomogeneous_point_E: ti.template()):
        return self.project_E_to_A(inhomogeneous_point_E)

    @ti.func
    def project_A_to_E(self, inhomogeneous_point_A):
        homogeneous_point_A = ti.Vector(
            [
                inhomogeneous_point_A[0],
                inhomogeneous_point_A[1],
                inhomogeneous_point_A[2],
                1.0,
            ]
        )
        homogeneous_point_E = (
            self.T_BE[None] @ self.T_AB[None] @ homogeneous_point_A
        )
        inhomogeneous_point_E = ti.Vector(
            [
                homogeneous_point_E[0],
                homogeneous_point_E[1],
                homogeneous_point_E[2],
            ]
        )
        return inhomogeneous_point_E

    @ti.kernel
    def compute_vertices_E(self):
        for i in range(self.num_vertices):
            self.vertices_E[i] = self.project_A_to_E(self.vertices_undeformed_A[0, i])

    @ti.func
    def project_E_to_A(self, inhomogeneous_point_E):
        homogeneous_point_E = ti.Vector(
            [
                inhomogeneous_point_E[0],
                inhomogeneous_point_E[1],
                inhomogeneous_point_E[2],
                1.0,
            ]
        )
        homogeneous_point_A = (
            self.T_BA[None] @ self.T_EB[None] @ homogeneous_point_E
        )
        inhomogeneous_point_A = ti.Vector(
            [
                homogeneous_point_A[0],
                homogeneous_point_A[1],
                homogeneous_point_A[2],
            ]
        )
        return inhomogeneous_point_A

    @ti.func
    def project_A_point_2d(self, inhomogeneous_point_A):
        homogeneous_point_A = ti.Vector(
            [
                inhomogeneous_point_A[0],
                inhomogeneous_point_A[1],
                inhomogeneous_point_A[2],
                1.0,
            ]
        )
        homogeneous_point_E = (
            self.T_BE[None] @ self.T_AB[None] @ homogeneous_point_A
        )
        inhomogeneous_point_E = ti.Vector(
            [
                homogeneous_point_E[0],
                homogeneous_point_E[1],
                homogeneous_point_E[2],
            ]
        )
        projection_2d = self.fisheye_model.project_3d_2d_ti(
            inhomogeneous_point_E
        )
        return projection_2d

    @ti.kernel
    def mark_surface_nodes_in_contact(self, frame_idx: ti.i32):
        for i in range(self.dome_surface_node_tags.shape[0]):
            node_ix = self.dome_surface_node_tags[i]
            deformed_A = self.vertices_deformed_A[frame_idx, node_ix]
            if deformed_A[2] <= SYSTEM_PARAMS_COMPUTED.phantom_top_surface_z:
                self.dome_surface_node_contact_mask[i] = 1
            else:
                self.dome_surface_node_contact_mask[i] = 0

    @ti.kernel
    def extract_markers(self, frame_idx: ti.i32):
        for i in range(self.dome_surface_node_tags.shape[0]):
            node_ix = self.dome_surface_node_tags[i]
            deformed_A = self.vertices_deformed_A[frame_idx, node_ix]
            undeformed_A = self.vertices_undeformed_A[
                frame_idx, node_ix
            ]
            self.projection_2d_dome_surface_nodes_deformed[i] += (
                self.project_A_point_2d(deformed_A)
            )
            self.projection_2d_dome_surface_nodes_undeformed[i] = (
                self.project_A_point_2d(undeformed_A)
            )
        for i in range(self.num_markers):
            nearest_surface_indices = self.marker_interpolation_indices[i]
            interpolation_weights = self.marker_interpolation_weights[i]
            interpolated_deformed_pos_2d = ti.Vector([0.0, 0.0])
            interpolated_undeformed_pos_2d = ti.Vector([0.0, 0.0])
            for neighbor_idx in range(self.marker_interpolation_knn_k):
                interpolated_deformed_pos_2d += (
                    interpolation_weights[neighbor_idx]
                    * self.projection_2d_dome_surface_nodes_deformed[
                        nearest_surface_indices[neighbor_idx]
                    ]
                )
                interpolated_undeformed_pos_2d += (
                    interpolation_weights[neighbor_idx]
                    * self.projection_2d_dome_surface_nodes_undeformed[
                        nearest_surface_indices[neighbor_idx]
                    ]
                )
            self.deformed_markers[i] += interpolated_deformed_pos_2d
            self.undeformed_markers[i] = interpolated_undeformed_pos_2d

    @ti.kernel
    def extract_clock_arm_2d_projections(self, frame_idx: ti.i32):
        for i in range(self.clock_arms_node_idxs.shape[0]):
            node_idx = self.clock_arms_node_idxs[i]
            vertex = self.vertices_undeformed_A[frame_idx, node_idx]
            self.projection_2d_clock_arms[i] = self.project_A_point_2d(vertex)

    @ti.kernel
    def copy_markers_to_initial_markers_for_drift_correction_unused(self):
        return
        for marker_idx in range(self.num_markers):
            self.initial_markers_unused[marker_idx] = self.undeformed_markers[
                marker_idx
            ]

    def get_keypoint_idxs(self):
        self.compute_vertices_E()
        points = self.vertices_E.to_numpy()
        
        tip_z = np.max(points[:, 2])
        max_z = tip_z - SYSTEM_PARAMS.geometry.clock_arms_z_min_offset
        min_z = tip_z - SYSTEM_PARAMS.geometry.clock_arms_z_max_offset
        valid_mask = (points[:, 2] <= max_z) & (points[:, 2] >= min_z)
        filtered_points = points[valid_mask]
        filtered_indices = np.where(valid_mask)[0]
        
        idx_max_x = np.argmax(filtered_points[:, 0])
        self.clock_arms_node_idxs[0] = filtered_indices[idx_max_x]
        idx_min_y = np.argmax(filtered_points[:, 1])
        self.clock_arms_node_idxs[1] = filtered_indices[idx_min_y]

    def save_predicted_markers_to_image(self):
        initial_camera_image = cv2.imread(
            SYSTEM_PARAMS.files.vitactip_photo_default_state
        )
        surface_node_visualization = initial_camera_image.copy()
        with open(SYSTEM_PARAMS.files.initial_vertex_positions_undeformed, "wb") as f:
            pickle.dump(self.vertices_B_testing.to_numpy()[0], f)
        for projected_point in self.undeformed_markers.to_numpy():
            point_center = (
                int(round(projected_point[0])),
                int(round(projected_point[1])),
            )
            cv2.circle(
                surface_node_visualization,
                point_center,
                radius=3,
                color=(0, 255, 0),
                thickness=2,
            )
        cv2.imwrite(
            SYSTEM_PARAMS.files.vitactip_photo_default_state_predicted_markers,
            surface_node_visualization,
        )

    @ti.kernel
    def test_mapping_from_global_space_to_camera_space(self):
        for i in range(self.vertices_undeformed_A.shape[1]):
            initial_vertex_pos = self.vertices_undeformed_A[0, i]
            homogeneous_initial_pos = ti.Vector(
                [
                    initial_vertex_pos[0],
                    initial_vertex_pos[1],
                    initial_vertex_pos[2],
                    1.0,
                ]
            )
            transformed_initial_pos = self.T_AB[None] @ homogeneous_initial_pos
            camera_space_initial_pos = ti.Vector(
                [
                    transformed_initial_pos[0],
                    transformed_initial_pos[1],
                    transformed_initial_pos[2],
                ]
            )
            self.vertices_B_testing[0, i] = camera_space_initial_pos

    @ti.kernel
    def set_vel(self, f: ti.i32):
        for p in range(self.num_vertices):
            self.vertex_velocities[f, p] = self.vertex_control_velocities[p]

    @ti.kernel
    def set_pose_control_1(self):
        self.R_CD[None] = self.R_CA[None].transpose() @ self.R_A[None] @ self.R_CA[None]
        self.translation_CD[None] = (
            self.R_CA[None].transpose() @ self.translation_A[None]
        )
        self.translation_CD[None] = (
            self.translation_CD[None]
            * self.dt[None]
            * (SYSTEM_PARAMS.contact.num_sub_frames - 1)
        )
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

        self.sensor_outward_normal[None] = self.R_A[None] @ self.sensor_outward_normal[None]

    @ti.kernel
    def set_control_vel(self, f: ti.i32):
        for i in range(self.num_vertices):
            current_vertex_positions_undeformed = self.vertices_undeformed_A[f, i]
            target_vertex_positions_undeformed = self.T_A[None] @ ti.Vector(
                [
                    current_vertex_positions_undeformed[0],
                    current_vertex_positions_undeformed[1],
                    current_vertex_positions_undeformed[2],
                    1.0,
                ]
            )
            self.vertex_control_velocities[i][0] = (
                target_vertex_positions_undeformed[0]
                - current_vertex_positions_undeformed[0]
            ) / (self.dt[None] * (SYSTEM_PARAMS.contact.num_sub_frames - 1))
            self.vertex_control_velocities[i][1] = (
                target_vertex_positions_undeformed[1]
                - current_vertex_positions_undeformed[1]
            ) / (self.dt[None] * (SYSTEM_PARAMS.contact.num_sub_frames - 1))
            self.vertex_control_velocities[i][2] = (
                target_vertex_positions_undeformed[2]
                - current_vertex_positions_undeformed[2]
            ) / (self.dt[None] * (SYSTEM_PARAMS.contact.num_sub_frames - 1))

    @ti.kernel
    def get_external_force(self, f: ti.i32):
        for k in range(self.num_contact_surface_triangles):
            a, b, c = self.contact_surface[k]
            self.total_surface_force[f] += (
                1 / 3 * self.contact_forces_on_vertices[f, a] * self.dx
            )
            self.total_surface_force[f] += (
                1 / 3 * self.contact_forces_on_vertices[f, b] * self.dx
            )
            self.total_surface_force[f] += (
                1 / 3 * self.contact_forces_on_vertices[f, c] * self.dx
            )

    def compute_current_orientation(self):
        rot_mat = self.R_BA.to_numpy()
        rotation_object = R.from_matrix(rot_mat)
        self.R_BA_quat.from_numpy(rotation_object.as_quat())

    @ti.kernel
    def set_up_pose_helper(self):
        for idx in range(self.num_vertices):
            vertices_B = self.vertices_B[idx]
            vertices_A = self.T_BA[None] @ ti.Vector(
                [
                    vertices_B[0],
                    vertices_B[1],
                    vertices_B[2],
                    1.0,
                ]
            )
            self.vertices_deformed_A[0, idx] = ti.Vector(
                [
                    vertices_A[0],
                    vertices_A[1],
                    vertices_A[2],
                ]
            )
            self.vertices_undeformed_A[0, idx] = self.vertices_deformed_A[0, idx]
        for i in range(self.num_tetrahedra):
            ia, ib, ic, id = self.tetrahedra[i]
            a, b, c, d = (
                self.vertices_deformed_A[0, ia],
                self.vertices_deformed_A[0, ib],
                self.vertices_deformed_A[0, ic],
                self.vertices_deformed_A[0, id],
            )
            deformation_gradient = ti.Matrix.cols([a - d, b - d, c - d])
            self.initial_deformation_gradient_inverse[i] = deformation_gradient.inverse()

    @ti.func
    def find_closest(self, grid_p, f):
        cur_min_offset = SYSTEM_PARAMS.vitactip.collision_search_distance
        cur_min_idx = -1
        for k in range(self.num_contact_surface_triangles):
            a, b, c = self.contact_surface[k]
            p_1 = self.vertices_deformed_A[f, a]
            p_2 = self.vertices_deformed_A[f, b]
            p_3 = self.vertices_deformed_A[f, c]
            p_c = 1 / 3 * (p_1 + p_2 + p_3)
            offset_p = (p_c - grid_p).norm(SYSTEM_PARAMS.contact.norm_eps)
            if offset_p < cur_min_offset:
                cur_min_offset = offset_p
                cur_min_idx = k
        return cur_min_idx

    @ti.func
    def find_sdf(
        self,
        point_position,
        point_velocity,
        triangle_index,
        frame,
        collision_offset=0.0,
    ):
        vertex1_idx, vertex2_idx, vertex3_idx = self.contact_surface[triangle_index]
        vertex1_pos = self.vertices_deformed_A[frame, vertex1_idx]
        vertex2_pos = self.vertices_deformed_A[frame, vertex2_idx]
        vertex3_pos = self.vertices_deformed_A[frame, vertex3_idx]
        triangle_normal = ti.math.cross(
            vertex2_pos - vertex1_pos, vertex3_pos - vertex1_pos
        )
        triangle_normal = triangle_normal.normalized(SYSTEM_PARAMS.contact.norm_eps)
        normal_direction = ti.math.sign(
            triangle_normal.dot(self.sensor_outward_normal[None])
        )
        triangle_normal = normal_direction * triangle_normal
        point_to_vertex1 = point_position - vertex1_pos
        signed_distance = point_to_vertex1.dot(triangle_normal)
        point_projected = point_position - signed_distance * triangle_normal
        surface_normal = -1 * triangle_normal
        edge1 = vertex3_pos - vertex1_pos
        edge2 = vertex2_pos - vertex1_pos
        point_projected_rel = point_projected - vertex1_pos
        dot_edge1_edge1 = edge1.dot(edge1)
        dot_edge1_edge2 = edge1.dot(edge2)
        dot_edge1_point = edge1.dot(point_projected_rel)
        dot_edge2_edge2 = edge2.dot(edge2)
        dot_edge2_point = edge2.dot(point_projected_rel)
        inv_denominator = 1 / (
            dot_edge1_edge1 * dot_edge2_edge2 - dot_edge1_edge2 * dot_edge1_edge2
        )
        barycentric_u = (
            dot_edge2_edge2 * dot_edge1_point - dot_edge1_edge2 * dot_edge2_point
        ) * inv_denominator
        barycentric_v = (
            dot_edge1_edge1 * dot_edge2_point - dot_edge1_edge2 * dot_edge1_point
        ) * inv_denominator
        signed_distance -= collision_offset
        relative_velocity = point_velocity - 1 / 3 * (
            self.vertex_velocities[frame, vertex1_idx]
            + self.vertex_velocities[frame, vertex2_idx]
            + self.vertex_velocities[frame, vertex3_idx]
        )
        is_contact = (
            signed_distance < 0
            and barycentric_u >= 0
            and barycentric_v >= 0.0
            and (barycentric_u + barycentric_v <= 1)
        )
        return signed_distance, surface_normal, relative_velocity, is_contact

    @ti.func
    def update_contact_force(self, triangle_index, contact_force, frame):
        vertex1_idx, vertex2_idx, vertex3_idx = self.contact_surface[triangle_index]
        self.contact_forces_on_vertices[frame, vertex1_idx] += 1 / 3 * contact_force
        self.contact_forces_on_vertices[frame, vertex2_idx] += 1 / 3 * contact_force
        self.contact_forces_on_vertices[frame, vertex3_idx] += 1 / 3 * contact_force

    @ti.kernel
    def update_internal_forces(self, frame: ti.i32):
        for tetra_idx in range(self.num_tetrahedra):
            vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx = self.tetrahedra[
                tetra_idx
            ]
            vertex1_pos = self.vertices_deformed_A[frame, vertex1_idx]
            vertex2_pos = self.vertices_deformed_A[frame, vertex2_idx]
            vertex3_pos = self.vertices_deformed_A[frame, vertex3_idx]
            vertex4_pos = self.vertices_deformed_A[frame, vertex4_idx]
            vertex1_vel = self.vertex_velocities[frame, vertex1_idx]
            vertex2_vel = self.vertex_velocities[frame, vertex2_idx]
            vertex3_vel = self.vertex_velocities[frame, vertex3_idx]
            vertex4_vel = self.vertex_velocities[frame, vertex4_idx]
            deformation_matrix = ti.Matrix.cols(
                [
                    vertex1_pos - vertex4_pos,
                    vertex2_pos - vertex4_pos,
                    vertex3_pos - vertex4_pos,
                ]
            )
            tetra_volume = ti.abs(deformation_matrix.determinant()) / 6
            deformation_gradient = (
                deformation_matrix @ self.initial_deformation_gradient_inverse[tetra_idx]
            )

            # F_T = deformation_gradient.inverse().transpose()
            # J = deformation_gradient.determinant()
            # J = ti.max(0.2, deformation_gradient.determinant())
            # log_J_i = ti.log(J)
            # stress_tensor = self.mu[None] * (deformation_gradient -  F_T) + self.lam[None] * log_J_i * F_T

            mu = self.mu[None]
            lam = self.lam[None]
            jacobian = deformation_gradient.determinant()
            first_invariant = (
                deformation_gradient.transpose() @ deformation_gradient
            ).trace()
            dJ_dF0 = deformation_gradient[:, 1].cross(deformation_gradient[:, 2])
            dJ_dF1 = deformation_gradient[:, 2].cross(deformation_gradient[:, 0])
            dJ_dF2 = deformation_gradient[:, 0].cross(deformation_gradient[:, 1])
            jacobian_derivative = ti.Matrix.cols([dJ_dF0, dJ_dF1, dJ_dF2])
            alpha = 1 + 0.75 * mu / lam
            stress_tensor = (
                mu * (1 - 1 / (first_invariant + 1)) * deformation_gradient
                + lam * (jacobian - alpha) * jacobian_derivative
            )
            elastic_force_matrix = (
                -tetra_volume
                * stress_tensor
                @ self.initial_deformation_gradient_inverse[tetra_idx].transpose()
            )
            relative_vel1 = vertex1_vel - vertex4_vel
            relative_vel2 = vertex2_vel - vertex4_vel
            relative_vel3 = vertex3_vel - vertex4_vel
            damping_force_matrix = -self.rayleigh_damping_alpha[None] * ti.Matrix.cols(
                [relative_vel1, relative_vel2, relative_vel3]
            )
            damping_force_matrix += (
                -self.rayleigh_damping_beta[None] * elastic_force_matrix
            )
            force_matrix = elastic_force_matrix + damping_force_matrix
            vertex_indices = ti.Vector(
                [vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx]
            )
            for k in ti.static(range(3)):
                vertex_force = ti.Vector([force_matrix[j, k] for j in range(3)])
                self.vertex_velocities[frame, vertex_indices[k]] += (
                    self.dt[None]
                    * vertex_force
                    / self.vertex_mass[vertex_indices[k]]
                )
                self.vertex_velocities[frame, vertex_indices[3]] += (
                    -self.dt[None]
                    * vertex_force
                    / self.vertex_mass[vertex_indices[3]]
                )

    @ti.kernel
    def update_external_forces(self, frame: ti.i32):
        for vertex_idx in range(self.num_vertices):
            updated_velocity = ti.Vector([0.0, 0.0, 0.0])
            updated_velocity += self.vertex_velocities[frame, vertex_idx]
            updated_velocity += (
                self.dt[None]
                * self.contact_forces_on_vertices[frame, vertex_idx]
                / self.vertex_mass[vertex_idx]
            )
            is_fixed_layer = self.is_fixed_layer[vertex_idx] == 1
            if is_fixed_layer:
                updated_velocity = self.vertex_control_velocities[vertex_idx]
            self.vertex_velocities[frame + 1, vertex_idx] += updated_velocity
            self.vertices_deformed_A[frame + 1, vertex_idx] += self.vertices_deformed_A[
                frame, vertex_idx
            ]
            self.vertices_deformed_A[frame + 1, vertex_idx] += (
                self.dt[None] * updated_velocity
            )
            self.vertices_undeformed_A[frame + 1, vertex_idx] = (
                self.vertices_undeformed_A[frame, vertex_idx]
                + self.dt[None] * self.vertex_control_velocities[vertex_idx]
            )
    
    @ti.func
    def reset_state(self):
        self.youngs_modulus.fill(0.0)
        self.poissons_ratio.fill(0.0)
        self.mu.fill(0.0)
        self.lam.fill(0.0)
        self.contact_forces_on_vertices.fill(0.0)
        self.total_surface_force.fill(0.0)
        self.deformed_markers.fill(0.0)
        self.projection_2d_dome_surface_nodes_deformed.fill(0.0)
        self.dome_surface_node_contact_mask.fill(0)
        for i in range(1, self.vertices_deformed_A.shape[0]):
            for j in range(self.vertices_deformed_A.shape[1]):
                self.vertices_deformed_A[i, j] = ti.Vector([0.0, 0.0, 0.0])
                self.vertex_velocities[i, j] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def clear_grad(self):
        self.deformed_markers.grad.fill(0.0)
        self.vertices_deformed_A.grad.fill(0.0)
        self.vertex_velocities.grad.fill(0.0)
        self.contact_forces_on_vertices.grad.fill(0.0)
        self.mu.grad.fill(0.0)
        self.lam.grad.fill(0.0)
        self.youngs_modulus.grad.fill(0.0)
        self.poissons_ratio.grad.fill(0.0)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.num_vertices):
            self.vertices_deformed_A[target, p] = self.vertices_deformed_A[source, p]
            self.vertices_undeformed_A[target, p] = self.vertices_undeformed_A[
                source, p
            ]
            self.vertex_velocities[target, p] = self.vertex_velocities[
                source, p
            ]

    @ti.kernel
    def load_step_from_cache(
        self,
        f: ti.i32,
        cache_vertices_deformed_A: ti.types.ndarray(),
        cache_T_BA: ti.types.ndarray(),
        cache_vertices_undeformed_A: ti.types.ndarray(),
        cache_rot: ti.types.ndarray(),
        cache_R_CD: ti.types.ndarray(),
        cache_R_CA: ti.types.ndarray(),
        cache_R_A: ti.types.ndarray(),
        cache_translation_A: ti.types.ndarray(),
        cache_T_CD: ti.types.ndarray(),
        cache_T_CA: ti.types.ndarray(),
        cache_T_BC: ti.types.ndarray(),
        cache_T_AB: ti.types.ndarray(),
        cache_R_BC: ti.types.ndarray(),
        cache_R_AB: ti.types.ndarray(),
        cache_T_A: ti.types.ndarray(),
        cache_vertex_velocities: ti.types.ndarray(),
        cache_sensor_outward_normal: ti.types.ndarray(),
    ):
        for j in range(4):
            for k in range(4):
                self.T_BA[None][j, k] = cache_T_BA[j, k]
                self.T_CD[None][j, k] = cache_T_CD[j, k]
                self.T_CA[None][j, k] = cache_T_CA[j, k]
                self.T_BC[None][j, k] = cache_T_BC[j, k]
                self.T_AB[None][j, k] = cache_T_AB[j, k]
                self.T_A[None][j, k] = cache_T_A[j, k]
        for j in range(3):
            for k in range(3):
                self.R_BA[None][j, k] = cache_rot[j, k]
                self.R_CD[None][j, k] = cache_R_CD[j, k]
                self.R_CA[None][j, k] = cache_R_CA[j, k]
                self.R_A[None][j, k] = cache_R_A[j, k]
                self.R_BC[None][j, k] = cache_R_BC[j, k]
                self.R_AB[None][j, k] = cache_R_AB[j, k]
            self.translation_A[None][j] = cache_translation_A[j]
            self.sensor_outward_normal[None][j] = cache_sensor_outward_normal[j]
        for p in range(self.num_vertices):
            for i in ti.static(range(3)):
                self.vertices_deformed_A[f, p][i] = cache_vertices_deformed_A[p, i]
                self.vertices_undeformed_A[f, p][i] = cache_vertices_undeformed_A[p, i]
                self.vertex_velocities[f, p][i] = cache_vertex_velocities[p, i]

    @ti.kernel
    def add_step_to_cache(
        self,
        f: ti.i32,
        cache_vertices_deformed_A: ti.types.ndarray(),
        cache_T_BA: ti.types.ndarray(),
        cache_vertices_undeformed_A: ti.types.ndarray(),
        cache_rot: ti.types.ndarray(),
        cache_R_CD: ti.types.ndarray(),
        cache_R_CA: ti.types.ndarray(),
        cache_R_A: ti.types.ndarray(),
        cache_translation_A: ti.types.ndarray(),
        cache_T_CD: ti.types.ndarray(),
        cache_T_CA: ti.types.ndarray(),
        cache_T_BC: ti.types.ndarray(),
        cache_T_AB: ti.types.ndarray(),
        cache_R_BC: ti.types.ndarray(),
        cache_R_AB: ti.types.ndarray(),
        cache_T_A: ti.types.ndarray(),
        cache_vertex_velocities: ti.types.ndarray(),
        cache_sensor_outward_normal: ti.types.ndarray(),
    ):
        for j in range(4):
            for k in range(4):
                cache_T_BA[j, k] = self.T_BA[None][j, k]
                cache_T_CD[j, k] = self.T_CD[None][j, k]
                cache_T_CA[j, k] = self.T_CA[None][j, k]
                cache_T_BC[j, k] = self.T_BC[None][j, k]
                cache_T_AB[j, k] = self.T_AB[None][j, k]
                cache_T_A[j, k] = self.T_A[None][j, k]
        for j in range(3):
            for k in range(3):
                cache_rot[j, k] = self.R_BA[None][j, k]
                cache_R_CD[j, k] = self.R_CD[None][j, k]
                cache_R_CA[j, k] = self.R_CA[None][j, k]
                cache_R_A[j, k] = self.R_A[None][j, k]
                cache_R_BC[j, k] = self.R_BC[None][j, k]
                cache_R_AB[j, k] = self.R_AB[None][j, k]
            cache_translation_A[j] = self.translation_A[None][j]
            cache_sensor_outward_normal[j] = self.sensor_outward_normal[None][j]
        for p in range(self.num_vertices):
            for i in ti.static(range(3)):
                cache_vertices_deformed_A[p, i] = self.vertices_deformed_A[f, p][i]
                cache_vertices_undeformed_A[p, i] = self.vertices_undeformed_A[f, p][i]
                cache_vertex_velocities[p, i] = self.vertex_velocities[f, p][i]

    def memory_to_cache(self, t):
        cur_step_name = f"{t:06d}"
        device = "cpu"
        self.simulation_cache[cur_step_name] = dict()
        self.simulation_cache[cur_step_name]["vertices_deformed_A"] = torch.zeros(
            (self.num_vertices, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["T_BA"] = torch.zeros(
            (4, 4), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["rot_h"] = torch.zeros(
            (3, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["vertices_undeformed_A"] = torch.zeros(
            (self.num_vertices, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["R_CD"] = torch.zeros(
            (3, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["R_CA"] = torch.zeros(
            (3, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["R_A"] = torch.zeros(
            (3, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["translation_A"] = torch.zeros(
            3, dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["T_CD"] = torch.zeros(
            (4, 4), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["T_CA"] = torch.zeros(
            (4, 4), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["T_BC"] = torch.zeros(
            (4, 4), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["T_AB"] = torch.zeros(
            (4, 4), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["R_BC"] = torch.zeros(
            (3, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["R_AB"] = torch.zeros(
            (3, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["T_A"] = torch.zeros(
            (4, 4), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["vertex_velocities"] = torch.zeros(
            (self.num_vertices, 3), dtype=TC_TYPE, device=device
        )
        self.simulation_cache[cur_step_name]["sensor_outward_normal"] = torch.zeros(
            3, dtype=TC_TYPE, device=device
        )
        self.add_step_to_cache(
            0,
            self.simulation_cache[cur_step_name]["vertices_deformed_A"],
            self.simulation_cache[cur_step_name]["T_BA"],
            self.simulation_cache[cur_step_name]["vertices_undeformed_A"],
            self.simulation_cache[cur_step_name]["rot_h"],
            self.simulation_cache[cur_step_name]["R_CD"],
            self.simulation_cache[cur_step_name]["R_CA"],
            self.simulation_cache[cur_step_name]["R_A"],
            self.simulation_cache[cur_step_name]["translation_A"],
            self.simulation_cache[cur_step_name]["T_CD"],
            self.simulation_cache[cur_step_name]["T_CA"],
            self.simulation_cache[cur_step_name]["T_BC"],
            self.simulation_cache[cur_step_name]["T_AB"],
            self.simulation_cache[cur_step_name]["R_BC"],
            self.simulation_cache[cur_step_name]["R_AB"],
            self.simulation_cache[cur_step_name]["T_A"],
            self.simulation_cache[cur_step_name]["vertex_velocities"],
            self.simulation_cache[cur_step_name]["sensor_outward_normal"],
        )
        self.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames - 1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f"{t:06d}"
        self.load_step_from_cache(
            0,
            self.simulation_cache[cur_step_name]["vertices_deformed_A"],
            self.simulation_cache[cur_step_name]["T_BA"],
            self.simulation_cache[cur_step_name]["vertices_undeformed_A"],
            self.simulation_cache[cur_step_name]["rot_h"],
            self.simulation_cache[cur_step_name]["R_CD"],
            self.simulation_cache[cur_step_name]["R_CA"],
            self.simulation_cache[cur_step_name]["R_A"],
            self.simulation_cache[cur_step_name]["translation_A"],
            self.simulation_cache[cur_step_name]["T_CD"],
            self.simulation_cache[cur_step_name]["T_CA"],
            self.simulation_cache[cur_step_name]["T_BC"],
            self.simulation_cache[cur_step_name]["T_AB"],
            self.simulation_cache[cur_step_name]["R_BC"],
            self.simulation_cache[cur_step_name]["R_AB"],
            self.simulation_cache[cur_step_name]["T_A"],
            self.simulation_cache[cur_step_name]["vertex_velocities"],
            self.simulation_cache[cur_step_name]["sensor_outward_normal"],
        )

    def get_keypoint_indices(self, f: ti.i32):
        positions = self.vertices_deformed_A.to_numpy()[f]
        z_coords = positions[:, 2]
        point_a_idx = int(np.argmin(z_coords))
        max_z = float(np.max(z_coords))
        z_mask = z_coords >= (
            max_z - SYSTEM_PARAMS.vitactip.keypoint_search_z_threshold
        )
        x_coords = positions[:, 0]
        x_coords_filtered = x_coords.copy()
        x_coords_filtered[~z_mask] = float("-inf")
        point_b_idx = int(np.argmax(x_coords_filtered))
        y_coords = positions[:, 1]
        y_coords_filtered = y_coords.copy()
        y_coords_filtered[~z_mask] = float("-inf")
        point_c_idx = int(np.argmax(y_coords_filtered))
        return np.array([point_a_idx, point_b_idx, point_c_idx])

    def get_keypoint_coordinates(
        self, f: int, keypoint_indices: np.ndarray
    ) -> np.ndarray:
        positions = self.vertices_undeformed_A.to_numpy()[f]
        coordinates = positions[keypoint_indices]
        return coordinates
