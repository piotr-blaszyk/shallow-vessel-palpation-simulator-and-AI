"""
a class to describe sensor elastomer with FEM
"""

import taichi as ti
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import json

from difftactile.sensor_model.fisheye_model import * 

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class ViTacTip:
    def __init__(self):
        self.set_up_system_params()
        self.init_mesh()
        self.set_up_physical_state()
    
    def set_up_system_params(self):
        with open('../tasks/system-params.json', 'r') as f:
            self.params = json.load(f)
        self.vitactip_params = self.params['vitactip']
        self.contact_params = self.params['contact']
        self.geometry_params = self.params['geometry']

        self.max_num_materials = self.vitactip_params['maximum_number_of_materials']
        self.number_of_materials = ti.field(dtype=int, shape=(), needs_grad=False)
        self.number_of_materials[None] = self.vitactip_params['number_of_materials']
        self.fixed_layer_distance_from_bottom = self.vitactip_params['fixed_layer_distance_from_bottom']
        self.keypoint_search_z_threshold = self.vitactip_params['keypoint_search_z_threshold']
        self.collision_search_distance = self.vitactip_params['collision_search_distance']
        self.distance_from_camera_lens_to_outer_shell_surface = self.geometry_params['distance_from_camera_lens_to_outer_shell_surface']
        
        self.sub_steps = self.contact_params['num_sub_frames']
        self.dt = self.contact_params['dt']
        self.norm_eps = self.contact_params['norm_eps']

        # Material parameters for all materials
        self.mass_density = ti.field(dtype=ti.f32, shape=(self.number_of_materials[None],), needs_grad=False)
        self.youngs_modulus = ti.field(dtype=ti.f32, shape=(self.number_of_materials[None],), needs_grad=False)
        self.poissons_ratio = ti.field(dtype=ti.f32, shape=(self.number_of_materials[None],), needs_grad=False)
        self.mu = ti.field(dtype=ti.f32, shape=(self.number_of_materials[None],), needs_grad=False)
        self.lam = ti.field(dtype=ti.f32, shape=(self.number_of_materials[None],), needs_grad=False)

        if self.number_of_materials[None] == 1:
            self.mass_density[0] = self.vitactip_params['single_material']['density']
            self.youngs_modulus[0] = self.vitactip_params['single_material']['youngs_modulus']
            self.poissons_ratio[0] = self.vitactip_params['single_material']['poissons_ratio']

            self.mass_density[1] = self.vitactip_params['single_material']['density']
            self.youngs_modulus[1] = self.vitactip_params['single_material']['youngs_modulus']
            self.poissons_ratio[1] = self.vitactip_params['single_material']['poissons_ratio']
        else:
            # Initialize material parameters
            # Shell (Vytaflex 60) - material index 0
            self.mass_density[0] = self.vitactip_params['shell']['density']
            self.youngs_modulus[0] = self.vitactip_params['shell']['youngs_modulus']
            self.poissons_ratio[0] = self.vitactip_params['shell']['poissons_ratio']
            
            # Gel (RTV27905) - material index 1
            self.mass_density[1] = self.vitactip_params['gel']['density']
            self.youngs_modulus[1] = self.vitactip_params['gel']['youngs_modulus']
            self.poissons_ratio[1] = self.vitactip_params['gel']['poissons_ratio']
        
        # Compute Lamé parameters for all materials
        for i in range(self.max_num_materials):
            self.mu[i] = self.youngs_modulus[i] / 2 / (1 + self.poissons_ratio[i])
            self.lam[i] = (self.youngs_modulus[i] * self.poissons_ratio[i] / 
                          ((1 + self.poissons_ratio[i]) * (1 - 2 * self.poissons_ratio[i])))
            
        # Add Rayleigh damping coefficients
        self.rayleigh_damping_alpha = ti.field(dtype=ti.f32, shape=(), needs_grad=False)  # mass damping coefficient
        self.rayleigh_damping_beta = ti.field(dtype=ti.f32, shape=(), needs_grad=False)   # stiffness damping coefficient
        
        # Set default values (these should be tuned based on your specific needs)
        self.rayleigh_damping_alpha[None] = self.vitactip_params['rayleigh_damping_alpha']  # typical values range from 0 to 0.1
        self.rayleigh_damping_beta[None] = self.vitactip_params['rayleigh_damping_beta']  # typical values range from 0.001 to 0.01

        # Add hourglass control parameters
        self.hourglass_enabled = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.hourglass_coefficient = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.hourglass_modulus_scale = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        
        # Set hourglass control parameters from config
        self.hourglass_enabled[None] = self.vitactip_params['hourglass_control']['enabled']
        self.hourglass_coefficient[None] = self.vitactip_params['hourglass_control']['coefficient']
        self.hourglass_modulus_scale[None] = self.vitactip_params['hourglass_control']['modulus_scale']

    def init_mesh(self):
        # Load mesh data from gmsh
        with open('../tasks/output/gmsh-mesh.pkl', 'rb') as f:
            mesh_data = pickle.load(f)
        
        # Unpack mesh data
        all_tetrahedra = mesh_data['all_tetrahedra']
        node_coordinates = mesh_data['node_coordinates'] / 1_000
        node_labels = mesh_data['node_labels']
        surface_triangles = mesh_data['surface_triangles']
        group_to_idx = mesh_data['group_to_idx']
        y_bottom = mesh_data['y_bottom'] / 1_000
        marker_node_tags = mesh_data['marker_node_tags']
        
        # Compute fixed layer nodes (nodes at the bottom)
        is_fixed_layer = np.abs(node_coordinates[:, 1] - y_bottom) < self.fixed_layer_distance_from_bottom  # Check if y-coordinate is at bottom
        
        # Append is_fixed_layer to node_labels
        node_labels = np.column_stack([node_labels, is_fixed_layer])
        
        # Compute element materials and masses
        element_materials = np.full(len(all_tetrahedra), fill_value=-1, dtype=np.int32)
        vertex_masses = np.zeros(len(node_coordinates), dtype=np.float32)
        
        # Compute element volumes and assign materials
        for i, tetra in enumerate(all_tetrahedra):
            if 0 in tetra:
                foo = 7
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
        
        max_y = np.max(node_coordinates[:, 1])
        y_translation = self.distance_from_camera_lens_to_outer_shell_surface - max_y
        translation_vector = np.array([0, y_translation, 0])
        node_coordinates = node_coordinates + translation_vector

        self.nodes = node_coordinates
        self.tetrahedra_npy = all_tetrahedra
        self.outer_surface_triangles = surface_triangles
        self.node_labels = node_labels
        self.element_materials_npy = element_materials
        self.vertex_masses_npy = vertex_masses
        self.marker_node_tags_np = marker_node_tags

        self.num_markers = self.marker_node_tags_np.shape[0]
        self.marker_node_tags = ti.field(shape=(self.num_markers,), dtype=int)
        self.marker_node_tags.from_numpy(self.marker_node_tags_np)
        
        self.is_fixed_layer = ti.field(int, len(self.nodes))
        is_fixed_layer_data = self.node_labels[:,-1].astype(np.int32)
        self.is_fixed_layer.from_numpy(is_fixed_layer_data)
        
        self.num_vertices = len(self.nodes)
        self.num_tetrahedra = len(self.tetrahedra_npy)
        self.num_contact_surface_triangles = len(self.outer_surface_triangles)

        self.element_materials = ti.field(dtype=ti.i32, shape=(self.num_tetrahedra,), needs_grad=False)
        self.vertex_mass = ti.field(dtype=ti.f32, shape=(self.num_vertices,), needs_grad=False)
        self.element_materials.from_numpy(self.element_materials_npy)
        self.vertex_mass.from_numpy(self.vertex_masses_npy)

        self.initial_vertex_positions = ti.Vector.field(3, float, self.num_vertices, needs_grad=False)
        self.initial_vertex_positions.from_numpy(self.nodes.astype(np.float32))

        self.deformed_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=False)
        self.undeformed_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=False)
        self.initial_undeformed_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=False)

        self.tetrahedra = ti.Vector.field(4, int, self.num_tetrahedra)
        self.tetrahedra.from_numpy(self.tetrahedra_npy.astype(np.int32))
        self.contact_surface = ti.Vector.field(3, int, self.num_contact_surface_triangles) # surface triangle mesh
        self.contact_surface.from_numpy(self.outer_surface_triangles.astype(np.int32))

        self.validate_markers_via_2d_projection()

    def set_up_physical_state(self):
        # vertex_positions_ideal: ideal/undeformed vertex positions in m
        self.vertex_positions_undeformed = ti.Vector.field(3, float, shape=(self.sub_steps, self.num_vertices), needs_grad=False)
        # vertex_positions_deformed: current deformed vertex positions in m
        self.vertex_positions_deformed = ti.Vector.field(3, float, shape=(self.sub_steps, self.num_vertices), needs_grad=False)
        # vertex_velocities: vertex velocities in m/s
        self.vertex_velocities = ti.Vector.field(3, float, shape=(self.sub_steps, self.num_vertices), needs_grad=False)

        # deformation_gradient_inverse: inverse of deformation gradient matrix (dimensionless)
        self.deformation_gradient_inverse = ti.Matrix.field(3, 3, float, self.num_tetrahedra, needs_grad=False)
        # element_potential_energy: potential energy of each tetrahedral element in J
        self.element_potential_energy = ti.field(float, self.num_tetrahedra, needs_grad=False)  # potential energy of each face (Neo-Hookean)

        # contact_forces_on_vertices: external contact forces applied to vertices in N
        self.contact_forces_on_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.num_vertices), needs_grad=False) # contact force between FEM node to the closest particle
        # surface_force_resultant: total surface force vector in N
        self.total_surface_force = ti.Vector.field(3, float, shape=(self.sub_steps), needs_grad=False) # surface aggreated 3-axis forces

        # contact model parameters (default)
        # sensor_outward_normal: outward normal direction vector (dimensionless)
        self.sensor_outward_normal = ti.Vector.field(3, float, (), needs_grad=False)

        ## control parameters
        # global_translational_velocity: global translational velocity in m/s
        self.global_translational_velocity = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)
        # global_angular_velocity_degrees: global angular velocity in degrees/s
        self.global_angular_velocity_degrees = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)

        # local_translational_velocity: local translational velocity in m/s
        self.local_translational_velocity = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)
        # local_angular_velocity: local angular velocity in degrees/s
        self.local_angular_velocity = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)

        # homogeneous_rotation_matrix: 3x3 rotation matrix (dimensionless)
        self.homogeneous_rotation_matrix = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)
        # world_rotation_matrix: world coordinate rotation matrix (dimensionless)
        self.world_rotation_matrix = ti.Matrix.field(3, 3, ti.f32, shape = ())
        # local_rotation_matrix: local coordinate rotation matrix (dimensionless)
        self.local_rotation_matrix = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)
        # inverse_rotation_matrix: inverse of rotation matrix (dimensionless)
        self.inverse_rotation_matrix = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)

        # homogeneous_transformation_matrix: 4x4 homogeneous transformation matrix (dimensionless)
        self.homogeneous_transformation_matrix = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False) ##
        # world_transformation_matrix: world coordinate transformation matrix (dimensionless)
        self.world_transformation_matrix = ti.Matrix.field(4, 4, ti.f32, shape = ()) ## ee to world
        # local_transformation_matrix: local coordinate transformation matrix (dimensionless)
        self.local_transformation_matrix = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False) ## ee1 -> ee2
        # inverse_transformation_matrix: inverse of transformation matrix (dimensionless)
        self.inverse_transformation_matrix = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False)
        # delta_transformation_matrix: incremental transformation matrix (dimensionless)
        self.delta_transformation_matrix = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False)
        # vertex_control_velocities: prescribed control velocities for vertices in m/s
        self.vertex_control_velocities = ti.Vector.field(3, float, shape = (self.num_vertices), needs_grad=False)
        # signed_distance_function: signed distance to surface in m
        self.signed_distance_function = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        # simulation_cache: cache for gradient computation (dimensionless)
        self.simulation_cache = dict() # for grad backward
        # first_initialization_flag: flag for first initialization (dimensionless)
        self.first_initialization_flag = ti.field(dtype=int, shape=(), needs_grad=False)
        self.first_initialization_flag[None] = 1

        # rotation_vector_degrees: rotation vector in degrees (dimensionless)
        self.rotation_vector_degrees = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        # translation_vector: translation vector in m
        self.translation_vector = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        # transformation_matrix: 4x4 transformation matrix (dimensionless)
        self.transformation_matrix = ti.Matrix.field(4, 4, dtype=float, shape=(), needs_grad=False)
        # rotation_matrix: 3x3 rotation matrix (dimensionless)
        self.rotation_matrix = ti.Matrix.field(3, 3, dtype=float, shape=(), needs_grad=False)

    def set_up_pose(self, rot_x, rot_y, rot_z, t_dx, t_dy, t_dz):
        # rotation_quaternion: rotation quaternion from Euler angles (dimensionless)
        rotation_object = R.from_rotvec(np.deg2rad([rot_x, rot_y, rot_z]))
        # initial_rotation_matrix: 3x3 rotation matrix (dimensionless)
        initial_rotation_matrix = rotation_object.as_matrix()
        # initial_transformation_matrix: 4x4 homogeneous transformation matrix (dimensionless)
        initial_transformation_matrix = np.eye(4)
        initial_transformation_matrix[0:3,0:3] = initial_rotation_matrix
        initial_transformation_matrix[0,3] = t_dx; initial_transformation_matrix[1,3] = t_dy; initial_transformation_matrix[2,3] = t_dz

        self.local_rotation_matrix[None] = np.eye(3)
        self.world_rotation_matrix[None] = initial_rotation_matrix
        self.homogeneous_rotation_matrix[None] = self.world_rotation_matrix[None] @ self.local_rotation_matrix[None]
        self.inverse_rotation_matrix[None] = self.homogeneous_rotation_matrix[None].inverse()

        self.local_transformation_matrix[None] = np.eye(4)
        self.world_transformation_matrix[None] = initial_transformation_matrix
        self.homogeneous_transformation_matrix[None] = self.world_transformation_matrix[None] @ self.local_transformation_matrix[None]
        self.inverse_transformation_matrix[None] = self.local_transformation_matrix[None].inverse() @ self.world_transformation_matrix[None].inverse()

        if False and self.first[None]:
            print("\nInitial rotation matrix (init_rot):")
            print(initial_rotation_matrix)
            print("\nInitial transformation matrix (trans_mat):")
            print(initial_transformation_matrix)
            print("\nLocal rotation matrix (rot_local):")
            print(self.local_rotation_matrix[None].to_numpy())
            print("\nWorld rotation matrix (rot_world):")
            print(self.world_rotation_matrix[None].to_numpy())
            print("\nHomogeneous rotation matrix (rot_h):")
            print(self.homogeneous_rotation_matrix[None].to_numpy())
            print("\nInverse rotation matrix (inv_rot):")
            print(self.inverse_rotation_matrix[None].to_numpy())
            print("\nLocal transformation matrix (trans_local):")
            print(self.local_transformation_matrix[None].to_numpy())
            print("\nWorld transformation matrix (trans_world):")
            print(self.world_transformation_matrix[None].to_numpy())
            print("\nHomogeneous transformation matrix (trans_h):")
            print(self.homogeneous_transformation_matrix[None].to_numpy())
            print("\nInverse transformation matrix (inv_trans_h):")
            print(self.inverse_transformation_matrix[None].to_numpy())
            print()
        
        self.first_initialization_flag[None] = 0
        self.set_up_pose_helper()

    def validate_markers_via_2d_projection(self):
        marker_points_3d = self.nodes[self.marker_node_tags_np]
        camera_3D_points = np.array([marker_points_3d[:,0], marker_points_3d[:,2], marker_points_3d[:,1]]).T
        marker_points_2d = project_points_to_pix(camera_3D_points)

        img = cv2.imread('./init.png')
        if img is None:
            print('Error: Could not load ./init.png')
            return
        for pt in marker_points_2d:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (x, y), 4, (0,0,255), -1)  # Red dot
        cv2.imwrite('output/init-3d-markers.png', img)

    @ti.kernel
    def extract_initial_markers(self, f:ti.i32):
        for i in range(self.num_markers):
            node_ix = self.marker_node_tags[i]
            init_pos = self.vertex_positions_undeformed[f, node_ix]
            hom_init_pos = ti.Vector([init_pos[0], init_pos[1], init_pos[2], 1.0])
            inv_init_pos = self.inverse_transformation_matrix[None] @ hom_init_pos
            cam_init_pos = ti.Vector([inv_init_pos[0], inv_init_pos[2], inv_init_pos[1]])
            cam_init_loc = project_3d_2d(cam_init_pos)
            self.initial_undeformed_markers[i] = cam_init_loc
            if False:
                print('extract_initial_markers')
                print(f'node_ix: {node_ix}')
                print(f'init_pos: {init_pos}')
                print(f'hom_init_pos: {hom_init_pos}')
                print(f'inv_init_pos: {inv_init_pos}')
                print(f'cam_init_pos: {cam_init_pos}')
                print(f'cam_init_loc: {cam_init_loc}')
                print(0)

    @ti.kernel
    def extract_markers(self, f:ti.i32):
        for i in range(self.num_markers):
            node_ix = self.marker_node_tags[i]
            pos = self.vertex_positions_deformed[f, node_ix]
            init_pos = self.vertex_positions_undeformed[f, node_ix]
            hom_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0])
            hom_init_pos = ti.Vector([init_pos[0], init_pos[1], init_pos[2], 1.0])
            inv_pos = self.inverse_transformation_matrix[None] @ hom_pos
            inv_init_pos = self.inverse_transformation_matrix[None] @ hom_init_pos
            cam_pos = ti.Vector([inv_pos[0], inv_pos[2], inv_pos[1]])
            cam_init_pos = ti.Vector([inv_init_pos[0], inv_init_pos[2], inv_init_pos[1]])
            cam_loc = project_3d_2d(cam_pos)
            cam_init_loc = project_3d_2d(cam_init_pos)
            drift = cam_init_loc - self.initial_undeformed_markers[i]
            self.deformed_markers[i] = cam_loc - drift
            self.undeformed_markers[i] = cam_init_loc - drift
            if False:
                print('extract_markers')
                print(f'node_ix: {node_ix}')
                print(f'pos: {pos}')
                print(f'init_pos: {init_pos}')
                print(f'hom_pos: {hom_pos}')
                print(f'hom_init_pos: {hom_init_pos}')
                print(f'inv_pos: {inv_pos}')
                print(f'inv_init_pos: {inv_init_pos}')
                print(f'cam_pos: {cam_pos}')
                print(f'cam_init_pos: {cam_init_pos}')
                print(f'cam_loc: {cam_loc}')
                print(f'cam_init_loc: {cam_init_loc}')
                print(f'drift: {drift}')
                print(f'cam_loc - drift: {cam_loc - drift}')
                print(f'cam_init_loc - drift: {cam_init_loc - drift}')
                print()

    @ti.kernel
    def set_material_params(self, shell_E:ti.f32, shell_nu:ti.f32, gel_E:ti.f32, gel_nu:ti.f32):
        return

        # Set shell material parameters (Vytaflex 60)
        self.shell_youngs_modulus[None], self.shell_poissons_ratio[None] = shell_E, shell_nu
        self.shell_mu[None] = self.shell_youngs_modulus[None] / 2 / (1 + self.shell_poissons_ratio[None])
        self.shell_lam[None] = self.shell_youngs_modulus[None] * self.shell_poissons_ratio[None] / (1 + self.shell_poissons_ratio[None]) / (1 - 2 * self.shell_poissons_ratio[None])
        
        # Set gel material parameters (RTV27905)
        self.gel_youngs_modulus[None], self.gel_poissons_ratio[None] = gel_E, gel_nu
        self.gel_mu[None] = self.gel_youngs_modulus[None] / 2 / (1 + self.gel_poissons_ratio[None])
        self.gel_lam[None] = self.gel_youngs_modulus[None] * self.gel_poissons_ratio[None] / (1 + self.gel_poissons_ratio[None]) / (1 - 2 * self.gel_poissons_ratio[None])

    @ti.func
    def eul2mat(self, rot_v, trans_v):
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
    def set_vel(self, f:ti.i32):
        for p in range(self.num_vertices):
            self.vertex_velocities[f, p] = self.vertex_control_velocities[p]

    @ti.kernel
    def set_pose_control(self):
        # this is in local coord
        self.local_translational_velocity[None] = self.inverse_rotation_matrix[None] @ self.global_translational_velocity[None]
        self.local_angular_velocity[None] = self.inverse_rotation_matrix[None] @ self.global_angular_velocity_degrees[None]

        self.rotation_vector_degrees[None] = self.local_angular_velocity[None] * self.dt * (self.sub_steps -1)
        self.translation_vector[None] = self.local_translational_velocity[None] * self.dt * (self.sub_steps -1)
        self.transformation_matrix[None], self.rotation_matrix[None] = self.eul2mat(self.rotation_vector_degrees[None], self.translation_vector[None])

        self.delta_transformation_matrix[None] = self.world_transformation_matrix[None] @ self.transformation_matrix[None] @ (self.world_transformation_matrix[None].inverse())

        self.local_transformation_matrix[None] = self.transformation_matrix[None] @ self.local_transformation_matrix[None]
        self.homogeneous_transformation_matrix[None] = self.world_transformation_matrix[None] @ self.local_transformation_matrix[None]
        self.inverse_transformation_matrix[None] = self.homogeneous_transformation_matrix[None].inverse()

        self.local_rotation_matrix[None] = self.rotation_matrix[None] @ self.local_rotation_matrix[None]
        self.homogeneous_rotation_matrix[None] = self.world_rotation_matrix[None] @ self.local_rotation_matrix[None]
        self.inverse_rotation_matrix[None] = self.homogeneous_rotation_matrix[None].inverse()

        if False:
            print(f'self.d_pos_global[None]: {self.global_translational_velocity[None]}')
            print(f'self.d_pos_local[None]: {self.local_translational_velocity[None]}')
            print(f'target angular speed (rotation matrix): self.my_rot_mat[None]: {self.rotation_matrix[None]}')
            print()

    def set_pose_control_maybe_print(self):
        if False:
            print("\nInput rotation vector (self.my_rot_v[None]):")
            print(self.rotation_vector_degrees[None].to_numpy())
            print("\nInput translation vector (self.my_trans_v[None]):")
            print(self.translation_vector[None].to_numpy())
            print("\nComputed transformation matrix (self.my_trans_mat[None]):")
            print(self.transformation_matrix[None].to_numpy())
            print("\nComputed rotation matrix (self.my_rot_mat[None]):")
            print(self.rotation_matrix[None].to_numpy())
            print("\nDelta transformation matrix (dtrans_h):")
            print(self.delta_transformation_matrix[None].to_numpy())
            print("\nLocal transformation matrix (trans_local):")
            print(self.local_transformation_matrix[None].to_numpy())
            print("\nHomogeneous transformation matrix (trans_h):")
            print(self.homogeneous_transformation_matrix[None].to_numpy())
            print("\nInverse transformation matrix (inv_trans_h):")
            print(self.inverse_transformation_matrix[None].to_numpy())
            print("\nLocal rotation matrix (rot_local):")
            print(self.local_rotation_matrix[None].to_numpy())
            print("\nHomogeneous rotation matrix (rot_h):")
            print(self.homogeneous_rotation_matrix[None].to_numpy())
            print("\nInverse rotation matrix (inv_rot):")
            print(self.inverse_rotation_matrix[None].to_numpy())
            print()

    @ti.kernel
    def set_pose_control_bp(self):

        rot_v = self.local_angular_velocity[None] * self.dt * (self.sub_steps -1)
        trans_v = self.local_translational_velocity[None] * self.dt * (self.sub_steps -1)
        trans_mat, rot_mat = self.eul2mat(rot_v, trans_v)
        self.delta_transformation_matrix[None] = self.world_transformation_matrix[None] @ trans_mat @ (self.world_transformation_matrix[None].inverse())

        self.inverse_transformation_matrix[None] = self.homogeneous_transformation_matrix[None].inverse()
        self.inverse_rotation_matrix[None] = self.homogeneous_rotation_matrix[None].inverse()

    @ti.kernel
    def set_control_vel(self, f:ti.i32):
        for i in range(self.num_vertices):
            current_vertex_positions_undeformed = self.vertex_positions_undeformed[f, i]
            target_vertex_positions_undeformed = self.delta_transformation_matrix[None] @ ti.Vector([current_vertex_positions_undeformed[0], current_vertex_positions_undeformed[1], current_vertex_positions_undeformed[2], 1.0]) # 4 x 1 homogeneous
            self.vertex_control_velocities[i][0] = (target_vertex_positions_undeformed[0] - current_vertex_positions_undeformed[0]) / (self.dt * (self.sub_steps -1))
            self.vertex_control_velocities[i][1] = (target_vertex_positions_undeformed[1] - current_vertex_positions_undeformed[1]) / (self.dt * (self.sub_steps -1))
            self.vertex_control_velocities[i][2] = (target_vertex_positions_undeformed[2] - current_vertex_positions_undeformed[2]) / (self.dt * (self.sub_steps -1))

    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for k in range(self.num_contact_surface_triangles):
            a, b, c = self.contact_surface[k]
            self.total_surface_force[f] += 1/3 * self.contact_forces_on_vertices[f,a] * self.dx
            self.total_surface_force[f] += 1/3 * self.contact_forces_on_vertices[f,b] * self.dx
            self.total_surface_force[f] += 1/3 * self.contact_forces_on_vertices[f,c] * self.dx

    @ti.func
    def get_euler_angles(self) -> ti.types.vector(3, ti.f32):
        # Extract Euler angles from rotation matrix
        rot_mat = self.homogeneous_rotation_matrix[None]
        
        # Extract roll (x-axis rotation)
        roll = ti.math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        
        # Extract pitch (y-axis rotation)
        pitch = ti.math.atan2(-rot_mat[2, 0], ti.sqrt(rot_mat[2, 1] * rot_mat[2, 1] + rot_mat[2, 2] * rot_mat[2, 2]))
        
        # Extract yaw (z-axis rotation)
        yaw = ti.math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        
        # Convert to degrees
        roll_deg = ti.math.degrees(roll)
        pitch_deg = ti.math.degrees(pitch)
        yaw_deg = ti.math.degrees(yaw)
        
        euler_angles = ti.Vector([roll_deg, pitch_deg, yaw_deg])

        if False:
            print(f'rotation matrix (self.rot_h[None]): {self.homogeneous_rotation_matrix[None]}')
            print(f'euler angles (euler_angles): {euler_angles}')
            print()

        return euler_angles

    @ti.kernel
    def set_up_pose_helper(self):
        for idx in range(self.num_vertices):
            current_vertex_positions_undeformed = self.initial_vertex_positions[idx] # before any world transformation
            target_vertex_positions_undeformed = self.homogeneous_transformation_matrix[None] @ ti.Vector([current_vertex_positions_undeformed[0], current_vertex_positions_undeformed[1], current_vertex_positions_undeformed[2], 1.0]) # 4 x 1 homogeneous

            self.vertex_positions_deformed[0, idx] = ti.Vector([target_vertex_positions_undeformed[0], target_vertex_positions_undeformed[1], target_vertex_positions_undeformed[2]])
            # reset init x to track the whole body movement
            self.vertex_positions_undeformed[0, idx] = self.vertex_positions_deformed[0, idx]

        for i in range(self.num_tetrahedra):
            ia, ib, ic, id = self.tetrahedra[i]
            a, b, c, d = self.vertex_positions_deformed[0, ia], self.vertex_positions_deformed[0, ib], self.vertex_positions_deformed[0, ic], self.vertex_positions_deformed[0, id]
            deformation_gradient = ti.Matrix.cols([a - d, b - d, c - d])
            self.deformation_gradient_inverse[i] = deformation_gradient.inverse()

        self.sensor_outward_normal[None] = self.homogeneous_rotation_matrix[None] @ ti.Vector([0.0, 1.0, 0.0])

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
        cur_min_offset = self.collision_search_distance # arbitrary large value
        # cur_min_idx: triangle index (dimensionless)
        cur_min_idx = -1
        for k in range(self.num_contact_surface_triangles):
            a, b, c = self.contact_surface[k]
            # p_1, p_2, p_3: triangle vertex positions in m
            p_1 = self.vertex_positions_deformed[f, a] # triangle's 1st node
            p_2 = self.vertex_positions_deformed[f, b] # triangle's 2nd node
            p_3 = self.vertex_positions_deformed[f, c] # triangle's 3rd node
            # p_c: triangle centroid position in m
            p_c = 1/3 * (p_1 + p_2 + p_3) # center of the segment
            # offset_p: distance in m
            offset_p = (p_c - grid_p).norm(self.norm_eps) # distance to the center point of the segment

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
        vertex1_pos = self.vertex_positions_deformed[frame, vertex1_idx]
        vertex2_pos = self.vertex_positions_deformed[frame, vertex2_idx]
        vertex3_pos = self.vertex_positions_deformed[frame, vertex3_idx]

        # triangle_normal: unit normal vector (dimensionless)
        triangle_normal = ti.math.cross(vertex2_pos-vertex1_pos, vertex3_pos-vertex1_pos) # plane's norm
        triangle_normal = triangle_normal.normalized(self.norm_eps)
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

    @ti.kernel
    def reset_contact(self):
        """
        Reset all contact-related force fields to zero.
        Clears external forces and surface forces.
        """
        self.contact_forces_on_vertices.fill(0.0)
        self.total_surface_force.fill(0.0)

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
            vertex1_pos = self.vertex_positions_deformed[frame, vertex1_idx]
            vertex2_pos = self.vertex_positions_deformed[frame, vertex2_idx]
            vertex3_pos = self.vertex_positions_deformed[frame, vertex3_idx]
            vertex4_pos = self.vertex_positions_deformed[frame, vertex4_idx]
            
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
                self.vertex_velocities[frame,vertex_indices[k]] += self.dt * vertex_force / self.vertex_mass[vertex_indices[k]]
                self.vertex_velocities[frame,vertex_indices[3]] += -1*self.dt * vertex_force / self.vertex_mass[vertex_indices[3]]


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
            updated_velocity += self.dt * self.contact_forces_on_vertices[frame,vertex_idx] / self.vertex_mass[vertex_idx]

            ### stick the bottom layer to be fixed using node_labels information
            # is_fixed_layer: boolean flag (dimensionless)
            is_fixed_layer = self.is_fixed_layer[vertex_idx] == 1
            if is_fixed_layer:
                updated_velocity = self.vertex_control_velocities[vertex_idx]
            self.vertex_velocities[frame+1, vertex_idx] = updated_velocity
            self.vertex_positions_deformed[frame+1, vertex_idx] = self.vertex_positions_deformed[frame, vertex_idx] + self.dt * updated_velocity
            # update virtual pos
            self.vertex_positions_undeformed[frame+1, vertex_idx] = self.vertex_positions_undeformed[frame, vertex_idx] + self.dt * self.vertex_control_velocities[vertex_idx]


    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.num_vertices):
            self.vertex_positions_deformed[target, p] = self.vertex_positions_deformed[source, p]
            self.vertex_velocities[target, p] = self.vertex_velocities[source, p]
            self.vertex_positions_undeformed[target, p] = self.vertex_positions_undeformed[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), cache_rot: ti.types.ndarray(), cache_predict_markers: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                self.homogeneous_transformation_matrix[None][j,k] = cache_trans[j,k]
        for j in range(3):
            for k in range(3):
                self.homogeneous_rotation_matrix[None][j,k] = cache_rot[j,k]
        for p in range(self.num_vertices):
            for i in ti.static(range(3)):
                self.vertex_positions_deformed[f, p][i] = cache_pos[p,i]
                self.vertex_velocities[f, p][i] = cache_vel[p,i]
                self.vertex_positions_undeformed[f, p][i] = cache_virtual_pos[p, i]
        for p in range(self.num_markers):
            for i in ti.static(range(2)):
                self.deformed_markers[p][i] = cache_predict_markers[p,i]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), cache_rot: ti.types.ndarray(), cache_predict_markers: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                cache_trans[j,k] = self.homogeneous_transformation_matrix[None][j,k]
        for j in range(3):
            for k in range(3):
                cache_rot[j,k] = self.homogeneous_rotation_matrix[None][j,k]
        for p in range(self.num_vertices):
            for i in ti.static(range(3)):
                cache_pos[p,i] = self.vertex_positions_deformed[f, p][i]
                cache_vel[p,i] = self.vertex_velocities[f, p][i]
                cache_virtual_pos[p, i] = self.vertex_positions_undeformed[f, p][i]
        for p in range(self.num_markers):
            for i in ti.static(range(2)):
                cache_predict_markers[p,i] = self.deformed_markers[p][i]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.simulation_cache[cur_step_name] = dict()

        self.simulation_cache[cur_step_name]['pos'] = torch.zeros((self.num_vertices, 3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['vel'] = torch.zeros((self.num_vertices, 3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['trans_h'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['rot_h'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['virtual_pos'] = torch.zeros((self.num_vertices, 3), dtype=TC_TYPE, device=device)
        self.simulation_cache[cur_step_name]['predict_markers'] = torch.zeros((self.num_markers, 2), dtype=TC_TYPE, device=device)
        self.add_step_to_cache(0, self.simulation_cache[cur_step_name]['pos'], self.simulation_cache[cur_step_name]['vel'], self.simulation_cache[cur_step_name]['trans_h'], self.simulation_cache[cur_step_name]['virtual_pos'], self.simulation_cache[cur_step_name]['rot_h'], self.simulation_cache[cur_step_name]['predict_markers'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        self.copy_frame(0, self.sub_steps-1)

        self.load_step_from_cache(0, self.simulation_cache[cur_step_name]['pos'], self.simulation_cache[cur_step_name]['vel'], self.simulation_cache[cur_step_name]['trans_h'], self.simulation_cache[cur_step_name]['virtual_pos'], self.simulation_cache[cur_step_name]['rot_h'], self.simulation_cache[cur_step_name]['predict_markers'])

    def get_keypoint_indices(self, f: ti.i32):
        # Convert positions to numpy array
        positions = self.vertex_positions_deformed.to_numpy()[f]
        
        # Point A: minimum z coordinate
        z_coords = positions[:, 2]
        point_a_idx = int(np.argmin(z_coords))
        
        # Points B and C: high z coordinate points
        max_z = float(np.max(z_coords))
        z_mask = (z_coords >= (max_z - self.keypoint_search_z_threshold))
        
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
        positions = self.nodes
        
        # Point A: max y coordinate
        y_coords = positions[:, 1]
        point_a_idx = int(np.argmax(y_coords))

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
        positions = self.vertex_positions_undeformed.to_numpy()[f]
        
        # Extract coordinates for the specified indices
        coordinates = positions[keypoint_indices]
        
        return coordinates