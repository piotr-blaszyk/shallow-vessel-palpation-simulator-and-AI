"""
a class to describe sensor elastomer with FEM
"""

import taichi as ti
import torch
import cv2
from math import pi
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
import pickle
import sys
from difftactile.sensor_model.fisheye_model import * 

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class FEMDomeSensor:
    def __init__(self, dt=5e-5, sub_steps = 50, init_img_path=None):
        np.set_printoptions(precision=3, floatmode='maxprec', suppress=False)
        self.sub_steps = sub_steps
        self.dt = dt
        self.eps = 1e-11  # Small epsilon value for numerical stability in vector normalization

        # Material parameters for shell (Vytaflex 60)
        self.shell_rho = 1.145  # density [g/cm^3]
        self.shell_E = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.shell_nu = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.shell_mu = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.shell_lam = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        
        # Material parameters for gel (RTV27905)
        self.gel_rho = 0.97  # density [g/cm^3]
        self.gel_E = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.gel_nu = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.gel_mu = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.gel_lam = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

        # Initialize default material parameters
        self.shell_E[None], self.shell_nu[None] = 5e3, 0.43  # Shell (Vytaflex 60)
        self.gel_E[None], self.gel_nu[None] = 5e2, 0.49  # Gel (RTV27905)
        
        # Compute Lamé parameters for both materials
        self.shell_mu[None] = self.shell_E[None] / 2 / (1 + self.shell_nu[None])
        self.shell_lam[None] = self.shell_E[None] * self.shell_nu[None] / (1 + self.shell_nu[None]) / (1 - 2 * self.shell_nu[None])
        self.gel_mu[None] = self.gel_E[None] / 2 / (1 + self.gel_nu[None])
        self.gel_lam[None] = self.gel_E[None] * self.gel_nu[None] / (1 + self.gel_nu[None]) / (1 - 2 * self.gel_nu[None])

        self.all_nodes, self.all_f2v, self.surface_f2v, self.node_labels, element_materials, vertex_masses, self.marker_node_tags_np = self.init_mesh()

        self.marker_node_tags = ti.field(shape=(self.marker_node_tags_np[0],), dtype=int)
        self.marker_node_tags.from_numpy(self.marker_node_tags_np)
        
        # Create is_fixed_layer field from the last row of node_labels
        self.is_fixed_layer = ti.field(int, len(self.all_nodes))
        is_fixed_layer_data = self.node_labels[-1, :].astype(np.int32)  # Get the last row
        self.is_fixed_layer.from_numpy(is_fixed_layer_data)
        
        self.n_verts = len(self.all_nodes)
        self.n_cells = len(self.all_f2v)
        self.num_triangles = len(self.surface_f2v)

        # Track material type for each tetrahedron (0 for gel, 1 for shell)
        self.element_material = ti.field(dtype=ti.i32, shape=(self.n_cells,), needs_grad=False)
        # Compute mass per vertex based on material distribution
        self.mass_per_vertex = ti.field(dtype=ti.f32, shape=(self.n_verts,), needs_grad=False)
        
        # Transfer material assignments to Taichi fields
        self.element_material.from_numpy(element_materials)
        self.mass_per_vertex.from_numpy(vertex_masses)

        self.dim = 3

        self.init_x = ti.Vector.field(3, float, self.n_verts, needs_grad=False)
        self.init_x.from_numpy(self.all_nodes.astype(np.float32))

        self.shell_outer_layer_nodes = np.unique(self.surface_f2v.flatten())
        self.markers_surface_id = ti.field(int, len(self.shell_outer_layer_nodes))
        self.markers_surface_id.from_numpy(self.shell_outer_layer_nodes.astype(np.int32))
        self.num_surface = len(self.shell_outer_layer_nodes)

        # cam model
        # self.num_k_closest = 5
        # self.initial_markers, interp_idx, interp_weight = self.init_cam_model(init_img_path)
        # self.num_markers = len(self.initial_markers)
        # self.visualise_2d(interp_idx)
        self.num_markers = self.marker_node_tags_np.shape[0]

        self.predict_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=False)
        self.virtual_markers = ti.Vector.field(2, float, self.num_markers, needs_grad=False)

        # self.interp_weight = ti.Vector.field(self.num_k_closest, float, self.num_markers, needs_grad=False)
        # self.interp_weight.from_numpy(interp_weight.astype(np.float32))
        # self.interp_idx = ti.Vector.field(self.num_k_closest, int, self.num_markers)
        # self.interp_idx.from_numpy(interp_idx.astype(np.int32))

        self.f2v = ti.Vector.field(4, int, self.n_cells)
        self.f2v.from_numpy(self.all_f2v.astype(np.int32))
        self.contact_seg = ti.Vector.field(3, int, self.num_triangles) # surface triangle mesh
        self.contact_seg.from_numpy(self.surface_f2v.astype(np.int32))

        self.virtual_pos = ti.Vector.field(3, float, shape=(self.sub_steps, self.n_verts), needs_grad=False)
        self.pos = ti.Vector.field(3, float, shape=(self.sub_steps, self.n_verts), needs_grad=False)
        self.vel = ti.Vector.field(3, float, shape=(self.sub_steps, self.n_verts), needs_grad=False)

        self.B = ti.Matrix.field(3, 3, float, self.n_cells, needs_grad=False)
        self.phi = ti.field(float, self.n_cells, needs_grad=False)  # potential energy of each face (Neo-Hookean)

        self.external_force_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_verts), needs_grad=False) # contact force between FEM node to the closest particle
        self.surf_f = ti.Vector.field(3, float, shape=(self.sub_steps), needs_grad=False) # surface aggreated 3-axis forces

        # contact model parameters (default)
        self.out_direction = ti.Vector.field(3, float, (), needs_grad=False)

        ## control parameters
        self.d_pos_global = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)
        self.d_ori_global_euler_angles = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)

        self.d_pos_local = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)
        self.d_ori_local = ti.Vector.field(3, ti.f32, shape = (), needs_grad=False)

        self.rot_h = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)
        self.rot_world = ti.Matrix.field(3, 3, ti.f32, shape = ())
        self.rot_local = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)
        self.inv_rot = ti.Matrix.field(3, 3, ti.f32, shape = (), needs_grad=False)

        self.trans_h = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False) ##
        self.trans_world = ti.Matrix.field(4, 4, ti.f32, shape = ()) ## ee to world
        self.trans_local = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False) ## ee1 -> ee2
        self.inv_trans_h = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False)
        self.dtrans_h = ti.Matrix.field(4, 4, ti.f32, shape = (), needs_grad=False)
        self.control_vel = ti.Vector.field(3, float, shape = (self.n_verts), needs_grad=False)
        self.sdf = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.cache = dict() # for grad backward
        self.first = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.first[None] = True

        self.my_rot_v = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.my_trans_v = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.my_trans_mat = ti.Matrix.field(4, 4, dtype=float, shape=(), needs_grad=False)
        self.my_rot_mat = ti.Matrix.field(3, 3, dtype=float, shape=(), needs_grad=False)

    def init(self, rot_x, rot_y, rot_z, t_dx, t_dy, t_dz):
        rot = R.from_rotvec(np.deg2rad([rot_x, rot_y, rot_z]))
        init_rot = rot.as_matrix()
        trans_mat = np.eye(4)
        trans_mat[0:3,0:3] = init_rot
        trans_mat[0,3] = t_dx; trans_mat[1,3] = t_dy; trans_mat[2,3] = t_dz

        self.rot_local[None] = np.eye(3)
        self.rot_world[None] = init_rot
        self.rot_h[None] = self.rot_world[None] @ self.rot_local[None]
        self.inv_rot[None] = self.rot_h[None].inverse()

        self.trans_local[None] = np.eye(4)
        self.trans_world[None] = trans_mat
        self.trans_h[None] = self.trans_world[None] @ self.trans_local[None]
        self.inv_trans_h[None] = self.trans_local[None].inverse() @ self.trans_world[None].inverse()

        if False and self.first[None]:
            print("\nInitial rotation matrix (init_rot):")
            print(init_rot)
            print("\nInitial transformation matrix (trans_mat):")
            print(trans_mat)
            print("\nLocal rotation matrix (rot_local):")
            print(self.rot_local[None].to_numpy())
            print("\nWorld rotation matrix (rot_world):")
            print(self.rot_world[None].to_numpy())
            print("\nHomogeneous rotation matrix (rot_h):")
            print(self.rot_h[None].to_numpy())
            print("\nInverse rotation matrix (inv_rot):")
            print(self.inv_rot[None].to_numpy())
            print("\nLocal transformation matrix (trans_local):")
            print(self.trans_local[None].to_numpy())
            print("\nWorld transformation matrix (trans_world):")
            print(self.trans_world[None].to_numpy())
            print("\nHomogeneous transformation matrix (trans_h):")
            print(self.trans_h[None].to_numpy())
            print("\nInverse transformation matrix (inv_trans_h):")
            print(self.inv_trans_h[None].to_numpy())
            print()
        
        self.first[None] = False

        self.init_pos()
        with open(f"output/tactile_sensor.pos.init.pkl", 'wb') as f:
            pickle.dump(self.pos.to_numpy()[0], f)
        np.savetxt(f'output/tactile_sensor.pos.init.csv', self.pos.to_numpy()[0], delimiter=",", fmt='%.2f')
        # with open(f"./output/tactile_sensor.init_x.pkl", 'wb') as f:
        #     pickle.dump(self.init_x.to_numpy(), f)
        # np.savetxt(f'output/tactile_sensor.init_x.csv', self.init_x.to_numpy()[0], delimiter=",", fmt='%.2f')

    def init_cam_model(self, init_img_path=None):
        if init_img_path is None:
            init_img = cv2.imread("./init.png")
        else:
            init_img = cv2.imread(init_img_path)
        initial_markers, _, _ = get_marker_image(init_img)
        # Overlay initial_markers on the image and save
        overlay_img = init_img.copy()
        for pos in initial_markers:
            center = (int(round(pos[0])), int(round(pos[1])))
            cv2.circle(overlay_img, center, radius=5, color=(0, 0, 255), thickness=2)  # Red circle
        cv2.imwrite("../tasks/output/init_cam_model.png", overlay_img)
        surface_nodes = self.all_nodes[self.shell_outer_layer_nodes]
        cam_3D_nodes = np.array([surface_nodes[:,0], surface_nodes[:,2], surface_nodes[:,1]]).T
        with open(f"output/fem_sensor.cam_3d_nodes.pkl", 'wb') as f:
            pickle.dump(cam_3D_nodes, f)
        np.savetxt(f'output/fem_sensor.cam_3d_nodes.csv', cam_3D_nodes, delimiter=",", fmt='%.2f')
        cam_points = project_points_to_pix(cam_3D_nodes)
        # Overlay cam_points as green circles and save
        cam_points_img = init_img.copy()
        for pt in cam_points:
            center = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(cam_points_img, center, radius=3, color=(0, 255, 0), thickness=2)  # Green circle
        cv2.imwrite("../tasks/output/cam_points.png", cam_points_img)
        # interpolated markers in 2d & 3d
        interp_idx = []
        interp_weight = []
        surf_2d = []
        for i in range(initial_markers.shape[0]):
            offset = np.linalg.norm(initial_markers[i,0:2] - cam_points,axis=1)
            idx = np.argpartition(offset, self.num_k_closest)
            smallest_idx = idx[:self.num_k_closest]
            inv_offset = 1/offset[smallest_idx]
            total_offset = np.sum(inv_offset)
            weights = inv_offset / total_offset
            loc_2d = np.matmul(cam_points[smallest_idx].T, weights).T
            surf_2d.append(loc_2d)
            interp_idx.append(smallest_idx)
            interp_weight.append(weights)

        surf_2d = np.array(surf_2d)
        interp_idx = np.array(interp_idx)
        interp_weight = np.array(interp_weight)

        # Flatten interp_idx before saving
        interp_idx_flat = interp_idx.flatten()
        with open(f"output/fem_sensor.interp_idx_flat.pkl", 'wb') as f:
            pickle.dump(interp_idx_flat, f)
        with open(f"output/fem_sensor.shell_outer_layer_nodes.pkl", 'wb') as f:
            pickle.dump(self.shell_outer_layer_nodes, f)
        np.savetxt('output/fem_sensor.interp_idx_flat.csv', interp_idx_flat, delimiter=",", fmt='%d')
        np.savetxt('output/fem_sensor.shell_outer_layer_nodes.csv', self.shell_outer_layer_nodes, delimiter=",", fmt='%d')

        return surf_2d, interp_idx, interp_weight

    def visualise_2d(self, interp_idx):
        """
        Project selected 3D marker points to the image plane and overlay them as red dots on './init.png'.
        Save the result as 'output/init-3d-markers.png'.
        """
        # Get the relevant 3D marker points
        marker_indices = np.unique(interp_idx.flatten())
        marker_points_3d = self.all_nodes[self.shell_outer_layer_nodes][marker_indices]
        # Reorder axes to match camera convention
        cam_3D_nodes = np.array([marker_points_3d[:,0], marker_points_3d[:,2], marker_points_3d[:,1]]).T
        # Project to 2D
        marker_points_2d = project_points_to_pix(cam_3D_nodes)

        # Load the image
        img = cv2.imread('./init.png')
        if img is None:
            print('Error: Could not load ./init.png')
            return
        for pt in marker_points_2d:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (x, y), 4, (0,0,255), -1)  # Red dot
        cv2.imwrite('output/init-3d-markers.png', img)

    @ti.kernel
    def extract_markers(self, f:ti.i32):
        for i in range(self.num_markers):
            node_ix = self.marker_node_tags[i]
            pos = self.pos[f, node_ix]
            init_pos = self.virtual_pos[f, node_ix]
            hom_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0])
            hom_init_pos = ti.Vector([init_pos[0], init_pos[1], init_pos[2], 1.0])
            inv_pos = self.inv_trans_h[None] @ hom_pos
            inv_init_pos = self.inv_trans_h[None] @ hom_init_pos
            cam_pos = ti.Vector([inv_pos[0], inv_pos[2], inv_pos[1]])
            cam_init_pos = ti.Vector([inv_init_pos[0], inv_init_pos[2], inv_init_pos[1]])
            cam_loc = project_3d_2d(cam_pos)
            cam_init_loc = project_3d_2d(cam_init_pos)
            self.predict_markers[i] = cam_loc
            self.virtual_markers[i] = cam_init_loc

    @ti.kernel
    def set_material_params(self, shell_E:ti.f32, shell_nu:ti.f32, gel_E:ti.f32, gel_nu:ti.f32):
        # Set shell material parameters (Vytaflex 60)
        self.shell_E[None], self.shell_nu[None] = shell_E, shell_nu
        self.shell_mu[None] = self.shell_E[None] / 2 / (1 + self.shell_nu[None])
        self.shell_lam[None] = self.shell_E[None] * self.shell_nu[None] / (1 + self.shell_nu[None]) / (1 - 2 * self.shell_nu[None])
        
        # Set gel material parameters (RTV27905)
        self.gel_E[None], self.gel_nu[None] = gel_E, gel_nu
        self.gel_mu[None] = self.gel_E[None] / 2 / (1 + self.gel_nu[None])
        self.gel_lam[None] = self.gel_E[None] * self.gel_nu[None] / (1 + self.gel_nu[None]) / (1 - 2 * self.gel_nu[None])

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
        for p in range(self.n_verts):
            self.vel[f, p] = self.control_vel[p]

    @ti.kernel
    def set_pose_control(self):
        # this is in local coord
        self.d_pos_local[None] = self.inv_rot[None] @ self.d_pos_global[None]
        self.d_ori_local[None] = self.inv_rot[None] @ self.d_ori_global_euler_angles[None]

        self.my_rot_v[None] = self.d_ori_local[None] * self.dt * (self.sub_steps -1)
        self.my_trans_v[None] = self.d_pos_local[None] * self.dt * (self.sub_steps -1)
        self.my_trans_mat[None], self.my_rot_mat[None] = self.eul2mat(self.my_rot_v[None], self.my_trans_v[None])

        self.dtrans_h[None] = self.trans_world[None] @ self.my_trans_mat[None] @ (self.trans_world[None].inverse())

        self.trans_local[None] = self.my_trans_mat[None] @ self.trans_local[None]
        self.trans_h[None] = self.trans_world[None] @ self.trans_local[None]
        self.inv_trans_h[None] = self.trans_h[None].inverse()

        self.rot_local[None] = self.my_rot_mat[None] @ self.rot_local[None]
        self.rot_h[None] = self.rot_world[None] @ self.rot_local[None]
        self.inv_rot[None] = self.rot_h[None].inverse()

        if False:
            print(f'self.d_pos_global[None]: {self.d_pos_global[None]}')
            print(f'self.d_pos_local[None]: {self.d_pos_local[None]}')
            print(f'target angular speed (rotation matrix): self.my_rot_mat[None]: {self.my_rot_mat[None]}')
            print()

    def set_pose_control_maybe_print(self):
        if False:
            print("\nInput rotation vector (self.my_rot_v[None]):")
            print(self.my_rot_v[None].to_numpy())
            print("\nInput translation vector (self.my_trans_v[None]):")
            print(self.my_trans_v[None].to_numpy())
            print("\nComputed transformation matrix (self.my_trans_mat[None]):")
            print(self.my_trans_mat[None].to_numpy())
            print("\nComputed rotation matrix (self.my_rot_mat[None]):")
            print(self.my_rot_mat[None].to_numpy())
            print("\nDelta transformation matrix (dtrans_h):")
            print(self.dtrans_h[None].to_numpy())
            print("\nLocal transformation matrix (trans_local):")
            print(self.trans_local[None].to_numpy())
            print("\nHomogeneous transformation matrix (trans_h):")
            print(self.trans_h[None].to_numpy())
            print("\nInverse transformation matrix (inv_trans_h):")
            print(self.inv_trans_h[None].to_numpy())
            print("\nLocal rotation matrix (rot_local):")
            print(self.rot_local[None].to_numpy())
            print("\nHomogeneous rotation matrix (rot_h):")
            print(self.rot_h[None].to_numpy())
            print("\nInverse rotation matrix (inv_rot):")
            print(self.inv_rot[None].to_numpy())
            print()

    @ti.kernel
    def set_pose_control_bp(self):

        rot_v = self.d_ori_local[None] * self.dt * (self.sub_steps -1)
        trans_v = self.d_pos_local[None] * self.dt * (self.sub_steps -1)
        trans_mat, rot_mat = self.eul2mat(rot_v, trans_v)
        self.dtrans_h[None] = self.trans_world[None] @ trans_mat @ (self.trans_world[None].inverse())

        self.inv_trans_h[None] = self.trans_h[None].inverse()
        self.inv_rot[None] = self.rot_h[None].inverse()

    @ti.kernel
    def set_control_vel(self, f:ti.i32):
        for i in range(self.n_verts):
            init_t_pos = self.virtual_pos[f, i]
            after_t_pos = self.dtrans_h[None] @ ti.Vector([init_t_pos[0], init_t_pos[1], init_t_pos[2], 1.0]) # 4 x 1 homogeneous
            self.control_vel[i][0] = (after_t_pos[0] - init_t_pos[0]) / (self.dt * (self.sub_steps -1))
            self.control_vel[i][1] = (after_t_pos[1] - init_t_pos[1]) / (self.dt * (self.sub_steps -1))
            self.control_vel[i][2] = (after_t_pos[2] - init_t_pos[2]) / (self.dt * (self.sub_steps -1))

    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for k in range(self.num_triangles):
            a, b, c = self.contact_seg[k]
            self.surf_f[f] += 1/3 * self.external_force_field[f,a] * self.dx
            self.surf_f[f] += 1/3 * self.external_force_field[f,b] * self.dx
            self.surf_f[f] += 1/3 * self.external_force_field[f,c] * self.dx

    @ti.func
    def get_euler_angles(self) -> ti.types.vector(3, ti.f32):
        # Extract Euler angles from rotation matrix
        rot_mat = self.rot_h[None]
        
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
            print(f'rotation matrix (self.rot_h[None]): {self.rot_h[None]}')
            print(f'euler angles (euler_angles): {euler_angles}')
            print()

        return euler_angles

    def init_mesh(self):
        # Load mesh data from gmsh
        with open('output/gmsh-mesh.pkl', 'rb') as f:
            mesh_data = pickle.load(f)
        
        # Unpack mesh data
        all_tetrahedra = mesh_data['all_tetrahedra']
        node_coordinates = mesh_data['node_coordinates']
        node_labels = mesh_data['node_labels']
        surface_node_tags = mesh_data['surface_node_tags']
        surface_triangles = mesh_data['surface_triangles']
        node_tags = mesh_data['node_tags']
        group_to_idx = mesh_data['group_to_idx']
        y_bottom = mesh_data['y_bottom']
        R_inner = mesh_data['R_inner']
        R_outer = mesh_data['R_outer']
        marker_node_tags = mesh_data['marker_node_tags']
        
        # Compute fixed layer nodes (nodes at the bottom)
        is_fixed_layer = np.abs(node_coordinates[:, 1] - y_bottom) < 1  # Check if y-coordinate is at bottom
        
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
            matrix = np.vstack([pos2 - pos1, pos3 - pos1, pos4 - pos1]).T
            volume = abs(np.linalg.det(matrix)) / 6.0
            
            # Get the node labels for all nodes in this tetrahedron
            tetra_node_labels = node_labels[tetra]
            
            # Check if any node is part of the gel
            has_gel = np.any(tetra_node_labels[:, group_to_idx['gel']])
            # Check if all nodes are part of the shell
            all_shell = np.all(tetra_node_labels[:, group_to_idx['shell']])
            
            # Assign material based on node composition
            if has_gel:
                element_materials[i] = 2  # gel material
            elif all_shell:
                element_materials[i] = 1  # shell material
            
            # Determine density based on material
            material_density = self.shell_rho if element_materials[i] == 1 else self.gel_rho
            element_mass = volume * material_density
            
            # Distribute element mass to vertices
            for vertex_idx in tetra:
                vertex_masses[vertex_idx] += element_mass / 4.0  # Equal distribution
        
        max_y = np.max(node_coordinates[:, 1])
        y_translation = 20 - max_y
        translation_vector = np.array([0, y_translation, 0])
        node_coordinates = node_coordinates + translation_vector

        return node_coordinates, all_tetrahedra, surface_triangles, node_labels, element_materials, vertex_masses, marker_node_tags

    @ti.kernel
    def init_pos(self):
        for idx in range(self.n_verts):
            before_t_pos = self.init_x[idx] # before any world transformation
            after_t_pos = self.trans_h[None] @ ti.Vector([before_t_pos[0], before_t_pos[1], before_t_pos[2], 1.0]) # 4 x 1 homogeneous

            self.pos[0, idx] = ti.Vector([after_t_pos[0], after_t_pos[1], after_t_pos[2]])
            # reset init x to track the whole body movement
            self.virtual_pos[0, idx] = self.pos[0, idx]

        for i in range(self.n_cells):
            ia, ib, ic, id = self.f2v[i]
            a, b, c, d = self.pos[0, ia], self.pos[0, ib], self.pos[0, ic], self.pos[0, id]
            B_i_inv = ti.Matrix.cols([a - d, b - d, c - d])
            self.B[i] = B_i_inv.inverse()

        self.out_direction[None] = self.rot_h[None] @ ti.Vector([0.0, 1.0, 0.0])

    @ti.func
    def find_closest(self, grid_p, f):
        cur_min_offset = 100.0 # arbitrary large value
        cur_min_idx = -1
        for k in range(self.num_triangles):
            a, b, c = self.contact_seg[k]
            p_1 = self.pos[f, a] # triangle's 1st node
            p_2 = self.pos[f, b] # triangle's 2nd node
            p_3 = self.pos[f, c] # triangle's 3rd node
            p_c = 1/3 * (p_1 + p_2 + p_3) # center of the segment
            offset_p = (p_c - grid_p).norm(self.eps) # distance to the center point of the segment

            if (offset_p < cur_min_offset):
                cur_min_offset = offset_p
                cur_min_idx = k

        return cur_min_idx

    @ti.func
    def find_sdf(self, point_position, point_velocity, triangle_index, frame, collision_offset = 0.0):
        vertex1_idx, vertex2_idx, vertex3_idx = self.contact_seg[triangle_index]
        vertex1_pos = self.pos[frame, vertex1_idx]
        vertex2_pos = self.pos[frame, vertex2_idx]
        vertex3_pos = self.pos[frame, vertex3_idx]

        triangle_normal = ti.math.cross(vertex2_pos-vertex1_pos, vertex3_pos-vertex1_pos) # plane's norm
        triangle_normal = triangle_normal.normalized(self.eps)
        normal_direction = ti.math.sign(triangle_normal.dot(self.out_direction[None]))
        triangle_normal = normal_direction * triangle_normal # facing up

        point_to_vertex1 = point_position - vertex1_pos # vector from the first node to the particle
        signed_distance = point_to_vertex1.dot(triangle_normal) # distance to the plane
        point_projected = point_position - signed_distance * triangle_normal # projection of point on the segment
        surface_normal = -1* triangle_normal

        edge1 = vertex3_pos - vertex1_pos
        edge2 = vertex2_pos - vertex1_pos
        point_projected_rel = point_projected - vertex1_pos
        dot_edge1_edge1 = edge1.dot(edge1)
        dot_edge1_edge2 = edge1.dot(edge2)
        dot_edge1_point = edge1.dot(point_projected_rel)
        dot_edge2_edge2 = edge2.dot(edge2)
        dot_edge2_point = edge2.dot(point_projected_rel)
        inv_denominator = 1 / (dot_edge1_edge1 * dot_edge2_edge2 - dot_edge1_edge2 * dot_edge1_edge2)
        barycentric_u = (dot_edge2_edge2 * dot_edge1_point - dot_edge1_edge2 * dot_edge2_point) * inv_denominator
        barycentric_v = (dot_edge1_edge1 * dot_edge2_point - dot_edge1_edge2 * dot_edge1_point) * inv_denominator

        ### correct with an offset for pbd collision
        signed_distance -= collision_offset
        relative_velocity = point_velocity - 1/3 * (self.vel[frame, vertex1_idx] + self.vel[frame, vertex2_idx] + self.vel[frame, vertex3_idx])
        is_contact = signed_distance < 0 and barycentric_u >= 0 and barycentric_v >= 0.0 and (barycentric_u + barycentric_v <= 1)
        return signed_distance, surface_normal, relative_velocity, is_contact

    @ti.kernel
    def reset_contact(self):
        self.external_force_field.fill(0.0)
        self.surf_f.fill(0.0)

    @ti.func
    def update_contact_force(self, triangle_index, contact_force, frame):
        vertex1_idx, vertex2_idx, vertex3_idx = self.contact_seg[triangle_index]
        # Distribute contact force equally among triangle vertices
        self.external_force_field[frame, vertex1_idx] += 1/3 * contact_force
        self.external_force_field[frame, vertex2_idx] += 1/3 * contact_force
        self.external_force_field[frame, vertex3_idx] += 1/3 * contact_force

    @ti.kernel
    def update_internal_forces(self, frame:ti.i32):
        for tetra_idx in range(self.n_cells):
            vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx = self.f2v[tetra_idx]
            vertex1_pos, vertex2_pos, vertex3_pos, vertex4_pos = self.pos[frame, vertex1_idx], self.pos[frame, vertex2_idx], self.pos[frame, vertex3_idx], self.pos[frame, vertex4_idx]
            deformation_matrix = ti.Matrix.cols([vertex1_pos - vertex4_pos, vertex2_pos - vertex4_pos, vertex3_pos - vertex4_pos])
            tetra_volume = ti.abs(deformation_matrix.determinant()) / 6
            deformation_gradient = deformation_matrix @ self.B[tetra_idx]

            ## Get material parameters based on element type
            is_shell = self.element_material[tetra_idx] == 1
            mu = self.shell_mu[None] if is_shell else self.gel_mu[None]
            lam = self.shell_lam[None] if is_shell else self.gel_lam[None]

            ## stable neo-hooken
            jacobian = deformation_gradient.determinant()
            first_invariant = (deformation_gradient.transpose() @ deformation_gradient).trace()
            dJ_dF0 = deformation_gradient[:,1].cross(deformation_gradient[:,2])
            dJ_dF1 = deformation_gradient[:,2].cross(deformation_gradient[:,0])
            dJ_dF2 = deformation_gradient[:,0].cross(deformation_gradient[:,1])
            jacobian_derivative = ti.Matrix.cols([dJ_dF0, dJ_dF1, dJ_dF2])
            alpha = 1 + 0.75 * mu/lam
            stress_tensor = mu * (1 - 1/(first_invariant+1)) * deformation_gradient + lam * (jacobian - alpha) * jacobian_derivative

            force_matrix = -tetra_volume * stress_tensor @ self.B[tetra_idx].transpose()
            vertex_indices = ti.Vector([vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx])
            for k in ti.static(range(3)):
                vertex_force = ti.Vector([force_matrix[j,k] for j in range(3)])
                self.vel[frame,vertex_indices[k]] += self.dt * vertex_force / self.mass_per_vertex[vertex_indices[k]]
                self.vel[frame,vertex_indices[3]] += -1*self.dt * vertex_force / self.mass_per_vertex[vertex_indices[3]]


    @ti.kernel
    def update_external_forces(self, frame:ti.i32):
        for vertex_idx in range(self.n_verts):
            updated_velocity = ti.Vector([0.0, 0.0, 0.0])
            updated_velocity += self.vel[frame,vertex_idx]
            updated_velocity += self.dt * self.external_force_field[frame,vertex_idx] / self.mass_per_vertex[vertex_idx]

            ### stick the bottom layer to be fixed using node_labels information
            is_fixed_layer = self.is_fixed_layer[vertex_idx] == 1
            if is_fixed_layer:
                updated_velocity = self.control_vel[vertex_idx]
            self.vel[frame+1, vertex_idx] = updated_velocity
            self.pos[frame+1, vertex_idx] = self.pos[frame, vertex_idx] + self.dt * updated_velocity
            # update virtual pos
            self.virtual_pos[frame+1, vertex_idx] = self.virtual_pos[frame, vertex_idx] + self.dt * self.control_vel[vertex_idx]


    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_verts):
            self.pos[target, p] = self.pos[source, p]
            self.vel[target, p] = self.vel[source, p]
            self.virtual_pos[target, p] = self.virtual_pos[source, p]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_verts):
            self.pos.grad[target, p] = self.pos.grad[source, p]
            self.vel.grad[target, p] = self.vel.grad[source, p]
            self.virtual_pos.grad[target, p] = self.virtual_pos.grad[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), cache_rot: ti.types.ndarray(), cache_predict_markers: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                self.trans_h[None][j,k] = cache_trans[j,k]
        for j in range(3):
            for k in range(3):
                self.rot_h[None][j,k] = cache_rot[j,k]
        for p in range(self.n_verts):
            for i in ti.static(range(self.dim)):
                self.pos[f, p][i] = cache_pos[p,i]
                self.vel[f, p][i] = cache_vel[p,i]
                self.virtual_pos[f, p][i] = cache_virtual_pos[p, i]
        for p in range(self.num_markers):
            for i in ti.static(range(2)):
                self.predict_markers[p][i] = cache_predict_markers[p,i]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_trans: ti.types.ndarray(), cache_virtual_pos: ti.types.ndarray(), cache_rot: ti.types.ndarray(), cache_predict_markers: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                cache_trans[j,k] = self.trans_h[None][j,k]
        for j in range(3):
            for k in range(3):
                cache_rot[j,k] = self.rot_h[None][j,k]
        for p in range(self.n_verts):
            for i in ti.static(range(self.dim)):
                cache_pos[p,i] = self.pos[f, p][i]
                cache_vel[p,i] = self.vel[f, p][i]
                cache_virtual_pos[p, i] = self.virtual_pos[f, p][i]
        for p in range(self.num_markers):
            for i in ti.static(range(2)):
                cache_predict_markers[p,i] = self.predict_markers[p][i]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['pos'] = torch.zeros((self.n_verts, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['vel'] = torch.zeros((self.n_verts, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['trans_h'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['rot_h'] = torch.zeros((3,3), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['virtual_pos'] = torch.zeros((self.n_verts, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['predict_markers'] = torch.zeros((self.num_markers, 2), dtype=TC_TYPE, device=device)
        self.add_step_to_cache(0, self.cache[cur_step_name]['pos'], self.cache[cur_step_name]['vel'], self.cache[cur_step_name]['trans_h'], self.cache[cur_step_name]['virtual_pos'], self.cache[cur_step_name]['rot_h'], self.cache[cur_step_name]['predict_markers'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)

        self.load_step_from_cache(0, self.cache[cur_step_name]['pos'], self.cache[cur_step_name]['vel'], self.cache[cur_step_name]['trans_h'], self.cache[cur_step_name]['virtual_pos'], self.cache[cur_step_name]['rot_h'], self.cache[cur_step_name]['predict_markers'])

    @ti.kernel
    def clear_loss_grad(self):
        self.predict_markers.grad.fill(0.0)
        self.d_pos_local.grad[None].fill(0.0)
        self.d_ori_local.grad[None].fill(0.0)
        self.rot_h.grad[None].fill(0.0)
        self.rot_local.grad[None].fill(0.0)
        self.inv_rot.grad[None].fill(0.0)
        self.trans_h.grad[None].fill(0.0)
        self.trans_local.grad[None].fill(0.0)
        self.inv_trans_h.grad[None].fill(0.0)
        self.dtrans_h.grad[None].fill(0.0)
        self.control_vel.grad.fill(0.0)

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        self.surf_f.grad.fill(0.0)
        self.external_force_field.grad.fill(0.0)
        for p in range(self.n_verts):
            for t in range(f):
                self.pos.grad[t, p].fill(0.0)
                self.vel.grad[t, p].fill(0.0)
                self.virtual_pos.grad[t, p].fill(0.0)

    def get_min_z_from_cache(self, t):
        """Returns the minimum z-coordinate from the cached positions at timestep t.
        
        Args:
            t (int): The timestep to get the minimum z-coordinate from
            
        Returns:
            float: The minimum z-coordinate across all vertices
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        # Get the z-coordinates (third column) and find minimum
        z_coords = self.cache[cur_step_name]['pos'][:, 2]
        return float(z_coords.min().item())

    def get_min_z_ix(self, t):
        """Returns the minimum z-coordinate from the cached positions at timestep t.
        
        Args:
            t (int): The timestep to get the minimum z-coordinate from
            
        Returns:
            float: The minimum z-coordinate across all vertices
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        # Get the z-coordinates (third column) and find minimum
        z_coords = self.cache[cur_step_name]['pos'][:, 2]
        return z_coords.argmin().item()

    def get_high_z_max_y_ix(self, t):
        """Returns the index of the point that has maximum y-coordinate among points
        whose z-coordinate is within 0.2 of the maximum z value.
        
        Args:
            t (int): The timestep to get the point index from
            
        Returns:
            int: The index of the vertex that meets the criteria
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        # Get all positions
        positions = self.cache[cur_step_name]['pos']
        
        # Get z-coordinates and find maximum
        z_coords = positions[:, 2]
        max_z = float(z_coords.max().item())
        
        # Create mask for points within 0.2 of max z
        z_mask = (z_coords >= (max_z - 0.2))
        
        # Get y-coordinates of points that meet z criteria
        y_coords = positions[:, 1]
        y_coords_filtered = y_coords.clone()
        y_coords_filtered[~z_mask] = float('-inf')  # Set non-matching points to negative infinity
        
        # Find index of maximum y among filtered points
        return int(y_coords_filtered.argmax().item())

    def get_xyz_angle_from_cache(self, t, idx):
        """Returns the x and y coordinates of a point from the cached positions at timestep t,
        and computes the angle this point forms with a reference point, assuming the point lies
        on a circle centered at the reference point.
        
        Args:
            t (int): The timestep to get the coordinates from
            idx (int): The index of the point to get coordinates for
            ref_x (float): x-coordinate of the reference point (center of circle)
            ref_y (float): y-coordinate of the reference point (center of circle)
            
        Returns:
            tuple: (x, y, angle) where:
                - x, y are the coordinates as floats
                - angle is in degrees, measured counterclockwise from positive x-axis
        """
        cur_step_name = f'{t:06d}'
        if cur_step_name not in self.cache:
            raise KeyError(f"No cached data found for timestep {t}")
            
        if idx < 0 or idx >= self.n_verts:
            raise ValueError(f"Index {idx} out of bounds. Must be between 0 and {self.n_verts-1}")
            
        # Get the x and y coordinates (first and second columns)
        point = self.cache[cur_step_name]['pos'][idx]
        x = float(point[0].item())
        y = float(point[1].item())
        z = float(point[2].item())
        
        # Compute angle using atan2 and convert to degrees
        dx = x - 12.50
        dy = y - 11.50
        angle_rad = np.arctan2(dy, dx)
        
        return (x, y, z), angle_rad

    def get_keypoint_indices(self, f: ti.i32):
        # Convert positions to numpy array
        positions = self.pos.to_numpy()[f]
        
        # Point A: minimum z coordinate
        z_coords = positions[:, 2]
        point_a_idx = int(np.argmin(z_coords))
        
        # Points B and C: high z coordinate points
        max_z = float(np.max(z_coords))
        z_mask = (z_coords >= (max_z - 0.2))
        
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
        positions = self.all_nodes
        
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
        positions = self.pos.to_numpy()[f]
        
        # Extract coordinates for the specified indices
        coordinates = positions[keypoint_indices]
        
        return coordinates

    def compute_mean_deformation_top_10_percent(self):
        """
        Compute the mean Euclidean distance between the initial virtual positions and the current positions at frame 0,
        but only for the 10% of points that deformed the most.
        Returns:
            float: The mean Euclidean distance of the top 10% most deformed points between self.virtual_pos[0] and self.pos[0]
        """
        # Convert Taichi fields to numpy arrays for frame 0
        virtual_pos_np = self.virtual_pos.to_numpy()[0]  # shape: (n_verts, 3)
        pos_np = self.pos.to_numpy()[0]                  # shape: (n_verts, 3)

        # Compute Euclidean distances for each point
        distances = np.linalg.norm(virtual_pos_np - pos_np, axis=1)
        n_top = max(1, int(0.1 * len(distances)))
        # Get the mean of the largest 10% of distances
        top_distances = np.partition(distances, -n_top)[-n_top:]
        mean_top_distance = np.mean(top_distances)
        return mean_top_distance