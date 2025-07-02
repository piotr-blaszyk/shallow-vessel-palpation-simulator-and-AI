import taichi as ti
import numpy as np
np.set_printoptions(precision=6, suppress=False, formatter={'float': '{:0.6e}'.format})
import pickle
import json
import cv2
import sys

from difftactile.sensor_model.vitactip import ViTacTip
from difftactile.object_model.phantom import Phantom
from difftactile.main.constants import *

RUN_ON_LAB_MACHINE = True
@ti.data_oriented
class Contact:
    def __init__(self):
        self.set_up_system_params()
        self.vitactip = ViTacTip()
        self.phantom = Phantom()
        self.set_up_initial_positions_and_trajectory_first_init_only()
        self.set_up_initial_positions_and_trajectory()
        self.set_up_keypoints()
        self.set_up_collision_detection()
        self.set_up_pid()
        self.set_up_snapshot()
        self.visualisation_initialise()

    def set_up_collision_detection(self):
        self.num_sensor = 1
        self.contact_idx = ti.Vector.field(
            self.num_sensor,
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS.phantom.n_grid_x,
                SYSTEM_PARAMS.phantom.n_grid_y,
                SYSTEM_PARAMS.phantom.n_grid_z,
            ),
        )
        self.contact_detect_flag = ti.field(float, (), needs_grad=False)

    def set_up_system_params(self):
        self.normal_stiffness = ti.field(dtype=float, shape=(), needs_grad=False)
        self.normal_damping = ti.field(dtype=float, shape=(), needs_grad=False)
        self.tangential_stiffness = ti.field(dtype=float, shape=(), needs_grad=False)
        self.coulomb_friction_coeff = ti.field(dtype=float, shape=(), needs_grad=False)
        self.normal_stiffness[None] = SYSTEM_PARAMS.contact.normal_stiffness
        self.normal_damping[None] = SYSTEM_PARAMS.contact.normal_damping
        self.tangential_stiffness[None] = SYSTEM_PARAMS.contact.tangential_stiffness
        self.coulomb_friction_coeff[None] = SYSTEM_PARAMS.contact.coulomb_friction_coeff

    def set_up_snapshot(self):        # Allocate snapshot fields (SYSTEM_PARAMS.contact.num_opt_steps, num_markers, 2)
        self.predict_markers_snapshots = ti.Vector.field(2, dtype=ti.f32, shape=(SYSTEM_PARAMS.contact.num_opt_steps, self.vitactip.num_markers), needs_grad=False)
        self.virtual_markers_snapshots = ti.Vector.field(2, dtype=ti.f32, shape=(SYSTEM_PARAMS.contact.num_opt_steps, self.vitactip.num_markers), needs_grad=False)
        self.ground_truth_labels = ti.field(dtype=int, shape=(SYSTEM_PARAMS.contact.num_opt_steps,), needs_grad=False)

    def set_up_pid(self):
        # PID controller parameters
        self.pid_controller_kp = ti.field(dtype=float, shape=(), needs_grad=False)  # Proportional gain
        self.pid_controller_ki = ti.field(dtype=float, shape=(), needs_grad=False)  # Integral gain
        self.pid_controller_kd = ti.field(dtype=float, shape=(), needs_grad=False)  # Derivative gain
        self.pid_controller_kp[None] = SYSTEM_PARAMS.contact.pid_kp
        self.pid_controller_ki[None] = SYSTEM_PARAMS.contact.pid_ki
        self.pid_controller_kd[None] = SYSTEM_PARAMS.contact.pid_kd

        self.pid_controller_max_speed_translation = ti.field(dtype=float, shape=(), needs_grad=False)
        self.pid_controller_max_speed_rotation = ti.field(dtype=float, shape=(), needs_grad=False)
        self.pid_controller_max_speed_translation[None] = SYSTEM_PARAMS.contact.pid_max_speed_translation
        self.pid_controller_max_speed_rotation[None] = SYSTEM_PARAMS.contact.pid_max_speed_rotation
        
        # Error accumulation for integral term
        self.pos_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.ori_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)  # Changed from 4 to 3 for Euler angles
        
        # Previous error for derivative term
        self.prev_pos_error = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.prev_ori_error = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)  # Changed from 4 to 3 for Euler angles

        # Add fields to track current target and control state
        self.current_target_idx = ti.field(dtype=int, shape=(), needs_grad=False)
        self.current_target_idx[None] = 0
        self.position_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.position_tolerance[None] = SYSTEM_PARAMS.contact.pid_position_tolerance
        self.orientation_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.orientation_tolerance[None] = SYSTEM_PARAMS.contact.pid_orientation_tolerance
        
        # Add fields for dwell time control
        self.dwell_frames = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_frames[None] = SYSTEM_PARAMS.contact.dwell_frames # Number of frames to stay at each target
        self.dwell_counter = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_counter[None] = 0
        self.is_dwelling = ti.field(dtype=int, shape=(), needs_grad=False)
        self.is_dwelling[None] = 0
        self.last_target_reached = ti.field(dtype=int, shape=(), needs_grad=False)
        self.last_target_reached[None] = 0
        self.frames_since_last_target_reached = ti.field(dtype=int, shape=(), needs_grad=False)
        self.frames_since_last_target_reached[None] = 0

    def set_up_initial_positions_and_trajectory_first_init_only(self):
        self.phantom_closest_vertex = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        self.phantom_centroid_pose = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose
        self.vitactip_tip_pose = SYSTEM_PARAMS_COMPUTED.vitactip_tip_pose
        self.min_coord = SYSTEM_PARAMS_COMPUTED.min_coord
        self.max_coord_x = SYSTEM_PARAMS_COMPUTED.max_coord_x
        self.max_coord_y = SYSTEM_PARAMS_COMPUTED.max_coord_y
        self.max_coord_z = SYSTEM_PARAMS_COMPUTED.max_coord_z

        self.tactile_sensor_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.phantom_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.trajectory = ti.Vector.field(6, dtype=float, shape=3, needs_grad=False)
        self.tumour_present = ti.field(dtype=int, shape=(), needs_grad=False)
        self.tumour_present[None] = 0
    
    def set_up_keypoints(self):
        self.keypoint_indices = np.concatenate((
            self.vitactip.get_keypoint_indices(0), 
            self.phantom.get_keypoint_index(),
        ), dtype=int)

    def set_up_initial_positions_and_trajectory(self):
        ix = self.vitactip.get_keypoint_indices_numpy_point_a()
        camera_lens_to_sensor_tip = self.vitactip.node_coordinates[ix, 2]
        tumour_present = False
        self.tumour_present[None] = tumour_present
        self.phantom.set_state_from_outside(
            pos=self.phantom_centroid_pose[:3],
            ori=self.phantom_centroid_pose[3:],
            vel=[0.0, 0.0, 0.0],
            cylinder_tuple=None,
            stiffness_tuple=None,
            tumour_present=tumour_present,
        )
        x, y, z, xr, yr, zr = self.vitactip_tip_pose
        press_depth = SYSTEM_PARAMS.geometry.gap + 0.006
        self.trajectory_npy = np.array([
            [x, y, z, xr, yr, zr],
            [x, y, z-press_depth, xr, yr, zr],
            [x, y+0.020, z-press_depth, xr, yr, zr],
        ], dtype=float)
        assert self.trajectory.shape[0] == self.trajectory_npy.shape[0], f"Set self.trajectory length to {self.trajectory_npy.shape[0]} match trajectory_npy"
        self.trajectory.from_numpy(self.trajectory_npy)

        self.sensor_dome_tip_initial_pose = self.trajectory_npy[0].tolist()
        self.sensor_dome_tip_initial_pose[2] += camera_lens_to_sensor_tip
        t_dx, t_dy, t_dz, rot_x, rot_y, rot_z = self.sensor_dome_tip_initial_pose
        self.vitactip.set_up_pose(rot_x, rot_y, rot_z, t_dx, t_dy, t_dz)
        self.tactile_sensor_initial_position[0] = ti.Vector(self.sensor_dome_tip_initial_pose[:3])
        self.phantom_initial_position[0] = ti.Vector(self.phantom_centroid_pose[:3])
    
    def reset_pid_controller(self):
        self.pos_error_sum.fill(0)
        self.ori_error_sum.fill(0)
        self.prev_pos_error.fill(0)
        self.prev_ori_error.fill(0)
        self.current_target_idx[None] = 0
        self.dwell_counter[None] = 0
        self.is_dwelling[None] = 0
        self.last_target_reached[None] = 0
        self.frames_since_last_target_reached[None] = 0

    def update(self, f):
        self.phantom.compute_trial_deformation_gradient(f)
        self.phantom.svd_of_trial_deformation_gradient(f)
        self.phantom.p2g(f)
        self.vitactip.update_internal_forces(f)
        self.phantom.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.phantom.grid_op(f)
        self.phantom.g2p(f)
        self.vitactip.update_external_forces(f)

    def reset(self):
        self.vitactip.reset_contact()
        self.phantom.reset()
        self.contact_idx.fill(-1)
        self.contact_detect_flag[None] = 0.0

    @ti.func
    def calculate_contact_force(self, signed_distance, surface_normal, relative_velocity):
        """
        Calculate contact forces between phantom and vitactip sensor.
        
        Args:
            signed_distance: penetration depth in meters (m)
            surface_normal: unit normal vector (dimensionless)
            relative_velocity: relative velocity vector in m/s
            
        Returns:
            total_contact_force: total contact force vector in N
            normal_force: normal force vector in N  
            tangential_force: tangential force vector in N
        """
        # tangential_force: force vector in N
        tangential_force = ti.Vector([0.0, 0.0, 0.0])
        # tangential_velocity: velocity vector in m/s
        tangential_velocity = ti.Vector([0.0, 0.0, 0.0])
        # contact_relative_velocity: relative velocity vector in m/s
        contact_relative_velocity = relative_velocity
        # normal_velocity_magnitude: velocity magnitude in m/s
        normal_velocity_magnitude = ti.max(surface_normal.dot(contact_relative_velocity), 0)
        # normal_force: force vector in N
        normal_force = (
            -(self.normal_stiffness[None] + self.normal_damping[None] * normal_velocity_magnitude) * signed_distance * surface_normal
        )
        # tangential_velocity: velocity vector in m/s
        tangential_velocity = contact_relative_velocity - surface_normal.dot(contact_relative_velocity) * surface_normal
        # tangential_velocity_magnitude: velocity magnitude in m/s
        tangential_velocity_magnitude = tangential_velocity.norm(SYSTEM_PARAMS.contact.norm_eps)
        if tangential_velocity_magnitude > SYSTEM_PARAMS.contact.tangential_velocity_detection_threshold:
            # tangential_force: force vector in N
            tangential_force = (
                1.0
                * (tangential_velocity / tangential_velocity_magnitude)
                * ti.min(
                    self.tangential_stiffness[None] * tangential_velocity_magnitude,
                    self.coulomb_friction_coeff[None] * normal_force.norm(SYSTEM_PARAMS.contact.norm_eps),
                )
            )
        # total_contact_force: force vector in N
        total_contact_force = normal_force + tangential_force
        return total_contact_force, normal_force, tangential_force

    @ti.kernel
    def check_collision(self, frame: ti.i32):
        """
        Check for potential collisions between phantom grid nodes and vitactip sensor.
        Finds the closest sensor vertex for each occupied grid node.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x, SYSTEM_PARAMS.phantom.n_grid_y, SYSTEM_PARAMS.phantom.n_grid_z
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                # grid_node_position: position vector in m
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                # closest_sensor_vertex_idx: vertex index (dimensionless)
                closest_sensor_vertex_idx = self.vitactip.find_closest(grid_node_position, frame)
                self.contact_idx[frame, i, j, k] = closest_sensor_vertex_idx

    @ti.kernel
    def collision(self, frame: ti.i32):
        """
        Handle collision response between phantom grid nodes and vitactip sensor.
        Calculates contact forces and applies them to both objects.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x, SYSTEM_PARAMS.phantom.n_grid_y, SYSTEM_PARAMS.phantom.n_grid_z
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                # grid_node_position: position vector in m
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                # grid_node_velocity: velocity vector in m/s
                grid_node_velocity = self.phantom.grid_node_momentum_in[frame, i, j, k] / (
                    self.phantom.grid_node_mass[frame, i, j, k] + SYSTEM_PARAMS.phantom.eps
                )
                # closest_sensor_vertex_idx: vertex index (dimensionless)
                closest_sensor_vertex_idx = self.contact_idx[frame, i, j, k]
                if closest_sensor_vertex_idx[0] != -1:
                    # penetration_depth: depth in m, surface_normal: unit vector (dimensionless), 
                    # relative_velocity: velocity vector in m/s, is_in_contact: boolean (dimensionless)
                    penetration_depth, surface_normal, relative_velocity, is_in_contact = (
                        self.vitactip.find_sdf(grid_node_position, grid_node_velocity, closest_sensor_vertex_idx, frame)
                    )
                    if is_in_contact:
                        # total_contact_force: force vector in N
                        total_contact_force, _, _ = self.calculate_contact_force(
                            penetration_depth, -1 * surface_normal, -1 * relative_velocity
                        )
                        self.phantom.update_contact_impulse(total_contact_force, frame, i, j, k)
                        self.vitactip.update_contact_force(closest_sensor_vertex_idx, -1 * total_contact_force, frame)

    def memory_to_cache(self, t):
        self.vitactip.memory_to_cache(t)
        self.phantom.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.vitactip.memory_from_cache(t)
        self.phantom.memory_from_cache(t)

    @ti.kernel
    def pid_controller(self, ts: ti.i32):
        # Get current position and orientation using reference keypoint
        current_pos = self.vitactip.vertex_positions_undeformed_global_coordinates[0, self.keypoint_indices[0]]
        current_ori = self.vitactip.get_euler_angles()
        
        # Get current target position and orientation
        target = self.trajectory[self.current_target_idx[None]]
        target_pos = ti.Vector([target[0], target[1], target[2]])
        target_ori = ti.Vector([target[3], target[4], target[5]])  # Now using Euler angles
        
        # Compute position and orientation errors
        pos_error = target_pos - current_pos
        ori_error = target_ori - current_ori
        
        # Check if current target is reached
        pos_error_magnitude = pos_error.norm()
        ori_error_magnitude = ori_error.norm()

        if self.last_target_reached[None] == 1:
            self.frames_since_last_target_reached[None] += 1

        # If target is reached and not already dwelling, start dwelling (only for non-final targets)
        if (self.last_target_reached[None] == 0 and self.is_dwelling[None] == 0 and 
            pos_error_magnitude < self.position_tolerance[None] and 
            ori_error_magnitude < self.orientation_tolerance[None]):
            self.is_dwelling[None] = 1
            self.dwell_counter[None] = 0
            print(f'target {self.current_target_idx[None]} ({target}) reached at time step {ts}!')

        # If dwelling, increment counter and check if dwell time is complete
        if self.is_dwelling[None] == 1:
            self.dwell_counter[None] += 1
            if self.dwell_counter[None] >= self.dwell_frames[None]:
                self.is_dwelling[None] = 0
                if self.current_target_idx[None] < self.trajectory.shape[0] - 1:
                    self.current_target_idx[None] += 1
                    # Reset error sums when switching targets
                    self.pos_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.ori_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_pos_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_ori_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    
                    # Get new target position and orientation
                    target = self.trajectory[self.current_target_idx[None]]
                    target_pos = ti.Vector([target[0], target[1], target[2]])
                    target_ori = ti.Vector([target[3], target[4], target[5]])
                    
                    # Recompute errors for new target
                    pos_error = target_pos - current_pos
                    ori_error = target_ori - current_ori
                else:
                    self.last_target_reached[None] = 1
        
        # If dwelling, set control outputs to zero to maintain position
        # But if at final target, never dwell, always actively control
        if self.is_dwelling[None] == 1:
            self.vitactip.global_translational_velocity[None] = ti.Vector([0.0, 0.0, 0.0])
            self.vitactip.global_angular_velocity_degrees[None] = ti.Vector([0.0, 0.0, 0.0])
        else:
            # Update error sums for integral term
            self.pos_error_sum[None] += pos_error
            self.ori_error_sum[None] += ori_error
            
            # Compute derivative term
            pos_derivative = pos_error - self.prev_pos_error[None]
            ori_derivative = ori_error - self.prev_ori_error[None]
            
            # Store current error for next iteration
            self.prev_pos_error[None] = pos_error
            self.prev_ori_error[None] = ori_error
            
            # Compute PID control output
            pos_control = self.pid_controller_kp[None] * pos_error + self.pid_controller_ki[None] * self.pos_error_sum[None] + self.pid_controller_kd[None] * pos_derivative
            
            clamp_speed = True
            # Clamp pos_control to max_speed
            max_speed_pos = self.pid_controller_max_speed_translation[None]
            pos_control_norm = pos_control.norm()
            if clamp_speed and pos_control_norm > max_speed_pos:
                pos_control = pos_control / pos_control_norm * max_speed_pos
            
            ori_control = self.pid_controller_kp[None] * ori_error + self.pid_controller_ki[None] * self.ori_error_sum[None] + self.pid_controller_kd[None] * ori_derivative
            
            # Clamp ori_control to max_speed_ori
            max_speed_ori = self.pid_controller_max_speed_rotation[None]
            ori_control_norm = ori_control.norm()
            if clamp_speed and ori_control_norm > max_speed_ori:
                ori_control = ori_control / ori_control_norm * max_speed_ori

            if SYSTEM_PARAMS.meta.enable_pid_controller == 1:
                self.vitactip.global_translational_velocity[None] = pos_control
                self.vitactip.global_angular_velocity_degrees[None] = ori_control
            else:
                self.vitactip.global_translational_velocity[None] = ti.Vector([0.0, 0.0, 0.0])
                self.vitactip.global_angular_velocity_degrees[None] = ti.Vector([0.0, 0.0, 0.0])
        
            if False:
                # Print all variables used in the function
                print("\nPID Control Variables:")
                print(f"Current Position (current_pos): {current_pos}")
                print(f"Current Orientation (current_ori): {current_ori}")
                print(f"Target Position (target_pos): {target_pos}")
                print(f"Target Orientation (target_ori): {target_ori}")
                print(f"Position Error (pos_error): {pos_error}")
                print(f"Orientation Error (ori_error): {ori_error}")
                print(f"Position Error Magnitude (pos_error_magnitude): {pos_error_magnitude}")
                print(f"Orientation Error Magnitude (ori_error_magnitude): {ori_error_magnitude}")
                print(f"Position Error Sum (self.pos_error_sum[None]): {self.pos_error_sum[None]}")
                print(f"Orientation Error Sum (self.ori_error_sum[None]): {self.ori_error_sum[None]}")
                print(f"Previous Position Error (self.prev_pos_error[None]): {self.prev_pos_error[None]}")
                print(f"Previous Orientation Error (self.prev_ori_error[None]): {self.prev_ori_error[None]}")
                print(f"Position Derivative (pos_derivative): {pos_derivative}")
                print(f"Orientation Derivative (ori_derivative): {ori_derivative}")
                print(f"Position Control Output (pos_control): {pos_control}")
                print(f"Orientation Control Output (ori_control): {ori_control}")
                print(f"Current Target Index (self.current_target_idx[None]): {self.current_target_idx[None]}")
                print(f"Is Dwelling (self.is_dwelling[None]): {self.is_dwelling[None]}")
                print(f"Dwell Counter (self.dwell_counter[None]): {self.dwell_counter[None]}")
                print()

    @ti.kernel
    def take_snapshot(self, opts: ti.i32):
        for i in range(self.vitactip.num_markers):
            self.predict_markers_snapshots[opts, i] = self.vitactip.deformed_markers[i]
            self.virtual_markers_snapshots[opts, i] = self.vitactip.undeformed_markers[i]
        self.ground_truth_labels[opts] = self.tumour_present[None]

    def save_marker_data_and_ground_truth_labels_to_file(self):
        # Convert to numpy arrays
        predict_np = self.predict_markers_snapshots.to_numpy()
        virtual_np = self.virtual_markers_snapshots.to_numpy()
        labels_np = self.ground_truth_labels.to_numpy()
        # Save all to a single pickle file
        out = { 
            'predict_markers_snapshots': predict_np,
            'virtual_markers_snapshots': virtual_np,
            'ground_truth_labels': labels_np,
        }

        with open(SYSTEM_PARAMS.files.marker_snapshots_and_labels, 'wb') as f:
            pickle.dump(out, f)
        # Save each array to a separate CSV file in the requested format
        def save_markers_with_empty_rows(arr, filename):
            num_steps, num_markers, dim = arr.shape
            rows = []
            for step in range(num_steps):
                rows.append(arr[step])  # shape (num_markers, 2)
                rows.append(np.full((1, dim), np.nan))  # empty row
            out_arr = np.vstack(rows)
            np.savetxt(filename, out_arr, delimiter=',')
        save_markers_with_empty_rows(predict_np, SYSTEM_PARAMS.files.predict_markers_snapshots)
        save_markers_with_empty_rows(virtual_np, SYSTEM_PARAMS.files.virtual_markers_snapshots)
        np.savetxt(SYSTEM_PARAMS.files.ground_truth_labels, labels_np, delimiter=',', fmt='%d')

    def save_tactile_sensor_mesh_to_pickle(self, ts):
        particles = self.vitactip.vertex_positions_deformed_global_coordinates.to_numpy()[0]
        with open(SYSTEM_PARAMS.files.deformed_node_coordinates.format(ts), 'wb') as f:
            pickle.dump(particles, f)
        # print(f'mesh exported at ts: {ts}!')
    
    def save_tactile_sensor_mesh_node_mapping_to_pickle(self):
        f2v = self.vitactip.tetrahedra.to_numpy()
        with open(SYSTEM_PARAMS.files.tactile_sensor_f2v, 'wb') as f:
            pickle.dump(f2v, f)
        # print('mesh node mapping exported!')
    
    def visualisation_initialise(self):
        self.key_points = ti.Vector.field(3, dtype=ti.f32, shape=(9,), needs_grad=False)
        self.key_points_per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=(9,), needs_grad=False)
        self.sensor_points = ti.Vector.field(
            3, dtype=float, shape=(self.vitactip.num_vertices)
        )
        self.healthy_tissue_points = ti.Vector.field(
            3, dtype=float, shape=(self.phantom.actual_total_num_particles,)
        )
        self.tumour_points = ti.Vector.field(
            3, dtype=float, shape=(self.phantom.actual_total_num_particles,)
        )
        self.healthy_tissue_points_von_mises_stress = ti.field(
            dtype=float, shape=(self.phantom.actual_total_num_particles,)
        )
        self.tumour_points_von_mises_stress = ti.field(
            dtype=float, shape=(self.phantom.actual_total_num_particles,)
        )
        
        # Initialize fields for tactile readout visualization
        # For marker points and their deformed positions
        self.marker_points = ti.Vector.field(2, dtype=float, shape=(self.vitactip.num_markers,))
        self.marker_offsets = ti.Vector.field(2, dtype=float, shape=(self.vitactip.num_markers,))
        
        # For arrow lines (each arrow needs start and end point)
        self.arrow_line_vertices = ti.Vector.field(2, dtype=float, shape=(self.vitactip.num_markers * 2,))
        
        # For clock arm points
        self.clock_arm_points = ti.Vector.field(2, dtype=float, shape=(2,))
        self.clock_arm_points_per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=(2,), needs_grad=False)
        
        # Window size field for tactile readout
        self.window_size = ti.Vector.field(2, dtype=float, shape=())
        self.window_size[None] = [640.0, 480.0]

    @ti.kernel
    def visualisation_reset_3d_scene(self):
        self.healthy_tissue_points.fill(0)
        self.tumour_points.fill(0)

    @ti.kernel
    def visualisation_draw_3d_scene(self, f: ti.i32):
        for p in range(self.phantom.actual_total_num_particles):
            if self.phantom.titles[p] == 0:
                self.healthy_tissue_points[p] = self.phantom.particle_position[f, p]
                self.healthy_tissue_points_von_mises_stress[p] = self.phantom.particle_von_mises_stress[0, p]
            elif self.phantom.titles[p] == 1:
                self.tumour_points[p] = self.phantom.particle_position[f, p]
                self.tumour_points_von_mises_stress[p] = self.phantom.particle_von_mises_stress[0, p]

        for p in range(self.vitactip.num_vertices):
            self.sensor_points[p] = self.vitactip.vertex_positions_deformed_global_coordinates[f, p]

    @ti.kernel
    def prepare_tactile_readout_data(self):
        # Process marker positions and create arrow vertices
        for i in range(self.vitactip.num_markers):
            # Get undeformed and deformed positions
            undeformed = self.vitactip.undeformed_markers[i]
            deformed = self.vitactip.deformed_markers[i]
            
            # Flip y coordinates
            # undeformed[1] = self.window_size[None][1] - undeformed[1]
            # deformed[1] = self.window_size[None][1] - deformed[1]
            
            # Calculate offset
            offset = deformed - undeformed
            
            # Normalize coordinates by window size
            undeformed = undeformed / self.window_size[None]
            
            # Store normalized undeformed position for marker points
            self.marker_points[i] = undeformed
            
            # Store arrow vertices (start and end points)
            self.arrow_line_vertices[i * 2] = undeformed  # Start point
            self.arrow_line_vertices[i * 2 + 1] = undeformed + (offset / self.window_size[None])  # End point

    @ti.kernel
    def prepare_clock_arm_points(self):
        # Process clock arm positions
        for i in range(2):
            point = self.vitactip.projection_2d_clock_arms[i]
            # Flip y coordinate
            # point[1] = self.window_size[None][1] - point[1]
            # Normalize coordinates by window size
            self.clock_arm_points[i] = point / self.window_size[None]

    def visualisation_draw_tactile_readout(self):
        self.vitactip.extract_markers(0)
        self.prepare_tactile_readout_data()

        self.vitactip.extract_clock_arm_2d_projections(0)
        self.prepare_clock_arm_points()

        # Set background image each frame
        self.tactile_canvas.set_image(self.bg_image)
        self.tactile_canvas.circles(self.marker_points, radius=0.01, color=(1, 0, 0))
        self.tactile_canvas.lines(self.arrow_line_vertices, color=(0, 1, 0), width=0.01)
        
        # Draw clock arm points
        self.tactile_canvas.circles(
            self.clock_arm_points, 
            radius=0.05, 
            per_vertex_color=self.clock_arm_points_per_vertex_color,
        )
        
        self.tactile_window.show()

    def visualisation_set_up_gui(self):
        screen_width = 1920
        screen_height = 1080
        
        # Main 3D visualization window
        self.window = ti.ui.Window("high-level camera", (int(screen_width * 0.5), int(screen_height * 0.5)))
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0, 0, 0))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        x, y, z = self.vitactip_tip_pose[:3]
        self.camera.position(x, y, z+1.0)
        self.camera.up(1, 0, 0)
        self.camera.lookat(x, y, z)
        self.camera.fov(8)
        
        self.tactile_window = ti.ui.Window("tactile readout", (640, 480))
        self.tactile_canvas = self.tactile_window.get_canvas()
        # Load and store background image
        self.bg_image = cv2.imread(SYSTEM_PARAMS.files.vitactip_photo_default_state)
        self.bg_image = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        # Rotate image to correct orientation
        self.bg_image = cv2.rotate(self.bg_image, cv2.ROTATE_90_CLOCKWISE)

        clock_arm_points_per_vertex_color_npy = np.array([
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        self.clock_arm_points_per_vertex_color.from_numpy(clock_arm_points_per_vertex_color_npy)

        key_points_per_vertex_color_npy = np.tile([1., 0., 0.], (9, 1))
        key_points_per_vertex_color_npy[-2:, :] = clock_arm_points_per_vertex_color_npy
        self.key_points_per_vertex_color.from_numpy(key_points_per_vertex_color_npy)

    def visualisation_update_gui(self, ts):
        vitactip_coords = self.vitactip.vertex_positions_deformed_global_coordinates.to_numpy()[0]
        if np.isnan(vitactip_coords).any():
            nan_count = np.any(np.isnan(vitactip_coords), axis=1).sum()
            print(f'ViTacTip contains {nan_count} / {vitactip_coords.shape[0]} nan vertices at ts: {ts}')
        phantom_coords = self.phantom.particle_position.to_numpy()[0]
        if np.isnan(phantom_coords).any():
            nan_count = np.any(np.isnan(phantom_coords), axis=1).sum()
            print(f'phantom contains {nan_count} / {phantom_coords.shape[0]} nan vertices at ts: {ts}')

        move_to_the_front_offset = np.array([-0.050, 0, 0], dtype=float)
        z = self.min_coord
        x0, y0, _ = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose[:3]
        x1, y1, _ = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        floor = np.array([
            [x0, y1, z],
            [x0, y0, z],
            [x0, y0 + abs(y0 - y1), z],
        ])
        floor += move_to_the_front_offset
        vitactip_bottom = self.vitactip.get_keypoint_coordinates(0, self.keypoint_indices[0].reshape((1,)))
        trajectory_keypoints = self.trajectory_npy[:, :3].copy()
        vitactip_bottom += move_to_the_front_offset
        trajectory_keypoints += move_to_the_front_offset
        vitactip_clock_arms = self.vitactip.get_keypoint_coordinates(f=0, keypoint_indices=self.vitactip.clock_arms_node_idxs.to_numpy())
        self.keypoint_coords = np.vstack((vitactip_bottom, trajectory_keypoints, floor, vitactip_clock_arms))

        self.visualisation_draw_tactile_readout()
        
        # Update main 3D scene
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        self.visualisation_draw_3d_scene(0)
        
        self.scene.particles(
            self.healthy_tissue_points,
            color=(0.0, 0.0, 1.0),
            radius=6e-4,
        )
        self.scene.particles(
            self.tumour_points,
            color=(1.0, 1.0, 0.0),
            radius=6e-4,
        )
        self.scene.particles(
            self.sensor_points,
            color=(0.0, 1.0, 0.0),
            radius=6e-4,
        )
        
        assert self.keypoint_coords.shape[0] == self.key_points.shape[0], f"Set self.key_points to shape ({self.keypoint_coords.shape[0]},)"
        if self.keypoint_coords is not None:
            self.key_points.from_numpy(self.keypoint_coords)
            self.scene.particles(
                self.key_points,
                color=(1.0, 0.0, 0.0),
                per_vertex_color=self.key_points_per_vertex_color,
                radius=9e-4,
            )
        
        self.canvas.scene(self.scene)
        self.window.show()

def main():
    if RUN_ON_LAB_MACHINE:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cuda, device_memory_GB=9)
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)
    
    contact_model = Contact()
    contact_model.visualisation_set_up_gui()
    contact_model.save_tactile_sensor_mesh_node_mapping_to_pickle()
    
    for opts in range(SYSTEM_PARAMS.contact.num_opt_steps):
        print(f"optimisation step: {opts} / {SYSTEM_PARAMS.contact.num_opt_steps}")
        contact_model.set_up_initial_positions_and_trajectory()
        contact_model.reset_pid_controller()
        contact_model.visualisation_reset_3d_scene()
        if opts == 0:
            contact_model.vitactip.test_mapping_from_global_space_to_camera_space()
            contact_model.vitactip.extract_markers(0)
            contact_model.vitactip.copy_markers_to_initial_markers_for_drift_correction()
            contact_model.vitactip.save_predicted_markers_to_image()
            contact_model.vitactip.get_keypoint_idxs()
            initial_markers = contact_model.vitactip.deformed_markers.to_numpy()
            with open(SYSTEM_PARAMS.files.sim_markers_initial_positions, 'wb') as f:
                pickle.dump(initial_markers, f)
        print('forward')
        for ts in range(SYSTEM_PARAMS.contact.num_frames - 1):
            contact_model.pid_controller(ts)
            contact_model.vitactip.set_pose_control()
            contact_model.vitactip.set_control_vel(0)
            contact_model.vitactip.set_vel(0)
            contact_model.reset()
            for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(0)

            contact_model.visualisation_update_gui(ts)

            if ts % 1_000 == 0:
                print(f'ts: {ts}')

            if ts % 100 == 0:
                contact_model.take_snapshot(opts)
                contact_model.save_tactile_sensor_mesh_to_pickle(ts)
                # break
    # contact_model.save_marker_data_and_ground_truth_labels_to_file()

if __name__ == "__main__":
    main()
