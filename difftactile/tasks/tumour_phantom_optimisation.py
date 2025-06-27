from difftactile.tasks.tumour_phantom_visualisation import (
    ContactVisualisation,
    set_up_gui,
    update_gui,
)
from difftactile.sensor_model.vitactip import ViTacTip
from difftactile.object_model.phantom import Phantom

import taichi as ti
import numpy as np
import pickle
import json

RUN_ON_LAB_MACHINE = True
@ti.data_oriented
class Contact(ContactVisualisation):
    def __init__(self):
        super().__init__()
        self.set_up_system_params()
        self.vitactip = ViTacTip()
        self.phantom = Phantom()
        self.set_up_initial_positions_and_trajectory_first_init_only()
        self.set_up_initial_positions_and_trajectory()
        self.set_up_keypoints()
        self.set_up_collision_detection()
        self.init_visualisation()
        self.set_up_pid()
        self.set_up_snapshot()

    def set_up_collision_detection(self):
        self.num_sensor = 1
        self.contact_idx = ti.Vector.field(
            self.num_sensor,
            dtype=int,
            shape=(
                self.num_sub_frames,
                self.phantom.n_grid,
                self.phantom.n_grid,
                self.phantom.n_grid,
            ),
        )
        self.contact_detect_flag = ti.field(float, (), needs_grad=False)

    def set_up_system_params(self):
        with open('../tasks/system-params.json', 'r') as f:
            self.params = json.load(f)
            self.contact_params = self.params['contact']

        self.num_opt_steps = self.contact_params['num_opt_steps']
        self.num_frames = self.contact_params['num_frames']
        self.num_sub_frames = self.contact_params['num_sub_frames']
        self.dt = self.contact_params['dt']

        self.normal_stiffness = ti.field(dtype=float, shape=(), needs_grad=False)
        self.normal_damping = ti.field(dtype=float, shape=(), needs_grad=False)
        self.tangential_stiffness = ti.field(dtype=float, shape=(), needs_grad=False)
        self.coulomb_friction_coeff = ti.field(dtype=float, shape=(), needs_grad=False)
        self.normal_stiffness[None] = self.contact_params['normal_stiffness']
        self.normal_damping[None] = self.contact_params['normal_damping']
        self.tangential_stiffness[None] = self.contact_params['tangential_stiffness']
        self.coulomb_friction_coeff[None] = self.contact_params['coulomb_friction_coeff']

        self.norm_eps = 1e-11

    def set_up_snapshot(self):        # Allocate snapshot fields (num_opt_steps, num_markers, 2)
        self.predict_markers_snapshots = ti.Vector.field(2, dtype=ti.f32, shape=(self.num_opt_steps, self.vitactip.num_markers), needs_grad=False)
        self.virtual_markers_snapshots = ti.Vector.field(2, dtype=ti.f32, shape=(self.num_opt_steps, self.vitactip.num_markers), needs_grad=False)
        self.ground_truth_labels = ti.field(dtype=bool, shape=(self.num_opt_steps,), needs_grad=False)

    def set_up_pid(self):
        # PID controller parameters
        self.pid_controller_kp = ti.field(dtype=float, shape=(), needs_grad=False)  # Proportional gain
        self.pid_controller_ki = ti.field(dtype=float, shape=(), needs_grad=False)  # Integral gain
        self.pid_controller_kd = ti.field(dtype=float, shape=(), needs_grad=False)  # Derivative gain
        self.pid_controller_kp[None] = self.contact_params['pid_kp']
        self.pid_controller_ki[None] = self.contact_params['pid_ki']
        self.pid_controller_kd[None] = self.contact_params['pid_kd']
        
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
        self.position_tolerance[None] = 0.1  # 1 mm tolerance
        self.orientation_tolerance = ti.field(dtype=float, shape=(), needs_grad=False)
        self.orientation_tolerance[None] = 1  # 1 degree tolerance
        
        # Add fields for dwell time control
        self.dwell_frames = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_frames[None] = self.contact_params['dwell_frames'] # Number of frames to stay at each target
        self.dwell_counter = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_counter[None] = 0
        self.is_dwelling = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.is_dwelling[None] = False
        self.last_target_reached = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.last_target_reached[None] = False
        self.frames_since_last_target_reached = ti.field(dtype=int, shape=(), needs_grad=False)
        self.frames_since_last_target_reached[None] = 0

    def set_up_initial_positions_and_trajectory_first_init_only(self):
        with open('../tasks/initial-coordinates.json', 'r') as f:
            self.coordinates = json.load(f)

        self.phantom_closest_vertex = self.coordinates['phantom_closest_vertex']
        self.phantom_centroid_pose = self.coordinates['phantom_centroid_pose']
        self.vitactip_tip_pose = self.coordinates['vitactip_tip_pose']
        self.gap = self.coordinates['gap']

        self.tactile_sensor_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.phantom_initial_position = ti.Vector.field(3, dtype=ti.f32, shape=1, needs_grad=False)
        self.trajectory = ti.Vector.field(6, dtype=float, shape=1, needs_grad=False)
        self.tumour_present = ti.field(dtype=bool, shape=(), needs_grad=False)
        self.tumour_present[None] = False
    
    def set_up_keypoints(self):
        self.keypoint_indices = self.vitactip.get_keypoint_indices(0)
        self.keypoint_indices = np.concatenate((self.keypoint_indices, self.vitactip.marker_node_tags_np), dtype=int)

    def set_up_initial_positions_and_trajectory(self):
        ix = self.vitactip.get_keypoint_indices_numpy_point_a()
        camera_lens_to_sensor_tip = self.vitactip.nodes[ix, 1]
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
        self.trajectory_npy = np.array([
            [x, y, z, xr, yr, zr],
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
        self.is_dwelling[None] = False
        self.last_target_reached[None] = False
        self.frames_since_last_target_reached[None] = 0

    def update(self, f):
        self.phantom.compute_new_F(f)
        self.phantom.svd(f)
        self.phantom.p2g(f)
        self.vitactip.update_internal_forces(f)
        self.phantom.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.phantom.grid_op(f)
        self.phantom.g2p(f)
        self.vitactip.update_external_forces(f)

    def update_grad(self, f):
        self.vitactip.update_external_forces.grad(f)
        self.phantom.g2p.grad(f)
        self.phantom.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.vitactip.update_internal_forces.grad(f)
        self.phantom.p2g.grad(f)
        self.phantom.svd_grad(f)
        self.phantom.compute_new_F.grad(f)

    @ti.kernel
    def clear_loss_grad(self):
        self.normal_stiffness.grad[None] = 0.0
        self.normal_damping.grad[None] = 0.0
        self.tangential_stiffness.grad[None] = 0.0
        self.coulomb_friction_coeff.grad[None] = 0.0
        self.contact_detect_flag.grad[None] = 0.0

    def clear_traj_grad(self):
        self.vitactip.clear_loss_grad()
        self.phantom.clear_loss_grad()
        self.clear_loss_grad()

    def clear_all_grad(self):
        self.clear_traj_grad()
        self.vitactip.clear_step_grad(self.num_sub_frames)
        self.phantom.clear_step_grad(self.num_sub_frames)

    def reset(self):
        self.vitactip.reset_contact()
        self.phantom.reset()
        self.contact_idx.fill(-1)
        self.contact_detect_flag[None] = 0.0

    @ti.kernel
    def clamp_grid(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            self.phantom.n_grid, self.phantom.n_grid, self.phantom.n_grid
        ):
            self.phantom.grid_node_mass.grad[f, i, j, k] = ti.math.clamp(
                self.phantom.grid_node_mass.grad[f, i, j, k], -1000.0, 1000.0
            )
        for i in range(self.vitactip.n_verts):
            self.vitactip.pos.grad[f, i] = ti.math.clamp(
                self.vitactip.pos.grad[f, i], -1000.0, 1000.0
            )
            self.vitactip.vel.grad[f, i] = ti.math.clamp(
                self.vitactip.vel.grad[f, i], -1000.0, 1000.0
            )

    @ti.func
    def calculate_contact_force(self, signed_distance, surface_normal, relative_velocity):
        tangential_force = ti.Vector([0.0, 0.0, 0.0])
        tangential_velocity = ti.Vector([0.0, 0.0, 0.0])
        contact_relative_velocity = relative_velocity
        normal_velocity_magnitude = ti.max(surface_normal.dot(contact_relative_velocity), 0)
        normal_force = (
            -(self.normal_stiffness[None] + self.normal_damping[None] * normal_velocity_magnitude) * signed_distance * surface_normal
        )
        tangential_velocity = contact_relative_velocity - surface_normal.dot(contact_relative_velocity) * surface_normal
        tangential_velocity_magnitude = tangential_velocity.norm(self.norm_eps)
        if tangential_velocity_magnitude > 1e-4:
            tangential_force = (
                1.0
                * (tangential_velocity / tangential_velocity_magnitude)
                * ti.min(
                    self.tangential_stiffness[None] * tangential_velocity_magnitude,
                    self.coulomb_friction_coeff[None] * normal_force.norm(self.norm_eps),
                )
            )
        total_contact_force = normal_force + tangential_force
        return total_contact_force, normal_force, tangential_force

    @ti.kernel
    def check_collision(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            self.phantom.n_grid, self.phantom.n_grid, self.phantom.n_grid
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * self.phantom.grid_cube_size,
                        (j + 0.5) * self.phantom.grid_cube_size,
                        (k + 0.5) * self.phantom.grid_cube_size,
                    ]
                )
                closest_sensor_vertex_idx = self.vitactip.find_closest(grid_node_position, frame)
                self.contact_idx[frame, i, j, k] = closest_sensor_vertex_idx

    @ti.kernel
    def collision(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            self.phantom.n_grid, self.phantom.n_grid, self.phantom.n_grid
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * self.phantom.grid_cube_size,
                        (j + 0.5) * self.phantom.grid_cube_size,
                        (k + 0.5) * self.phantom.grid_cube_size,
                    ]
                )
                grid_node_velocity = self.phantom.grid_node_momentum_in[frame, i, j, k] / (
                    self.phantom.grid_node_mass[frame, i, j, k] + self.phantom.eps
                )
                closest_sensor_vertex_idx = self.contact_idx[frame, i, j, k]
                penetration_depth, surface_normal, relative_velocity, is_in_contact = (
                    self.vitactip.find_sdf(grid_node_position, grid_node_velocity, closest_sensor_vertex_idx, frame)
                )
                if is_in_contact:
                    total_contact_force, _, _ = self.calculate_contact_force(
                        penetration_depth, -1 * surface_normal, -1 * relative_velocity
                    )
                    self.phantom.update_contact_force(total_contact_force, frame, i, j, k)
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
        current_pos = self.vitactip.virtual_pos[0, self.keypoint_indices[0]]
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

        if self.last_target_reached[None]:
            self.frames_since_last_target_reached[None] += 1

        # If target is reached and not already dwelling, start dwelling (only for non-final targets)
        if (not self.last_target_reached[None] and not self.is_dwelling[None] and 
            pos_error_magnitude < self.position_tolerance[None] and 
            ori_error_magnitude < self.orientation_tolerance[None]):
            self.is_dwelling[None] = True
            self.dwell_counter[None] = 0
            print(f'target {self.current_target_idx[None]} ({target}) reached at time step {ts}!')
        
        # If dwelling, increment counter and check if dwell time is complete
        if self.is_dwelling[None]:
            self.dwell_counter[None] += 1
            if self.dwell_counter[None] >= self.dwell_frames[None]:
                self.is_dwelling[None] = False
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
                    self.last_target_reached[None] = True
        
        # If dwelling, set control outputs to zero to maintain position
        # But if at final target, never dwell, always actively control
        if self.is_dwelling[None]:
            self.vitactip.translational_velocity_global[None] = ti.Vector([0.0, 0.0, 0.0])
            self.vitactip.angular_velocity_global_degrees[None] = ti.Vector([0.0, 0.0, 0.0])
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
            max_speed_pos = 0.01  # 1 cm/s = 0.01 m/s
            pos_control_norm = pos_control.norm()
            if clamp_speed and pos_control_norm > max_speed_pos:
                pos_control = pos_control / pos_control_norm * max_speed_pos
            
            ori_control = self.pid_controller_kp[None] * ori_error + self.pid_controller_ki[None] * self.ori_error_sum[None] + self.pid_controller_kd[None] * ori_derivative
            
            # Clamp ori_control to max_speed_ori
            max_speed_ori = max_speed_pos * 2.0  # rad/s
            ori_control_norm = ori_control.norm()
            if clamp_speed and ori_control_norm > max_speed_ori:
                ori_control = ori_control / ori_control_norm * max_speed_ori

            # Set control outputs
            self.vitactip.translational_velocity_global[None] = pos_control
            self.vitactip.angular_velocity_global_degrees[None] = ori_control
        
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
            self.predict_markers_snapshots[opts, i] = self.vitactip.predict_markers[i]
            self.virtual_markers_snapshots[opts, i] = self.vitactip.virtual_markers[i]
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

        with open('output/marker_snapshots_and_labels.pkl', 'wb') as f:
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
        save_markers_with_empty_rows(predict_np, 'output/predict_markers_snapshots.csv')
        save_markers_with_empty_rows(virtual_np, 'output/virtual_markers_snapshots.csv')
        np.savetxt('output/ground_truth_labels.csv', labels_np, delimiter=',', fmt='%d')

    def save_tactile_sensor_mesh_to_pickle(self, ts):
        particles = self.vitactip.pos.to_numpy()[0]
        with open(f'output/tactile_sensor.deformed_node_coordinates.ts={ts}.pkl', 'wb') as f:
            pickle.dump(particles, f)
        print('mesh exported!')
    
    def save_tactile_sensor_mesh_node_mapping_to_pickle(self):
        f2v = self.vitactip.f2v.to_numpy()
        with open(f'output/tactile_sensor.f2v.pkl', 'wb') as f:
            pickle.dump(f2v, f)
        print('mesh node mapping exported!')

def main():
    np.set_printoptions(precision=3, floatmode='maxprec', suppress=False)
    if RUN_ON_LAB_MACHINE:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cuda, device_memory_GB=9)
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)

    gui_tuple = set_up_gui()

    with open('../tasks/system-params.json', 'r') as f:
        params = json.load(f)
        contact_params = params['contact']
    num_sub_frames = contact_params['num_sub_frames']
    num_frames = contact_params['num_frames']
    num_opt_steps = contact_params['num_opt_steps']

    contact_model = Contact()
    np.savetxt(f'output/trajectory.p_sensor1.csv', contact_model.p_sensor1.to_numpy(), delimiter=",", fmt='%.2f')
    np.savetxt(f'output/trajectory.o_sensor1.csv', contact_model.o_sensor1.to_numpy(), delimiter=",", fmt='%.2f')
    contact_model.save_tactile_sensor_mesh_node_mapping_to_pickle()
    
    for opts in range(num_opt_steps):
        print(f"optimisation step: {opts} / {num_opt_steps}")
        contact_model.set_up_initial_positions_and_trajectory()
        contact_model.reset_pid_controller()
        contact_model.reset_3d_scene()
        if opts == 0:
            contact_model.vitactip.extract_initial_markers(0)
            contact_model.vitactip.extract_markers(0)
            initial_markers = contact_model.vitactip.predict_markers.to_numpy()
            with open('output/sim-markers-initial-positions.pkl', 'wb') as f:
                pickle.dump(initial_markers, f)
        print('forward')
        for ts in range(num_frames - 1):
            contact_model.pid_controller(ts)
            contact_model.vitactip.set_pose_control()
            contact_model.vitactip.set_pose_control_maybe_print()
            contact_model.vitactip.set_control_vel(0)
            contact_model.vitactip.set_vel(0)
            contact_model.reset()
            for ss in range(num_sub_frames - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(0)

            keypoint_coords = contact_model.vitactip.get_keypoint_coordinates(0, contact_model.keypoint_indices[-1].reshape((1,)))
            # keypoint_coords = contact_model.trajectory_npy
            update_gui(contact_model, gui_tuple, num_frames, ts, keypoint_coords)

            if ts % 100 == 0:
                contact_model.take_snapshot(opts)
                contact_model.save_tactile_sensor_mesh_to_pickle(ts)
                # break
    # contact_model.save_marker_data_and_ground_truth_labels_to_file()

if __name__ == "__main__":
    main()
