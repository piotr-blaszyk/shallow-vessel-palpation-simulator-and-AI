import taichi as ti
import numpy as np
import pickle
import json
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from difftactile.sensor_model.vitactip import ViTacTip
from difftactile.object_model.phantom import Phantom
from difftactile.main.constants import *
from difftactile.main.cfl_and_contact_params_estimation import *
from scipy.spatial.transform import Rotation as R

RUN_ON_LAB_MACHINE = True


@ti.data_oriented
class Contact:
    def __init__(self):
        self.set_up_system_params()
        self.load_system_identification_data()
        self.vitactip = ViTacTip()
        self.phantom = Phantom()
        self.set_up_initial_positions_and_trajectory_first_init_only()
        self.set_up_initial_positions_and_trajectory()
        self.set_up_keypoints()
        self.set_up_collision_detection()
        self.set_up_pid()
        self.set_up_snapshot()
        self.set_up_loss_computation()
        self.visualisation_initialise()

    @ti.kernel
    def fp(self):
        self.fp_bp[None] = 0

    @ti.kernel
    def bp(self):
        self.fp_bp[None] = 1

    @ti.kernel
    def interpolate_experimental_frame(self, ts: ti.i32):
        start_ix = -1
        for i in range(self.current_target_idx[None] - 1):
            if ts >= self.sim_keypoints[i] and ts < self.sim_keypoints[i + 1]:
                start_ix = i
        if start_ix != -1:
            end_ix = start_ix + 1
            if self.exp_keypoints[start_ix] == -1:
                start_ix = start_ix - 1
            elif self.exp_keypoints[start_ix + 1] == -1:
                end_ix = start_ix + 1
            exp_keypoint = self.exp_keypoints[start_ix] + (
                self.exp_keypoints[end_ix] - self.exp_keypoints[start_ix]
            ) * (ts - self.sim_keypoints[start_ix]) / (
                self.sim_keypoints[end_ix] - self.sim_keypoints[start_ix]
            )
            for i in range(self.marker_position_exp.shape[0]):
                for j in range(2):
                    self.marker_position_exp[i][j] = self.marker_positions_exp[
                        ti.floor(exp_keypoint, dtype=ti.i32), i
                    ][j] + (exp_keypoint - ti.floor(exp_keypoint)) * (
                        self.marker_positions_exp[
                            ti.ceil(exp_keypoint, dtype=ti.i32), i
                        ][j]
                        - self.marker_positions_exp[
                            ti.floor(exp_keypoint, dtype=ti.i32), i
                        ][j]
                    )

    def load_system_identification_data(self):
        with open(SYSTEM_PARAMS.files.markers_paired, "rb") as f:
            markers_array = pickle.load(f)
        num_frames, num_markers, _ = markers_array.shape
        self.marker_positions_exp = ti.Vector.field(
            2, dtype=ti.f32, shape=(num_frames, num_markers), needs_grad=False
        )
        self.marker_positions_exp.from_numpy(markers_array)
        self.marker_position_exp = ti.Vector.field(
            2, dtype=ti.f32, shape=(num_markers,), needs_grad=False
        )

    def compute_mapping_between_experimental_and_sim_markers(self):
        exp_markers = self.marker_positions_exp.to_numpy()[0]
        sim_markers = self.vitactip.undeformed_markers.to_numpy()
        cost_matrix = cdist(sim_markers, exp_markers, metric="sqeuclidean")
        ixs_1, ixs_2 = linear_sum_assignment(cost_matrix)
        self.sim_to_exp_markers = ti.field(
            dtype=ti.i32, shape=(len(sim_markers),), needs_grad=False
        )
        mapping = np.full(len(sim_markers), -1, dtype=np.int32)
        for ix_1, ix_2 in zip(ixs_1, ixs_2):
            mapping[ix_1] = ix_2
        self.sim_to_exp_markers.from_numpy(mapping)

    def set_up_loss_computation(self):
        self.loss = ti.field(float, (), needs_grad=True)
        self.total_loss = ti.field(float, (), needs_grad=False)
        self.target_marker_positions = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        target_marker_positions_npy = np.ones(
            shape=(self.vitactip.num_markers, 2), dtype=float
        )
        self.target_marker_positions.from_numpy(target_marker_positions_npy)
        self.squared_error_sum = ti.field(dtype=float, shape=(), needs_grad=True)

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
            needs_grad=False,
        )

    def set_up_system_params(self):
        self.dt = ti.field(dtype=float, shape=(), needs_grad=False)
        self.dt[None] = SYSTEM_PARAMS.contact.dt
        self.normal_stiffness = ti.field(dtype=float, shape=(), needs_grad=True)
        self.normal_damping = ti.field(dtype=float, shape=(), needs_grad=True)
        self.tangential_stiffness = ti.field(dtype=float, shape=(), needs_grad=True)
        self.coulomb_friction_coeff = ti.field(dtype=float, shape=(), needs_grad=True)
        self.normal_stiffness[None] = SYSTEM_PARAMS.contact.normal_stiffness
        self.normal_damping[None] = SYSTEM_PARAMS.contact.normal_damping
        self.tangential_stiffness[None] = SYSTEM_PARAMS.contact.tangential_stiffness
        self.coulomb_friction_coeff[None] = SYSTEM_PARAMS.contact.coulomb_friction_coeff
        self.gradients_printed = False

    def set_up_snapshot(self):
        self.predict_markers_snapshots = ti.Vector.field(
            2,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_opt_steps, self.vitactip.num_markers),
            needs_grad=False,
        )
        self.virtual_markers_snapshots = ti.Vector.field(
            2,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_opt_steps, self.vitactip.num_markers),
            needs_grad=False,
        )
        self.ground_truth_labels = ti.field(
            dtype=int, shape=(SYSTEM_PARAMS.contact.num_opt_steps,), needs_grad=False
        )

    def set_up_pid(self):
        self.pos_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.prev_pos_error = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.current_target_idx = ti.field(dtype=int, shape=(), needs_grad=False)
        self.current_target_idx[None] = 0
        self.ori_error_magnitude_degrees = ti.field(
            dtype=float, shape=(), needs_grad=False
        )
        self.dwell_frames = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_frames[None] = SYSTEM_PARAMS.contact.dwell_frames
        self.dwell_counter = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_counter[None] = 0
        self.is_dwelling = ti.field(dtype=int, shape=(), needs_grad=False)
        self.is_dwelling[None] = 0
        self.last_target_reached = ti.field(dtype=int, shape=(), needs_grad=False)
        self.last_target_reached[None] = 0
        self.frames_since_last_target_reached = ti.field(
            dtype=int, shape=(), needs_grad=False
        )
        self.frames_since_last_target_reached[None] = 0
        self.mesh_needs_to_be_saved = ti.field(dtype=int, shape=(), needs_grad=False)
        self.mesh_needs_to_be_saved[None] = 0

    def set_up_initial_positions_and_trajectory_first_init_only(self):
        self.phantom_closest_vertex = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        self.phantom_centroid_pose = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose
        self.vitactip_tip_pose = SYSTEM_PARAMS_COMPUTED.vitactip_tip_pose
        self.min_coord = SYSTEM_PARAMS_COMPUTED.min_coord
        self.max_coord_x = SYSTEM_PARAMS_COMPUTED.max_coord_x
        self.max_coord_y = SYSTEM_PARAMS_COMPUTED.max_coord_y
        self.max_coord_z = SYSTEM_PARAMS_COMPUTED.max_coord_z
        self.tactile_sensor_initial_position = ti.Vector.field(
            3, dtype=ti.f32, shape=1, needs_grad=False
        )
        self.phantom_initial_position = ti.Vector.field(
            3, dtype=ti.f32, shape=1, needs_grad=False
        )
        self.trajectory = ti.Vector.field(7, dtype=float, shape=12, needs_grad=False)
        self.tumour_present = ti.field(dtype=int, shape=(), needs_grad=False)
        self.tumour_present[None] = 0
        self.sim_keypoints_np = -np.ones((12,))
        self.sim_keypoints = ti.field(
            dtype=int, shape=(self.sim_keypoints_np.shape[0],), needs_grad=False
        )
        self.sim_keypoints.from_numpy(self.sim_keypoints_np)
        self.exp_keypoints_np = np.array(
            [0, 36, 74, 125, -1, 164, 199, 243, -1, 344, 378, 421], dtype=np.int32
        )
        self.exp_keypoints = ti.field(
            dtype=int, shape=(self.exp_keypoints_np.shape[0],), needs_grad=False
        )
        self.exp_keypoints.from_numpy(self.exp_keypoints_np)

    def set_up_keypoints(self):
        self.keypoint_indices = np.concatenate(
            (
                self.vitactip.get_keypoint_indices(0),
                self.phantom.get_keypoint_index(),
            ),
            dtype=int,
        )

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
        x, y, z = self.vitactip_tip_pose[:3]
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        offset_1 = R.from_euler(seq="xyz", angles=[20, 0, 0], degrees=True)
        offset_2 = R.from_euler(seq="xyz", angles=[0, 0, -20], degrees=True)
        tilt_1 = og_r * offset_1
        tilt_2 = og_r * offset_2
        tilt_1.as_quat()
        press_depth_1 = SYSTEM_PARAMS.geometry.gap
        press_depth_2 = SYSTEM_PARAMS.geometry.gap + 0.006
        self.trajectory_npy = np.array(
            [
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_1, *og_r.as_quat()],
                [x, y, z - press_depth_2, *og_r.as_quat()],
                [x - 0.010, y, z - press_depth_2, *og_r.as_quat()],
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_1, *og_r.as_quat()],
                [x, y, z - press_depth_2, *og_r.as_quat()],
                [x, y, z - press_depth_2, *tilt_1.as_quat()],
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_1, *og_r.as_quat()],
                [x, y, z - press_depth_2, *og_r.as_quat()],
                [x, y, z - press_depth_2, *tilt_2.as_quat()],
            ],
            dtype=float,
        )
        assert self.trajectory.shape[0] == self.trajectory_npy.shape[0], (
            f"Set self.trajectory length to {self.trajectory_npy.shape[0]} match trajectory_npy"
        )
        self.trajectory.from_numpy(self.trajectory_npy)
        sensor_dome_tip_initial_pose = self.trajectory_npy[0].tolist()
        sensor_dome_tip_initial_pose[2] += camera_lens_to_sensor_tip
        sensor_dome_tip_initial_pose = np.array(sensor_dome_tip_initial_pose)
        self.vitactip.set_up_pose(sensor_dome_tip_initial_pose)
        self.tactile_sensor_initial_position[0] = ti.Vector(
            sensor_dome_tip_initial_pose[:3]
        )
        self.phantom_initial_position[0] = ti.Vector(self.phantom_centroid_pose[:3])

    def reset_pid_controller(self):
        self.pos_error_sum.fill(0)
        self.prev_pos_error.fill(0)
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

    def update_grad(self, f):
        self.vitactip.update_external_forces.grad(f)
        self.phantom.g2p.grad(f)
        self.phantom.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.vitactip.update_internal_forces.grad(f)
        self.phantom.p2g.grad(f)
        # self.phantom.svd_of_trial_deformation_gradient_grad(f)
        self.phantom.compute_trial_deformation_gradient.grad(f)

    @ti.kernel
    def clamp_grid(self, f: ti.i32):
        for i in range(self.vitactip.num_vertices):
            self.vitactip.vertices_deformed_A.grad[f, i] = ti.math.clamp(
                self.vitactip.vertices_deformed_A.grad[f, i], -1000.0, 1000.0
            )
            self.vitactip.vertex_velocities.grad[f, i] = ti.math.clamp(
                self.vitactip.vertex_velocities.grad[f, i], -1000.0, 1000.0
            )
        return
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x,
            SYSTEM_PARAMS.phantom.n_grid_y,
            SYSTEM_PARAMS.phantom.n_grid_z,
        ):
            self.phantom.grid_node_mass.grad[f, i, j, k] = ti.math.clamp(
                self.phantom.grid_node_mass.grad[f, i, j, k], -1000.0, 1000.0
            )

    @ti.kernel
    def reset_loss(self):
        self.total_loss[None] += self.loss[None]
        self.loss.fill(0.0)
        self.loss.grad.fill(1.0)

    @ti.kernel
    def clear_grad_helper(self):
        self.squared_error_sum.grad.fill(0.0)
        self.normal_stiffness.grad.fill(0.0)
        self.normal_damping.grad.fill(0.0)
        self.tangential_stiffness.grad.fill(0.0)
        self.coulomb_friction_coeff.grad.fill(0.0)

    def clear_grad(self):
        self.clear_grad_helper()
        self.vitactip.clear_grad()
        self.phantom.clear_grad()

    @ti.kernel
    def reset(self):
        self.squared_error_sum.fill(0.0)
        self.contact_idx.fill(-1)
        self.vitactip.reset()
        self.phantom.reset()

    @ti.kernel
    def compute_marker_loss_1(self, f: ti.i32):
        for i in range(self.vitactip.num_markers):
            exp_ix = self.sim_to_exp_markers[i]
            if exp_ix != -1:
                sim_marker = self.vitactip.deformed_markers[i]
                exp_marker = self.marker_position_exp[exp_ix]
                dx = exp_marker[0] - sim_marker[0]
                dy = exp_marker[1] - sim_marker[1]
                dx /= self.window_size[None][0]
                dy /= self.window_size[None][1]
                squared_error = dx * dx + dy * dy
                self.squared_error_sum[None] += squared_error

    @ti.kernel
    def compute_marker_loss_2(self, f: ti.i32):
        rmse = ti.sqrt(self.squared_error_sum[None] / self.marker_position_exp.shape[0])
        self.loss[None] += rmse

    @ti.func
    def calculate_contact_force(
        self, signed_distance, surface_normal, relative_velocity
    ):
        tangential_force = ti.Vector([0.0, 0.0, 0.0])
        tangential_velocity = ti.Vector([0.0, 0.0, 0.0])
        contact_relative_velocity = relative_velocity
        normal_velocity_magnitude = ti.max(
            surface_normal.dot(contact_relative_velocity), 0
        )
        normal_force = (
            -(
                self.normal_stiffness[None]
                + self.normal_damping[None] * normal_velocity_magnitude
            )
            * signed_distance
            * surface_normal
        )
        tangential_velocity = (
            contact_relative_velocity
            - surface_normal.dot(contact_relative_velocity) * surface_normal
        )
        tangential_velocity_magnitude = tangential_velocity.norm(
            SYSTEM_PARAMS.contact.norm_eps
        )
        if (
            tangential_velocity_magnitude
            > SYSTEM_PARAMS.contact.tangential_velocity_detection_threshold
        ):
            tangential_force = (
                1.0
                * (tangential_velocity / tangential_velocity_magnitude)
                * ti.min(
                    self.tangential_stiffness[None] * tangential_velocity_magnitude,
                    self.coulomb_friction_coeff[None]
                    * normal_force.norm(SYSTEM_PARAMS.contact.norm_eps),
                )
            )
        total_contact_force = normal_force + tangential_force
        return total_contact_force, normal_force, tangential_force

    @ti.kernel
    def check_collision(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x,
            SYSTEM_PARAMS.phantom.n_grid_y,
            SYSTEM_PARAMS.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                closest_sensor_vertex_idx = self.vitactip.find_closest(
                    grid_node_position, frame
                )
                self.contact_idx[frame, i, j, k] = closest_sensor_vertex_idx

    @ti.kernel
    def collision(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x,
            SYSTEM_PARAMS.phantom.n_grid_y,
            SYSTEM_PARAMS.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                grid_node_velocity = self.phantom.grid_node_momentum_in[
                    frame, i, j, k
                ] / (
                    self.phantom.grid_node_mass[frame, i, j, k]
                    + SYSTEM_PARAMS.phantom.eps
                )
                closest_sensor_vertex_idx = self.contact_idx[frame, i, j, k]
                if closest_sensor_vertex_idx[0] != -1:
                    (
                        penetration_depth,
                        surface_normal,
                        relative_velocity,
                        is_in_contact,
                    ) = self.vitactip.find_sdf(
                        grid_node_position,
                        grid_node_velocity,
                        closest_sensor_vertex_idx,
                        frame,
                    )
                    if is_in_contact:
                        total_contact_force, _, _ = self.calculate_contact_force(
                            penetration_depth,
                            -1 * surface_normal,
                            -1 * relative_velocity,
                        )
                        self.phantom.update_contact_impulse(
                            total_contact_force, frame, i, j, k
                        )
                        self.vitactip.update_contact_force(
                            closest_sensor_vertex_idx, -1 * total_contact_force, frame
                        )

    def memory_to_cache(self, t):
        self.vitactip.memory_to_cache(t)
        self.phantom.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.vitactip.memory_from_cache(t)
        self.phantom.memory_from_cache(t)

    def pid_controller_1(self):
        self.vitactip.compute_current_orientation()
        current_ori = R.from_quat(self.vitactip.R_BA_quat.to_numpy())
        target = self.trajectory.to_numpy()[self.current_target_idx[None]]
        target_ori = R.from_quat(target[3:])
        ori_error = target_ori * current_ori.inv()
        current_angle_radians = ori_error.magnitude()
        current_axis = ori_error.as_rotvec() / (
            current_angle_radians
            if current_angle_radians > SYSTEM_PARAMS.contact.pid_angle_eps
            else 1.0
        )
        if np.isclose(current_angle_radians % (2 * np.pi), 0) or np.isclose(
            current_angle_radians % (2 * np.pi), 2 * np.pi
        ):
            current_axis = np.array([1.0, 0.0, 0.0])
        time_duration = self.dt[None] * (
            SYSTEM_PARAMS.contact.num_sub_frames - 1
        )
        rotation_per_second = (
            current_angle_radians
            * SYSTEM_PARAMS.contact.pid_orientation_kp
            / time_duration
        )
        if rotation_per_second > np.deg2rad(
            SYSTEM_PARAMS.contact.pid_max_rotation_per_second_degrees
        ):
            ori_control = R.from_rotvec(
                current_axis
                * np.deg2rad(SYSTEM_PARAMS.contact.pid_max_rotation_per_second_degrees)
                * time_duration
            )
        else:
            ori_control = R.from_rotvec(
                current_axis * rotation_per_second * time_duration
            )
        ori_control_quat = ori_control.as_quat()
        self.vitactip.R_A_quat.from_numpy(ori_control_quat)
        self.ori_error_magnitude_degrees[None] = np.rad2deg(current_angle_radians)

    @ti.kernel
    def pid_controller_2(self, ts: ti.i32):
        current_pos = self.vitactip.vertices_undeformed_A[0, self.keypoint_indices[0]]
        target = self.trajectory[self.current_target_idx[None]]
        target_pos = ti.Vector([target[0], target[1], target[2]])
        pos_error = target_pos - current_pos
        pos_error_magnitude = pos_error.norm()
        if self.last_target_reached[None] == 1:
            self.frames_since_last_target_reached[None] += 1
        if (
            self.last_target_reached[None] == 0
            and self.is_dwelling[None] == 0
            and pos_error_magnitude < SYSTEM_PARAMS.contact.pid_position_tolerance
            and self.ori_error_magnitude_degrees[None]
            < SYSTEM_PARAMS.contact.pid_orientation_tolerance
        ):
            self.is_dwelling[None] = 1
            self.dwell_counter[None] = 0
            self.sim_keypoints[self.current_target_idx[None]] = ts
            self.mesh_needs_to_be_saved[None] = 1
            print(
                f"target {self.current_target_idx[None]} ({target}) reached at time step {ts}!"
            )
        target_reached_no_control = False
        if self.is_dwelling[None] == 1:
            self.dwell_counter[None] += 1
            if self.dwell_counter[None] >= self.dwell_frames[None]:
                self.is_dwelling[None] = 0
                if self.current_target_idx[None] < self.trajectory.shape[0] - 1:
                    self.current_target_idx[None] += 1
                    self.pos_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_pos_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    target_reached_no_control = True
                else:
                    self.last_target_reached[None] = 1
        if self.is_dwelling[None] == 1 or target_reached_no_control:
            self.vitactip.translation_A[None] = ti.Vector([0.0, 0.0, 0.0])
            self.vitactip.R_A_quat[None] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        else:
            self.pos_error_sum[None] += pos_error
            pos_derivative = pos_error - self.prev_pos_error[None]
            self.prev_pos_error[None] = pos_error
            pos_control = (
                SYSTEM_PARAMS.contact.pid_kp * pos_error
                + SYSTEM_PARAMS.contact.pid_ki * self.pos_error_sum[None]
                + SYSTEM_PARAMS.contact.pid_kd * pos_derivative
            )
            max_speed_pos = SYSTEM_PARAMS.contact.pid_max_speed_translation
            pos_control_norm = pos_control.norm()
            if pos_control_norm > max_speed_pos:
                pos_control = pos_control / pos_control_norm * max_speed_pos
            if SYSTEM_PARAMS.meta.enable_pid_controller == 1:
                self.vitactip.translation_A[None] = pos_control
            else:
                self.vitactip.translation_A[None] = ti.Vector([0.0, 0.0, 0.0])
                self.vitactip.R_A_quat[None] = ti.Vector([0.0, 0.0, 0.0, 1.0])

    def pid_controller_3(self):
        self.vitactip.R_A.from_numpy(
            R.from_quat(self.vitactip.R_A_quat.to_numpy()).as_matrix()
        )

    def take_2d_markers_snapshot(self, opts):
        self.take_snapshot_1(opts)
        self.take_snapshot_2(opts)

    @ti.kernel
    def take_snapshot_1(self, opts: ti.i32):
        for i in range(self.vitactip.num_markers):
            self.predict_markers_snapshots[opts, i] = self.vitactip.deformed_markers[i]
            self.virtual_markers_snapshots[opts, i] = self.vitactip.undeformed_markers[
                i
            ]

    @ti.kernel
    def take_snapshot_2(self, opts: ti.i32):
        self.ground_truth_labels[opts] = self.tumour_present[None]

    def save_marker_data_and_ground_truth_labels_to_file(self):
        predict_np = self.predict_markers_snapshots.to_numpy()
        virtual_np = self.virtual_markers_snapshots.to_numpy()
        labels_np = self.ground_truth_labels.to_numpy()
        out = {
            "predict_markers_snapshots": predict_np,
            "virtual_markers_snapshots": virtual_np,
            "ground_truth_labels": labels_np,
        }
        with open(SYSTEM_PARAMS.files.marker_snapshots_and_labels, "wb") as f:
            pickle.dump(out, f)

        def save_markers_with_empty_rows(arr, filename):
            num_steps, num_markers, dim = arr.shape
            rows = []
            for step in range(num_steps):
                rows.append(arr[step])
                rows.append(np.full((1, dim), np.nan))
            out_arr = np.vstack(rows)
            np.savetxt(filename, out_arr, delimiter=",")

        save_markers_with_empty_rows(
            predict_np, SYSTEM_PARAMS.files.predict_markers_snapshots
        )
        save_markers_with_empty_rows(
            virtual_np, SYSTEM_PARAMS.files.virtual_markers_snapshots
        )
        np.savetxt(
            SYSTEM_PARAMS.files.ground_truth_labels, labels_np, delimiter=",", fmt="%d"
        )

    def maybe_save_tactile_sensor_mesh_to_pickle(self, ts):
        if self.mesh_needs_to_be_saved[None] == 1:
            particles = self.vitactip.vertices_deformed_A.to_numpy()[0]
            with open(
                SYSTEM_PARAMS.files.deformed_node_coordinates.format(ts), "wb"
            ) as f:
                pickle.dump(particles, f)
            self.mesh_needs_to_be_saved[None] = 0

    def save_tactile_sensor_mesh_node_mapping_to_pickle(self):
        f2v = self.vitactip.tetrahedra.to_numpy()
        with open(SYSTEM_PARAMS.files.tactile_sensor_f2v, "wb") as f:
            pickle.dump(f2v, f)

    def visualisation_initialise(self):
        self.num_keypoints = 18
        self.key_points = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.num_keypoints,), needs_grad=False
        )
        self.key_points_per_vertex_color = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.num_keypoints,), needs_grad=False
        )
        self.sensor_points = ti.Vector.field(
            3, dtype=float, shape=(self.vitactip.num_vertices), needs_grad=False
        )
        self.healthy_tissue_points = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.phantom.actual_total_num_particles,),
            needs_grad=False,
        )
        self.tumour_points = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.phantom.actual_total_num_particles,),
            needs_grad=False,
        )
        self.healthy_tissue_points_von_mises_stress = ti.field(
            dtype=float,
            shape=(self.phantom.actual_total_num_particles,),
            needs_grad=False,
        )
        self.tumour_points_von_mises_stress = ti.field(
            dtype=float,
            shape=(self.phantom.actual_total_num_particles,),
            needs_grad=False,
        )
        self.sim_markers_undeformed = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.sim_markers_deformed = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.sim_markers_deformed_filtered = ti.Vector.field(
            2, dtype=float, shape=(self.marker_position_exp.shape[0],), needs_grad=False
        )
        self.sim_marker_offsets = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.exp_marker_points = ti.Vector.field(
            2, dtype=float, shape=(self.marker_position_exp.shape[0],), needs_grad=False
        )
        self.arrow_line_vertices = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers * 2,), needs_grad=False
        )
        self.clock_arm_points = ti.Vector.field(
            2, dtype=float, shape=(2,), needs_grad=False
        )
        self.clock_arm_points_per_vertex_color = ti.Vector.field(
            3, dtype=ti.f32, shape=(2,), needs_grad=False
        )
        self.window_size = ti.Vector.field(2, dtype=float, shape=(), needs_grad=False)
        self.window_size[None] = [640.0, 480.0]
        self.fp_bp = ti.field(dtype=int, shape=(), needs_grad=False)

    @ti.kernel
    def visualisation_reset_3d_scene(self):
        self.healthy_tissue_points.fill(0)
        self.tumour_points.fill(0)

    @ti.kernel
    def visualisation_draw_3d_scene(self, f: ti.i32):
        for p in range(self.phantom.actual_total_num_particles):
            if self.phantom.titles[p] == 0:
                self.healthy_tissue_points[p] = self.phantom.particle_position[f, p]
                self.healthy_tissue_points_von_mises_stress[p] = (
                    self.phantom.particle_von_mises_stress[0, p]
                )
            elif self.phantom.titles[p] == 1:
                self.tumour_points[p] = self.phantom.particle_position[f, p]
                self.tumour_points_von_mises_stress[p] = (
                    self.phantom.particle_von_mises_stress[0, p]
                )
        for p in range(self.vitactip.num_vertices):
            self.sensor_points[p] = self.vitactip.vertices_deformed_A[f, p]

    @ti.kernel
    def visualisation_prepare_tactile_readout_data_fp(self):
        for i in range(self.vitactip.num_markers):
            undeformed = self.vitactip.undeformed_markers[i]
            deformed = self.vitactip.deformed_markers[i]
            undeformed[1] = self.window_size[None][1] - undeformed[1]
            deformed[1] = self.window_size[None][1] - deformed[1]
            undeformed = undeformed / self.window_size[None]
            deformed = deformed / self.window_size[None]
            offset = deformed - undeformed
            self.sim_markers_undeformed[i] = undeformed
            self.sim_markers_deformed[i] = deformed
            self.arrow_line_vertices[i * 2] = undeformed
            self.arrow_line_vertices[i * 2 + 1] = undeformed + offset

            exp_ix = self.sim_to_exp_markers[i]
            if exp_ix != -1:
                self.sim_markers_deformed_filtered[exp_ix] = deformed

    @ti.kernel
    def visualisation_prepare_tactile_readout_data_bp(self):
        for i in range(self.marker_position_exp.shape[0]):
            point = self.marker_position_exp[i]
            point[1] = self.window_size[None][1] - point[1]
            self.exp_marker_points[i] = point / self.window_size[None]

    @ti.kernel
    def visualisation_prepare_clock_arm_points(self):
        for i in range(2):
            point = self.vitactip.projection_2d_clock_arms[i]
            point[1] = self.window_size[None][1] - point[1]
            self.clock_arm_points[i] = point / self.window_size[None]

    def visualisation_draw_tactile_readout(self):
        self.vitactip.extract_clock_arm_2d_projections(SYSTEM_PARAMS.contact.num_sub_frames - 1)
        self.visualisation_prepare_clock_arm_points()
        self.tactile_canvas.set_image(self.bg_image)
        self.visualisation_prepare_tactile_readout_data_fp()
        if (
            self.fp_bp[None] == 0
            or SYSTEM_PARAMS.visualisation.visualise_exp_markers_during_bp == 0
        ):
            self.tactile_canvas.circles(
                self.sim_markers_deformed, radius=0.01, color=(1, 0, 0)
            )
            if False:
                self.tactile_canvas.lines(
                    self.arrow_line_vertices, color=(0, 1, 0), width=0.01
                )
            self.tactile_canvas.circles(
                self.clock_arm_points,
                radius=0.02,
                per_vertex_color=self.clock_arm_points_per_vertex_color,
            )
        else:
            self.visualisation_prepare_tactile_readout_data_bp()
            self.tactile_canvas.circles(
                self.sim_markers_deformed_filtered, radius=0.01, color=(0, 1, 0)
            )
            self.tactile_canvas.circles(
                self.exp_marker_points, radius=0.01, color=(1, 0, 0)
            )
        self.tactile_window.show()

    def visualisation_set_up_gui(self):
        screen_width = 1920
        screen_height = 1080
        self.window = ti.ui.Window(
            "high-level camera", (int(screen_width * 0.5), int(screen_height * 0.5))
        )
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0, 0, 0))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        x, y, z = self.vitactip_tip_pose[:3]
        self.camera.position(x, y-1.0, z)
        self.camera.up(0, 0, 1)
        self.camera.lookat(x, y, z)
        self.camera.fov(8)
        self.tactile_window = ti.ui.Window("tactile readout", (640, 480))
        self.tactile_canvas = self.tactile_window.get_canvas()
        self.bg_image = cv2.imread(SYSTEM_PARAMS.files.vitactip_photo_default_state)
        self.bg_image = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGB)
        self.bg_image = cv2.rotate(self.bg_image, cv2.ROTATE_90_CLOCKWISE)
        clock_arm_points_per_vertex_color_npy = np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=float,
        )
        self.clock_arm_points_per_vertex_color.from_numpy(
            clock_arm_points_per_vertex_color_npy
        )
        key_points_per_vertex_color_npy = np.tile(
            [1.0, 0.0, 0.0], (self.num_keypoints, 1)
        )
        key_points_per_vertex_color_npy[-2:, :] = clock_arm_points_per_vertex_color_npy
        self.key_points_per_vertex_color.from_numpy(key_points_per_vertex_color_npy)

    def visualisation_update_gui(self, ts):
        vitactip_coords = self.vitactip.vertices_deformed_A.to_numpy()[0]
        if np.isnan(vitactip_coords).any():
            nan_count = np.any(np.isnan(vitactip_coords), axis=1).sum()
            print(
                f"ViTacTip contains {nan_count} / {vitactip_coords.shape[0]} nan vertices at ts: {ts}"
            )
        phantom_coords = self.phantom.particle_position.to_numpy()[0]
        if np.isnan(phantom_coords).any():
            nan_count = np.any(np.isnan(phantom_coords), axis=1).sum()
            print(
                f"phantom contains {nan_count} / {phantom_coords.shape[0]} nan vertices at ts: {ts}"
            )
        move_to_the_front_offset = np.array([-0.050, 0, 0], dtype=float)
        z = self.min_coord
        x0, y0, _ = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose[:3]
        x1, y1, _ = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        floor = np.array(
            [
                [x0, y1, z],
                [x0, y0, z],
                [x0, y0 + abs(y0 - y1), z],
            ]
        )
        floor += move_to_the_front_offset
        vitactip_bottom = self.vitactip.get_keypoint_coordinates(
            0, self.keypoint_indices[0].reshape((1,))
        )
        trajectory_keypoints = self.trajectory_npy[:, :3].copy()
        vitactip_clock_arms = self.vitactip.get_keypoint_coordinates(
            f=0, keypoint_indices=self.vitactip.clock_arms_node_idxs.to_numpy()
        )
        self.keypoint_coords = np.vstack(
            (vitactip_bottom, trajectory_keypoints, floor, vitactip_clock_arms)
        )
        self.visualisation_draw_tactile_readout()
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
        assert self.keypoint_coords.shape[0] == self.key_points.shape[0], (
            f"Set self.key_points to shape ({self.keypoint_coords.shape[0]},)"
        )
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

    def forward_pass_common_part(self, ts):
        self.vitactip.set_control_vel(0)
        self.vitactip.set_vel(0)
        self.reset()
        self.vitactip.set_up_system_params_2()
        self.phantom.set_stiffness()
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 1):
            self.update(ss)

    def backward_pass_common_part(self):
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 2, -1, -1):
            self.update_grad(ss)
        self.phantom.set_stiffness.grad()
        self.vitactip.set_up_system_params_2.grad()

    def print_gradients(self, ts):
        if not self.gradients_printed:
            print(f"time step: {ts}")
            print(f"mini batch size: {SYSTEM_PARAMS.meta.mini_batch_size}")
            print(f"loss: {self.loss.grad[None]}")
            print(f"squared_error_sum (grad): {self.squared_error_sum.grad[None]}")
            print(f"squared_error_sum (val): {self.squared_error_sum[None]}")
            print()
            self.print_gradients_single(
                "deformed_markers", self.vitactip.deformed_markers
            )
            self.print_gradients_single(
                "vertices_deformed_A", self.vitactip.vertices_deformed_A
            )
            self.print_gradients_single(
                "vertex_velocities", self.vitactip.vertex_velocities
            )
            self.print_gradients_single(
                "contact_forces_on_vertices", self.vitactip.contact_forces_on_vertices
            )
            self.print_gradients_single("vitactip.mu", self.vitactip.mu)
            self.print_gradients_single("vitactip.lam", self.vitactip.lam)
            self.print_gradients_single(
                "vitactip.youngs_modulus", self.vitactip.youngs_modulus
            )
            self.print_gradients_single("normal_stiffness", self.normal_stiffness)
            self.print_gradients_single("normal_damping", self.normal_damping)
            self.print_gradients_single(
                "tangential_stiffness", self.tangential_stiffness
            )
            self.print_gradients_single(
                "coulomb_friction_coeff", self.coulomb_friction_coeff
            )
            self.print_gradients_single("phantom.mu", self.phantom.mu)
            self.print_gradients_single("phantom.lam", self.phantom.lam)
            self.print_gradients_single(
                "phantom.youngs_modulus", self.phantom.youngs_modulus
            )
            print()
            self.gradients_printed = True

    def print_gradients_single(self, name, ti_var):
        grad_npy = ti_var.grad.to_numpy()
        val_npy = ti_var.to_numpy()
        print(f"{name} (grad) min: {grad_npy.min()}")
        print(f"{name} (grad) max: {grad_npy.max()}")
        print(f"{name} (val) min: {val_npy.min()}")
        print(f"{name} (val) max: {val_npy.max()}")
        print()

    def update_params(self, ts):
        print(f'ts: {ts}')
        print(f'loss: {self.loss[None]}')
        self.update_param_none(
            self.vitactip.youngs_modulus, 
            ['vitactip', 'youngs_modulus']
        )
        for i in range(2):
            self.update_param_indexed(
                self.phantom.youngs_modulus,
                ['phantom', 'youngs_modulus'],
                i
            )
        self.update_param_none(
            self.coulomb_friction_coeff,
            ['contact', 'coulomb_friction_coeff']
        )
        self.update_param_none(
            self.normal_stiffness,
            ['contact', 'normal_stiffness']
        )
        self.update_param_none(
            self.tangential_stiffness,
            ['contact', 'tangential_stiffness']
        )
        self.update_param_none(
            self.normal_damping,
            ['contact', 'normal_damping']
        )
        print()
    
    def update_param_none(self, ti_var, keys):
        min_val = SYSTEM_PARAMS.optimisation_min_values
        max_val = SYSTEM_PARAMS.optimisation_max_values
        learning_rate = SYSTEM_PARAMS_COMPUTED.learning_rates
        for i in range(len(keys)):
            min_val = min_val[keys[i]]
            max_val = max_val[keys[i]]
            learning_rate = learning_rate[keys[i]]

        update = - (
            ti_var.grad[None]
            * learning_rate
        )
        ti_var[None] += update
        ti_var[None] = ti.min(max_val, ti.max(min_val, ti_var[None]))
        
        param_name = '.'.join(keys)
        print(f'{param_name}: {ti_var[None]}')
        print(f'{param_name}.update: {update}')

    def update_param_indexed(self, ti_var, keys, idx):
        min_val = SYSTEM_PARAMS.optimisation_min_values
        max_val = SYSTEM_PARAMS.optimisation_max_values
        learning_rate = SYSTEM_PARAMS_COMPUTED.learning_rates
        for i in range(len(keys)):
            min_val = min_val[keys[i]]
            max_val = max_val[keys[i]]
            learning_rate = learning_rate[keys[i]]

        update = - (
            ti_var.grad[idx]
            * learning_rate
        )
        ti_var[idx] += update
        ti_var[idx] = ti.min(max_val, ti.max(min_val, ti_var[idx]))
        
        param_name = '.'.join(keys)
        print(f'{param_name}[{idx}]: {ti_var[idx]}')
        print(f'{param_name}.update: {update}')
    
    def save_final_params(self):
        results = dict()
        results["vitactip"] = {
            "youngs_modulus": float(self.vitactip.youngs_modulus[None])
        }
        results["phantom"] = {
            "youngs_modulus": [float(self.phantom.youngs_modulus[i]) for i in range(2)]
        }
        results["contact"] = {
            "coulomb_friction_coeff": float(self.coulomb_friction_coeff[None]),
            "normal_stiffness": float(self.normal_stiffness[None]),
            "tangential_stiffness": float(self.tangential_stiffness[None]),
            "normal_damping": float(self.normal_damping[None])
        }
        with open(SYSTEM_PARAMS.files.domain_adaptation_results, "w") as f:
            json.dump(results, f, indent=4)
    
    def set_dt(self):
        dt = calculate_cfl_timestep(
            phantom_youngs_modulus=self.phantom.youngs_modulus[None],
            vitactip_youngs_modulus=self.vitactip.youngs_modulus[None],
            verbose=False,
        )
        self.dt[None] = dt
        self.phantom.dt[None] = dt
        self.vitactip.dt[None] = dt


def main():
    if RUN_ON_LAB_MACHINE:
        ti.init(
            debug=False,
            offline_cache=False,
            log_level=ti.ERROR,
            arch=ti.cuda,
            device_memory_GB=9,
        )
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)
    contact_model = Contact()
    contact_model.visualisation_set_up_gui()
    contact_model.save_tactile_sensor_mesh_node_mapping_to_pickle()
    for opts in range(SYSTEM_PARAMS.contact.num_opt_steps):
        print(f"optimisation step: {opts} / {SYSTEM_PARAMS.contact.num_opt_steps - 1}")
        contact_model.set_up_initial_positions_and_trajectory()
        contact_model.reset_pid_controller()
        contact_model.visualisation_reset_3d_scene()
        if opts == 0:
            contact_model.vitactip.test_mapping_from_global_space_to_camera_space()
            contact_model.vitactip.extract_markers(0)
            contact_model.compute_mapping_between_experimental_and_sim_markers()
            contact_model.vitactip.save_predicted_markers_to_image()
            contact_model.vitactip.get_keypoint_idxs()
            initial_markers = contact_model.vitactip.deformed_markers.to_numpy()
            with open(SYSTEM_PARAMS.files.sim_markers_initial_positions, "wb") as f:
                pickle.dump(initial_markers, f)
        contact_model.set_dt()
        contact_model.fp()
        print("forward")
        for ts in range(SYSTEM_PARAMS.contact.num_frames):
            contact_model.pid_controller_1()
            contact_model.pid_controller_2(ts)
            contact_model.pid_controller_3()
            contact_model.vitactip.set_pose_control_1()
            contact_model.vitactip.set_pose_control_2()
            contact_model.vitactip.set_pose_control_3()
            contact_model.forward_pass_common_part(ts)
            contact_model.memory_to_cache(ts)
            contact_model.vitactip.extract_markers(
                SYSTEM_PARAMS.contact.num_sub_frames - 1
            )
            contact_model.visualisation_update_gui(ts)
            contact_model.maybe_save_tactile_sensor_mesh_to_pickle(ts)
        contact_model.bp()
        contact_model.total_loss.fill(0.0)
        contact_model.clear_grad()
        print("backward")
        for ts in range(SYSTEM_PARAMS.contact.num_frames - 1, -1, -1):
            contact_model.reset_loss()
            contact_model.memory_from_cache(ts)
            contact_model.forward_pass_common_part(ts)
            contact_model.vitactip.extract_markers(
                SYSTEM_PARAMS.contact.num_sub_frames - 1
            )
            contact_model.interpolate_experimental_frame(ts)
            contact_model.compute_marker_loss_1(ts)
            contact_model.compute_marker_loss_2(ts)
            contact_model.visualisation_update_gui(ts)
            contact_model.compute_marker_loss_2.grad(ts)
            contact_model.compute_marker_loss_1.grad(ts)
            contact_model.vitactip.extract_markers.grad(
                SYSTEM_PARAMS.contact.num_sub_frames - 1
            )
            contact_model.backward_pass_common_part()
            passes = SYSTEM_PARAMS.contact.num_frames - 1 - ts + 1
            if passes % (SYSTEM_PARAMS.contact.num_frames // 3) == 0:
                print(f'passes: {passes}')
                contact_model.update_params(ts)
                contact_model.clear_grad()
        print(
            f"optimisation step: {opts} / {SYSTEM_PARAMS.contact.num_opt_steps - 1} done; loss: {contact_model.total_loss[None]}"
        )
    print("optimisation loop done")
    contact_model.save_final_params()
    print("all done")
