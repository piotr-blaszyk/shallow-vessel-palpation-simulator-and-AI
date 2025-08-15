import taichi as ti
import numpy as np
import pickle
import json
import cv2
import sys
import os
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import math
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import random

from difftactile.sensor_model.fisheye_model import *
from difftactile.sensor_model.vitactip import ViTacTip
from difftactile.object_model.phantom import Phantom
from difftactile.main.constants import *
from difftactile.main.cfl_and_contact_params_estimation import *
from difftactile.main.apply_scaling import ScientificNotationEncoder

RUN_ON_LAB_MACHINE = True


class SyntheticImageGenerator:
    def __init__(self):
        pass

    @staticmethod
    def crop(points):
        # Create a copy and apply the offset to all points at once
        cropped_points = points.copy()
        cropped_points[..., 0] = points[..., 0] - SYSTEM_PARAMS.fisheye_model.crop_x
        cropped_points[..., 1] = points[..., 1] - SYSTEM_PARAMS.fisheye_model.crop_y
        
        # Create mask for valid points using vectorized operations
        mask = (
            (cropped_points[..., 0] >= 0) & (cropped_points[..., 0] < SYSTEM_PARAMS.fisheye_model.crop_width)
            & (cropped_points[..., 1] >= 0) & (cropped_points[..., 1] < SYSTEM_PARAMS.fisheye_model.crop_height)
        )
        
        return cropped_points, mask

    @staticmethod
    def alpha_shape(points):
        points = np.unique(points, axis=0)
        alpha=1e-10
        if len(points) < 4:
            return np.array([])

        tri = Delaunay(points)
        edges = set()

        for ia, ib, ic in tri.simplices:
            pa, pb, pc = points[ia], points[ib], points[ic]
            a = np.linalg.norm(pa - pb)
            b = np.linalg.norm(pb - pc)
            c = np.linalg.norm(pc - pa)
            s = (a + b + c) / 2.0
            area = max(s * (s - a) * (s - b) * (s - c), 1e-10) ** 0.5
            circum_r = a * b * c / (4.0 * area)

            if circum_r < 1.0 / alpha:
                edges.update([(ia, ib), (ib, ic), (ic, ia)])

        edge_segments = [ (points[i], points[j]) for i, j in edges ]
        m = MultiLineString(edge_segments)
        triangles = list(polygonize(m))
        concave = unary_union(triangles)

        if len(edge_segments) == 0:
            return np.array([])

        if isinstance(concave, Polygon):
            return np.array(concave.exterior.coords)
        else:
            return np.array(concave.geoms[0].exterior.coords)
    
    @staticmethod
    def filter_points(w, h, cx, cy, r, points):
        points_filtered = []
        for point in points:
            x, y = point
            if (0 <= x < w and 0 <= y < h and
                ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2):
                points_filtered.append([x, y])
        points_filtered = np.array(points_filtered, dtype=float)
        return points_filtered
    
    @staticmethod
    def filter_points_vectorised(cx, cy, r, points):
        # Extract x and y coordinates from the last dimension
        x = points[..., 0]
        y = points[..., 1]
        
        # Check circle condition
        in_circle = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
        
        # Combine conditions
        return in_circle


@ti.data_oriented
class Contact:
    def __init__(self):
        self.compute_sensor_bounds()
        self.fisheye_model = FisheyeModel()
        self.set_up_system_params()
        self.load_system_identification_data()
        self.vitactip = ViTacTip()
        self.phantom = Phantom()
        self.set_up_initial_positions_and_trajectory_first_init_only()
        self.set_up_trajectories_and_phantom_states()
        self.set_up_initial_positions_state_and_trajectory()
        self.set_up_keypoints()
        self.set_up_collision_detection()
        self.set_up_pid()
        self.set_up_snapshot()
        self.set_up_loss_computation()
        self.visualisation_initialise()
        self.training_data_collection_initialise()
    
    def compute_sensor_bounds(self):
        _min = np.array(SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex[:2])
        _mid = np.array(SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose[:2])
        phantom_r = np.abs(_mid - _min)
        _max = _min + phantom_r * 2
        sensor_r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        sensor_min = _min + sensor_r
        sensor_max = _max - sensor_r

        self.sensor_x_range_world = np.array([
            sensor_min[0],
            sensor_max[0]
        ])
        self.sensor_y_range_world = np.array([
            sensor_min[1],
            sensor_max[1]
        ])

        self.sensor_x_range_phantom = self.sensor_x_range_world.copy()
        self.sensor_x_range_phantom -= _mid[0]
        self.sensor_y_range_phantom = self.sensor_y_range_world.copy()
        self.sensor_y_range_phantom -= _mid[1]

    def training_data_collection_initialise(self):
        self.marker_data = []
        self.vein_data = []
        self.vein_mask_data = []
        self.vein_endpoints_data = []
        self.vein_endpoints_mask_data = []
        self.vein_cx_A = None
        self.target_3_ts = 12
        self.target_4_ts = 226

    @ti.kernel
    def fp(self):
        self.fp_bp[None] = 0

    @ti.kernel
    def bp(self):
        self.fp_bp[None] = 1

    def load_system_identification_data(self):
        self.load_system_identification_data_1()
        self.load_system_identification_data_2()
        self.load_system_identification_data_3()

    def load_system_identification_data_1(self):
        self.exp_marker_shapes_np = np.zeros(shape=(4, 3), dtype=int)
        for i in range(4):
            with open(SYSTEM_PARAMS.files.traj_markers.format(i), "rb") as f:
                markers_array = pickle.load(f)
            self.exp_marker_shapes_np[i] = markers_array.shape
        
        self.exp_marker_shapes = ti.Vector.field(3, dtype=int, shape=(self.exp_marker_shapes_np.shape[0]))
        self.exp_marker_shapes.from_numpy(self.exp_marker_shapes_np)
        max_0, max_1, max_2 = self.exp_marker_shapes_np.max(axis=0)
        self.exp_markers_max_shapes_np = np.array([max_0, max_1, max_2])
        self.exp_markers_max_shapes = ti.field(dtype=int, shape=(3,), needs_grad=False)
        self.exp_markers_max_shapes.from_numpy(self.exp_markers_max_shapes_np)
        self.exp_markers_np = -np.ones(shape=(4, max_0, max_1, 2), dtype=float)

        for i in range(4):
            with open(SYSTEM_PARAMS.files.traj_markers.format(i), "rb") as f:
                markers_array = pickle.load(f)
            x, y, z = markers_array.shape
            self.exp_markers_np[i, :x, :y, :z] = markers_array

        self.exp_markers = ti.Vector.field(2, dtype=float, shape=(4, max_0, max_1), needs_grad=False)
        self.marker_position_exp = ti.Vector.field(2, dtype=float, shape=(max_1,), needs_grad=False)
        self.sim_to_exp_markers = ti.field(dtype=int, shape=(127,), needs_grad=False)
        self.exp_to_sim_markers = ti.field(dtype=int, shape=(127,), needs_grad=False)

        self.traj_1_exp_marker_pairs_np = np.array([
            [104, 105],
            [105, 67],
            [67, 46],
            [46, 68],
            [99, 79],
            [79, 43],
            [43, 44],
            [44, 66],
            [66, 16],
            [41, 98],
            [98, 63],
            [63, 78],
            [78, 42]
        ])
        self.traj_1_exp_marker_pairs = ti.Vector.field(
            2, 
            dtype=int, 
            shape=(self.traj_1_exp_marker_pairs_np.shape[0],),
            needs_grad=False
        )
        self.traj_1_exp_marker_pairs.from_numpy(self.traj_1_exp_marker_pairs_np)
        self.traj_1_critical_frames_exp_np = np.array([175, 231], dtype=int)
        self.traj_1_critical_frames_exp = ti.field(
            dtype=int,
            shape=(self.traj_1_critical_frames_exp_np.shape[0],),
            needs_grad=False
        )
        self.traj_1_critical_frames_exp.from_numpy(self.traj_1_critical_frames_exp_np)
        self.cur_exp_frame = ti.field(
            dtype=float,
            shape=(),
            needs_grad=False
        )
    
    @ti.kernel
    def load_system_identification_data_2(self):
        self.exp_markers.fill(-1)
        self.marker_position_exp.fill(-1)
        self.sim_to_exp_markers.fill(-1)
        self.exp_to_sim_markers.fill(-1)
    
    def load_system_identification_data_3(self):
        self.exp_markers.from_numpy(self.exp_markers_np)

    def compute_mapping_between_experimental_and_sim_markers(self):
        x, y, z = self.exp_marker_shapes_np[self.trajectory_ix[None]]
        exp_markers = self.exp_markers.to_numpy()[self.trajectory_ix[None], 0, :y, :z]
        sim_markers = self.vitactip.undeformed_markers.to_numpy()
        assert sim_markers.shape[0] == 127
        cost_matrix = cdist(sim_markers, exp_markers, metric="sqeuclidean")
        ixs_1, ixs_2 = linear_sum_assignment(cost_matrix)
        self.sim_to_exp_markers_np = np.full(127, -1, dtype=np.int32)
        for ix_1, ix_2 in zip(ixs_1, ixs_2):
            self.sim_to_exp_markers_np[ix_1] = ix_2
        self.sim_to_exp_markers.from_numpy(self.sim_to_exp_markers_np)
        self.exp_to_sim_markers_np = -np.ones_like(self.sim_to_exp_markers_np)
        for i in range(self.sim_to_exp_markers_np.shape[0]):
            exp = self.sim_to_exp_markers_np[i]
            if exp != -1:
                self.exp_to_sim_markers_np[exp] = i
        self.exp_to_sim_markers.from_numpy(self.exp_to_sim_markers_np)
    
    @ti.func
    def project_point_on_line(self, point, line_point1, line_point2):
        line_vector = line_point2 - line_point1
        point_vector = point - line_point1
        line_direction = line_vector / ti.math.length(line_vector)
        projection_length = ti.math.dot(point_vector, line_direction)
        projected_point = line_point1 + projection_length * line_direction
        return projected_point

    @ti.kernel
    def interpolate_experimental_frame(self, ts: ti.i32):
        start_ix = -1
        end_ix = -1
        # 5 target points
        # current_target_idx = 4
        # for i in range(4); i = 0,1,2,3
        # 0,1
        # 1,2
        # 2,3
        # 3,4
        for i in range(self.current_target_idx[None]):
            if ts == self.sim_keypoints[i]:
                start_ix = i
            elif ts > self.sim_keypoints[i] and ts < self.sim_keypoints[i + 1]:
                start_ix = i
                end_ix = start_ix + 1
            elif ts == self.sim_keypoints[i+1]:
                start_ix = i+1
        # if sim entry is invalid, it's -1
        # if exp entry is invalid, it's -1
        cur_exp_keypoints = self.exp_keypoints[self.trajectory_ix[None]]

        if (
            start_ix != -1 
            and cur_exp_keypoints[start_ix] != -1
        ):
            exp_keypoint = -1.0
            if (
                False
                and self.trajectory_ix[None] == 1
                and start_ix >= 3
            ):
                target = self.phantom.particles_A[
                    SYSTEM_PARAMS.contact.num_sub_frames - 1,
                    self.vein_endpoints_indices[0]
                ]
                x_E = self.exp_vein_3d_coords_E[0]
                y_E = self.exp_vein_3d_coords_E[1]
                x_A = self.vitactip.project_E_to_A(x_E)
                y_A = self.vitactip.project_E_to_A(y_E)
                target_projected = self.project_point_on_line(
                    target,
                    x_A,
                    y_A
                )

                min_dist = SYSTEM_PARAMS.geometry.high_dist
                min_ix = -1
                for i in range(self.exp_vein_3d_coords_E_all.shape[0]):
                    exp_vein_point_E = self.exp_vein_3d_coords_E_all[i]
                    if ti.math.length(
                        exp_vein_point_E
                        - ti.Vector([-1.0, -1.0, -1.0], dt=float)
                    ) > 1e-6:
                        exp_vein_point_A = self.vitactip.project_E_to_A(exp_vein_point_E)
                        dist = ti.math.length(
                            exp_vein_point_A
                            - target_projected
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_ix = i
                min_dist_0 = min_dist
                min_ix_0 = min_ix
                min_point_A_0 = self.vitactip.project_E_to_A(
                    self.exp_vein_3d_coords_E_all[min_ix_0]
                )

                min_dist = SYSTEM_PARAMS.geometry.high_dist
                min_ix = -1
                for i in range(self.exp_vein_3d_coords_E_all.shape[0]):
                    exp_vein_point_E = self.exp_vein_3d_coords_E_all[i]
                    if (
                        ti.math.length(
                        exp_vein_point_E
                        - ti.Vector([-1.0, -1.0, -1.0], dt=float)
                        ) > 1e-6
                        and i != min_ix_0
                    ):
                        exp_vein_point_A = self.vitactip.project_E_to_A(exp_vein_point_E)
                        dist = ti.math.length(
                            exp_vein_point_A
                            - target_projected
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_ix = i
                min_dist_1 = min_dist
                min_ix_1 = min_ix
                min_point_A_1 = self.vitactip.project_E_to_A(
                    self.exp_vein_3d_coords_E_all[min_ix_1]
                )
                
                # assert abs(min_ix_0 - min_ix_1) == 1, f"Interpolation video frames aren't consecutive ({min_ix_0}, {min_ix_1})"
                dist_sum = min_dist_0 + min_dist_1
                a = -1
                b = -1
                a_dist = -1.0
                b_dist = -1.0
                if min_ix_0 < min_ix_1:
                    a = min_ix_0
                    b = min_ix_1
                    a_dist = min_dist_0
                    b_dist = min_dist_1
                else:
                    a = min_ix_1
                    b = min_ix_0
                    a_dist = min_dist_1
                    b_dist = min_dist_0

                offset = a_dist / dist_sum
                exp_keypoint = a + offset

                dist_exp_veins = ti.math.length(
                    min_point_A_1
                    - min_point_A_0
                )

                if (
                    min_dist_0 < dist_exp_veins
                    and min_dist_1 < dist_exp_veins
                ):
                    self.vein_ix_base[None] = a
                    self.vein_ix_offset[None] = offset
                    self.interpolation_valid[None] = 1
                else:
                    self.vein_ix_base[None] = -1
                    self.vein_ix_offset[None] = -1.0
                    self.interpolation_valid[None] = 0

                if False:
                    print(f'target: {target}')
                    print(f'x_E: {x_E}')
                    print(f'y_E: {y_E}')
                    print(f'x_A: {x_A}')
                    print(f'y_A: {y_A}')
                    print(f'target_projected: {target_projected}')
                    print(f'min_dist_0: {min_dist_0}')
                    print(f'min_ix_0: {min_ix_0}')
                    print(f'min_dist_1: {min_dist_1}')
                    print(f'min_ix_1: {min_ix_1}')
                    print(f'dist_sum: {dist_sum}')
                    print(f'a: {a}')
                    print(f'b: {b}')
                    print(f'a_dist: {a_dist}')
                    print(f'b_dist: {b_dist}')
                    print(f'offset: {offset}')
                    print(f'exp_keypoint: {exp_keypoint}')
            else:
                if end_ix != -1:
                    exp_keypoint = (
                        cur_exp_keypoints[start_ix] 
                        + (
                            cur_exp_keypoints[end_ix]
                            - cur_exp_keypoints[start_ix]
                        ) * (
                            ts - self.sim_keypoints[start_ix]
                        ) / (
                            self.sim_keypoints[end_ix] - self.sim_keypoints[start_ix]
                        )
                    )
                else:
                    exp_keypoint = cur_exp_keypoints[start_ix]
            if self.interpolation_valid[None] == 1:
                self.cur_exp_frame[None] = exp_keypoint
            for i in range(self.exp_marker_shapes[self.trajectory_ix[None]][1]):
                for j in range(2):
                    if self.interpolation_valid[None] == 1:
                        if end_ix != -1:
                            self.marker_position_exp[i][j] = (
                                self.exp_markers[self.trajectory_ix[None], ti.floor(exp_keypoint, dtype=ti.i32), i][j] 
                                + (exp_keypoint - ti.floor(exp_keypoint)) 
                                * (
                                    self.exp_markers[self.trajectory_ix[None], ti.ceil(exp_keypoint, dtype=ti.i32), i][j]
                                    - self.exp_markers[self.trajectory_ix[None], ti.floor(exp_keypoint, dtype=ti.i32), i][j]
                                )
                            )
                        else:
                            self.marker_position_exp[i][j] = (
                                self.exp_markers[self.trajectory_ix[None], ti.floor(exp_keypoint, dtype=ti.i32), i][j]
                            )
                    else:
                        self.marker_position_exp[i][j] = -1.0

    @ti.kernel
    def compute_vein_exp_vis(self):
        start_ix = self.vein_ix_base[None]
        if start_ix != -1:
            end_ix = start_ix + 1
            offset = self.vein_ix_offset[None]
            start_E = self.exp_vein_3d_coords_E_all[start_ix]
            end_E = self.exp_vein_3d_coords_E_all[end_ix]
            start_A = self.vitactip.project_E_to_A(start_E)
            end_A = self.vitactip.project_E_to_A(end_E)
            point = start_A + offset * (end_A - start_A)
            self.vein_exp_vis[None] = point
    
    @ti.kernel
    def compute_vein_exp_vis_all(self):
        for i in range(self.exp_vein_3d_coords_E_all.shape[0]):
            point_E = self.exp_vein_3d_coords_E_all[i]
            if ti.math.length(
                point_E
                - ti.Vector([-1.0, -1.0, -1.0], dt=float)
            ) > 1e-6:
                point_A = self.vitactip.project_E_to_A(point_E)
                self.vein_exp_vis_all[i] = point_A
   
    @ti.kernel
    def compute_validation_point(self):
        point_E = self.validation_point_3d_E[None]
        point_A = self.vitactip.project_E_to_A(point_E)
        self.validation_point_3d_A[None] = point_A

    def set_up_loss_computation(self):
        self.prev_loss = ti.field(float, (), needs_grad=False)
        self.loss = ti.field(float, (), needs_grad=True)
        self.loss_1 = ti.field(float, (), needs_grad=True)
        self.loss_2 = ti.field(float, (), needs_grad=True)
        self.trajectory_loss = ti.field(float, (), needs_grad=False)
        self.batch_loss = ti.field(float, (), needs_grad=False)
        self.batch_loss_1 = ti.field(float, (), needs_grad=False)
        self.batch_loss_2 = ti.field(float, (), needs_grad=False)
        self.squared_error_sum_1 = ti.field(dtype=float, shape=(), needs_grad=True)
        self.squared_error_sum_2 = ti.field(dtype=float, shape=(), needs_grad=True)
        self.error_sum_1 = ti.field(dtype=float, shape=(), needs_grad=False)
        self.error_sum_2 = ti.field(dtype=float, shape=(), needs_grad=False)
        self.mean_error_1 = ti.field(dtype=float, shape=(), needs_grad=False)
        self.mean_error_2 = ti.field(dtype=float, shape=(), needs_grad=False)

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
        self.trajectory_ix = ti.field(dtype=int, shape=(), needs_grad=False)
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
        self.courant_number = SYSTEM_PARAMS.meta.target_courant_number
        self.retry = False

    def set_up_snapshot(self):
        self.predict_markers_snapshots = ti.Vector.field(
            2,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_training_trajectories, self.vitactip.num_markers),
            needs_grad=False,
        )
        self.virtual_markers_snapshots = ti.Vector.field(
            2,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_training_trajectories, self.vitactip.num_markers),
            needs_grad=False,
        )
        self.ground_truth_labels = ti.field(
            dtype=int, shape=(SYSTEM_PARAMS.contact.num_training_trajectories,), needs_grad=False
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
        self.tumour_present_ground_truth_label = ti.field(dtype=int, shape=(), needs_grad=False)
        self.tumour_present_ground_truth_label[None] = 0
        self.sim_keypoints_np = -np.ones((5,))
        self.sim_keypoints = ti.field(
            dtype=int, shape=(self.sim_keypoints_np.shape[0],), needs_grad=False
        )
        self.sim_keypoints.from_numpy(self.sim_keypoints_np)
        self.exp_keypoints_np = np.array([
            [-1, -1, 0, 47, 93],
            [-1, -1, 0, 23, 230],
            [-1, -1, 0, 30, 95],
            [-1, -1, 0, 39, 133]
        ], dtype=int)
        self.exp_keypoints = ti.Vector.field(
            5, dtype=int, shape=(self.exp_keypoints_np.shape[0],), needs_grad=False
        )
        self.exp_keypoints.from_numpy(self.exp_keypoints_np)

        self.exp_vein_ixs = np.array([
            103,
            179
        ], dtype=int)
        self.exp_vein_2d_coords = np.array([
            [935, 881],
            [1197, 899]
        ], dtype=float)
        self.exp_vein_3d_coords_E_np = self.fisheye_model.project_pix_to_points_3d_plane(self.exp_vein_2d_coords)
        self.exp_vein_3d_coords_E = ti.Vector.field(
            3, dtype=float, shape=(self.exp_vein_3d_coords_E_np.shape[0],), needs_grad=False
        )
        self.exp_vein_3d_coords_E.from_numpy(self.exp_vein_3d_coords_E_np)
        self.vein_speed_E_np = (
            (self.exp_vein_3d_coords_E_np[1] - self.exp_vein_3d_coords_E_np[0])
            /
            (self.exp_vein_ixs[1] - self.exp_vein_ixs[0])
        )
        self.vein_speed_E_np.reshape((1, 3))
        self.vein_speed_E = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.vein_speed_E.from_numpy(self.vein_speed_E_np.reshape(3))
        self.exp_vein_3d_coords_E_all_np = -np.ones(shape=(self.exp_keypoints_np[1][4]+1, 3), dtype=float)
        for i in range(self.exp_keypoints_np[1][3], self.exp_keypoints_np[1][4]+1):
            self.exp_vein_3d_coords_E_all_np[i] = (
                self.exp_vein_3d_coords_E_np[0]
                + (i - self.exp_vein_ixs[0]) * self.vein_speed_E_np
            )
        self.exp_vein_3d_coords_E_all = ti.Vector.field(
            3, dtype=float, shape=(self.exp_vein_3d_coords_E_all_np.shape[0],), needs_grad=False
        )
        self.exp_vein_3d_coords_E_all.from_numpy(self.exp_vein_3d_coords_E_all_np)
        validation_point_2d = np.array([
            [1028, 947]
        ])
        self.validation_point_3d_E_np = self.fisheye_model.project_pix_to_points_3d_plane(validation_point_2d)
        with open(SYSTEM_PARAMS.files.validation_point_E, "wb") as f:
            pickle.dump(self.validation_point_3d_E_np, f)
        self.validation_point_3d_E = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.validation_point_3d_E.from_numpy(self.validation_point_3d_E_np.reshape(3))
        self.validation_point_3d_A = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.num_trajectories = SYSTEM_PARAMS.meta.num_trajectories
        self.max_num_trajectory_points = SYSTEM_PARAMS.meta.max_num_trajectory_points
        self.trajectories = ti.Vector.field(7, dtype=float, shape=(
            self.num_trajectories,
            self.max_num_trajectory_points
        ), needs_grad=False)
        self.trajectory_lengths = ti.field(dtype=int, shape=(self.num_trajectories,), needs_grad=False)

    def set_up_keypoints(self):
        self.keypoint_indices = np.concatenate(
            (
                self.vitactip.get_keypoint_indices(0),
                self.phantom.get_keypoint_index(),
            ),
            dtype=int,
        )
    
    def set_up_trajectories_and_phantom_states(self):
        x, y, z = self.vitactip_tip_pose[:3]
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        twist_1_offset = R.from_euler(seq="xyz", angles=[30, 0, 0], degrees=True)
        twist_2_offset = R.from_euler(seq="xyz", angles=[0, 0, -45], degrees=True)
        # slide_offset = R.from_euler(seq="xyz", angles=[0, 0, 180], degrees=True)
        twist_1_base_offset = R.from_euler(seq="xyz", angles=[0, 0, 90], degrees=True)
        slide_r = og_r
        twist_1_r = og_r * twist_1_base_offset
        twist_1 = twist_1_r * twist_1_offset
        twist_2 = og_r * twist_2_offset
        press_depth_surface = SYSTEM_PARAMS.geometry.gap
        press_depth_0 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_0
        press_depth_1 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_1
        press_depth_2 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_2
        press_depth_3 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_3
        slide_dist = SYSTEM_PARAMS.trajectory.slide_distance
        self.trajectory_names = [
            'press (no vein)',
            'slide (vein)',
            'twist-y (no vein)',
            'twist-z (no vein)',
        ]
        trajectories_python_array = [
            [
                [x, y, z, *og_r.as_quat()],
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_surface, *og_r.as_quat()],

                [x, y, z - press_depth_0, *og_r.as_quat()],

                [x, y, z - press_depth_surface, *og_r.as_quat()],
            ],
            [
                [x, y, z, *slide_r.as_quat()],
                [x, y, z, *slide_r.as_quat()],
                [x, y, z - press_depth_surface, *slide_r.as_quat()],

                [x, y, z - press_depth_1, *slide_r.as_quat()],
                [x + slide_dist, y, z - press_depth_1, *slide_r.as_quat()],
            ],
            [
                [x, y, z, *twist_1_r.as_quat()],
                [x, y, z, *twist_1_r.as_quat()],
                [x, y, z - press_depth_surface, *twist_1_r.as_quat()],

                [x, y, z - press_depth_2, *twist_1_r.as_quat()],
                [x, y, z - press_depth_2, *twist_1.as_quat()],
            ],
            [
                [x, y, z, *og_r.as_quat()],
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_surface, *og_r.as_quat()],

                [x, y, z - press_depth_3, *og_r.as_quat()],
                [x, y, z - press_depth_3, *twist_2.as_quat()],
            ]
        ]
        self.set_trajectories(trajectories_python_array)
        cz_offset = SYSTEM_PARAMS.geometry.phantom_z_length / 2 - SYSTEM_PARAMS.geometry.vein.depth_beneath_surface
        self.state_dicts = [
            [],
            [
                {
                    'cx': 0,
                    'cy': 0,
                    'cz': cz_offset,
                    'theta': SYSTEM_PARAMS.geometry.vein.theta,
                    'h': SYSTEM_PARAMS.geometry.vein.h,
                    'r': SYSTEM_PARAMS.geometry.vein.r
                }
            ],
            [],
            [],
        ]
        assert(len(trajectories_python_array) == len(self.state_dicts))

    def randomise_train_step(self):
        t1 = self.get_random_grid_search_trajectory()
        t2 = self.get_fully_random_trajectory()
        trajectories_python_array = [t1, t2]
        self.set_trajectories(trajectories_python_array)

        self.state_dicts[0] = self.generate_random_state_dicts()
        self.state_dicts[1] = self.generate_random_state_dicts()
    
    def get_random_slide_params(self):
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        xr = np.random.uniform(-10, 10)
        yr = np.random.uniform(-10, 10)
        zr = np.random.uniform(0, 60)
        rand_r = R.from_euler(seq="xyz", angles=[0, 0, zr], degrees=True)
        og_r = og_r * rand_r
        slide_r = og_r
        srq = slide_r.as_quat()

        press_depth_surface = SYSTEM_PARAMS.geometry.gap
        press_depth_1 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_1
        if True:
            k_0 = SYSTEM_PARAMS.trajectory.press_depth_offset_0
            k_1 = SYSTEM_PARAMS.trajectory.press_depth_offset_1
            press_depth_rand = np.random.uniform(-k_0, k_1)
            press_depth_1 = press_depth_1 + press_depth_rand
        
        return (
            srq,
            press_depth_surface,
            press_depth_1
        )

    def get_random_grid_search_trajectory(self):
        (
            srq,
            press_depth_surface,
            press_depth_1
        ) = self.get_random_slide_params()
        _, _, z = self.vitactip_tip_pose[:3]
        x = self.sensor_x_range_world[0]
        y = self.sensor_y_range_world[0]
        dx = self.sensor_x_range_world[1] - self.sensor_x_range_world[0]
        dy = self.sensor_y_range_world[1] - self.sensor_y_range_world[0]
        r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        d_single = random.uniform(0.5 * r, 2 * r)
        trajectory = [
            [x, y, z, *srq],
            [x, y, z - press_depth_surface, *srq],
            [x, y, z - press_depth_1, *srq],
        ]
        xy_dirs = [
            [0, 1],
            [1, 0],
            [0, -1],
            [1, 0]
        ]
        xy_i = 0
        while True:
            a, b, c = trajectory[-1][:3]
            if (
                a > self.sensor_x_range_world[1]
                or len(trajectory) == self.trajectories.shape[1]
            ):
                break
            x_dir, y_dir = xy_dirs[xy_i]
            xy_i += 1
            xy_i %= 4
            a2 = a + x_dir * d_single
            b2 = b + y_dir * dy
            trajectory.append(
                [a2, b2, c, *srq]
            )
        return trajectory

    def get_fully_random_trajectory(self):
        (
            srq,
            press_depth_surface,
            press_depth_1
        ) = self.get_random_slide_params()
        _, _, z = self.vitactip_tip_pose[:3]
        x = self.sensor_x_range_world[0]
        y = self.sensor_y_range_world[0]
        
        # Initial press-down motion
        trajectory = [
            [x, y, z, *srq],
            [x, y, z - press_depth_surface, *srq],
            [x, y, z - press_depth_1, *srq],
        ]
        
        # Calculate maximum possible magnitude based on sensor bounds
        x_min, x_max = self.sensor_x_range_world
        y_min, y_max = self.sensor_y_range_world
        max_dx = x_max - x_min
        max_dy = y_max - y_min
        max_magnitude = min(max_dx, max_dy) / 2  # Conservative estimate
        
        # Generate remaining trajectory points using polar coordinates
        current_x, current_y = x, y
        while len(trajectory) < self.trajectories.shape[1]:
            magnitude = random.uniform(0, max_magnitude)
            
            # Keep trying angles until we find one that keeps point in bounds
            while True:
                angle = random.uniform(0, 2 * math.pi)
                new_x = current_x + magnitude * math.cos(angle)
                new_y = current_y + magnitude * math.sin(angle)
                
                # Check if new point is within bounds
                if (x_min <= new_x <= x_max and 
                    y_min <= new_y <= y_max):
                    trajectory.append(
                        [new_x, new_y, z - press_depth_1, *srq]
                    )
                    current_x, current_y = new_x, new_y
                    break
                    
        return trajectory

    def set_up_initial_positions_state_and_trajectory(self):
        state_dicts = self.state_dicts[self.trajectory_ix[None]]

        self.phantom.set_state_from_outside(
            pos=self.phantom_centroid_pose[:3],
            ori=self.phantom_centroid_pose[3:],
            vel=[0.0, 0.0, 0.0],
            state_dicts=state_dicts,
        )
        sensor_dome_tip_initial_pose = self.trajectories[self.trajectory_ix[None], 0].to_numpy()
        self.vitactip.set_up_pose(sensor_dome_tip_initial_pose)
        self.tactile_sensor_initial_position[0] = ti.Vector(
            sensor_dome_tip_initial_pose[:3]
        )
        self.phantom_initial_position[0] = ti.Vector(self.phantom_centroid_pose[:3])
    
    def set_trajectories(self, trajectories_python_arr):
        # Create a zero-initialized array for padded trajectories
        trajectories_np = np.zeros((self.num_trajectories, self.max_num_trajectory_points, 7), dtype=np.float32)
        
        # Create an array to store the actual length of each trajectory
        trajectory_lengths = np.zeros(self.num_trajectories, dtype=int)
        
        # Fill in the actual trajectory data
        for i, trajectory in enumerate(trajectories_python_arr):
            traj_len = min(len(trajectory), self.max_num_trajectory_points)
            trajectory_lengths[i] = traj_len
            trajectories_np[i, :traj_len] = np.array(trajectory[:traj_len])
        
        self.trajectories.from_numpy(trajectories_np)
        self.trajectory_lengths.from_numpy(trajectory_lengths)
    
    def generate_random_state_dicts(self):
        state_dicts = []
        num_veins = SYSTEM_PARAMS.meta.max_num_veins
        placed_cy_values = []
        min_separation = SYSTEM_PARAMS.geometry.min_vein_separation
        for i in range(num_veins):
            theta_rand = 0
            cz_offset = SYSTEM_PARAMS.geometry.phantom_z_length / 2 - SYSTEM_PARAMS.geometry.vein.depth_beneath_surface
            cx = self.sensor_x_range_phantom[0]
            
            # px = SYSTEM_PARAMS.geometry.phantom_x_length
            # py = SYSTEM_PARAMS.geometry.phantom_y_length
            # pd = (px**2 + py**2) ** (1/2)
            # h = np.random.uniform(1/4 * pd, pd)

            while True:
                cy = np.random.uniform(*self.sensor_y_range_phantom)
                valid_position = True
                for prev_cy in placed_cy_values:
                    if abs(cy - prev_cy) < min_separation:
                        valid_position = False
                        break
                if valid_position:
                    placed_cy_values.append(cy)
                    break
                
            h = SYSTEM_PARAMS.geometry.vein.h
            state_dict = {
                'cx': cx,
                'cy': cy,
                'cz': cz_offset,
                'theta': SYSTEM_PARAMS.geometry.vein.theta + theta_rand,
                'h': h,
                'r': SYSTEM_PARAMS.geometry.vein.r
            }
            state_dicts.append(state_dict)
            
        print('placed_cy_values')
        print(placed_cy_values)
        return state_dicts

    @ti.kernel
    def reset_exp_sim_traj(self):
        self.marker_position_exp.fill(-1)
        self.sim_keypoints.fill(-1)
        self.sim_to_exp_markers.fill(-1)
        self.exp_to_sim_markers.fill(-1)
        self.exp_marker_points.fill(-1)
        self.sim_markers_deformed_filtered.fill(-1)
        self.sim_markers_deformed_filtered_z.fill(-1)
        self.sim_markers_deformed_z.fill(-1)
        self.cur_exp_frame.fill(-1)
        self.vein_ix_base.fill(-1)
        self.vein_ix_offset.fill(-1)
        self.vein_exp_vis.fill(0)
        self.vein_exp_vis_all.fill(0)
        self.interpolation_valid.fill(1)

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
    def clear_grad_helper(self):
        self.loss_1.grad.fill(0.0)
        self.loss_2.grad.fill(0.0)
        self.squared_error_sum_1.grad.fill(0.0)
        self.squared_error_sum_2.grad.fill(0.0)
        self.normal_stiffness.grad.fill(0.0)
        self.normal_damping.grad.fill(0.0)
        self.tangential_stiffness.grad.fill(0.0)
        self.coulomb_friction_coeff.grad.fill(0.0)
        self.vitactip_youngs_modulus_log.grad.fill(0)
        self.phantom_youngs_modulus_0_log.grad.fill(0)
        self.phantom_youngs_modulus_1_log.grad.fill(0)
        self.vitactip_poissons_ratio_log.grad.fill(0)
        self.phantom_poissons_ratio_0_log.grad.fill(0)
        self.phantom_poissons_ratio_1_log.grad.fill(0)
        self.coulomb_friction_coeff_log.grad.fill(0)
        self.normal_stiffness_log.grad.fill(0)
        self.tangential_stiffness_log.grad.fill(0)
        self.normal_damping_log.grad.fill(0)

    def clear_grad(self):
        self.clear_grad_helper()
        self.vitactip.clear_grad()
        self.phantom.clear_grad()
    
    @ti.kernel
    def reset_loss(self):
        self.batch_loss[None] += self.loss[None]
        self.batch_loss_1[None] += self.loss_1[None]
        self.batch_loss_2[None] += self.loss_2[None]
        self.loss.fill(0.0)
        self.loss_1.fill(0.0)
        self.loss_2.fill(0.0)
        self.squared_error_sum_1.fill(0.0)
        self.squared_error_sum_2.fill(0.0)
        self.error_sum_1.fill(0.0)
        self.error_sum_2.fill(0.0)
        self.mean_error_1.fill(0.0)
        self.mean_error_2.fill(0.0)
    
    @ti.kernel
    def reset_batch_loss(self):
        self.trajectory_loss[None] += self.batch_loss[None]
        self.batch_loss[None] = 0
        self.batch_loss_1[None] = 0
        self.batch_loss_2[None] = 0

    @ti.kernel
    def reset_state(self):
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.contact_idx.fill(-1)
        self.coulomb_friction_coeff.fill(0)
        self.normal_stiffness.fill(0)
        self.tangential_stiffness.fill(0)
        self.normal_damping.fill(0)

    @ti.func
    def dist(self, a, b) -> ti.f32:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dx /= SYSTEM_PARAMS.fisheye_model.target_image_width
        dy /= SYSTEM_PARAMS.fisheye_model.target_image_height
        squared_error = dx * dx + dy * dy
        return ti.sqrt(squared_error)

    @ti.kernel
    def compute_marker_loss_1(self):
        for i in range(self.vitactip.num_markers):
            if self.interpolation_valid[None] == 1:
                exp_ix = self.sim_to_exp_markers[i]
                if exp_ix != -1:
                    sim_marker = self.vitactip.deformed_markers[i]
                    exp_marker = self.marker_position_exp[exp_ix]
                    self.squared_error_sum_1[None] += self.dist(sim_marker, exp_marker) ** 2
                    self.error_sum_1[None] += self.dist(sim_marker, exp_marker)

    @ti.kernel
    def compute_marker_loss_2(self):
        if self.interpolation_valid[None] == 1:
            rmse = ti.sqrt(self.squared_error_sum_1[None] / self.marker_position_exp.shape[0])
            self.loss_1[None] += SYSTEM_PARAMS.optimisation.loss_1_weight * rmse
            self.mean_error_1[None] += self.error_sum_1[None] / self.marker_position_exp.shape[0]
    
    @ti.kernel
    def compute_marker_loss_3(self):
        if (
            self.interpolation_valid[None] == 1
            and self.trajectory_ix[None] == 1
            and self.cur_exp_frame[None] >= self.traj_1_critical_frames_exp[0]
            and self.cur_exp_frame[None] < self.traj_1_critical_frames_exp[1]
        ):
            for i in range(self.traj_1_exp_marker_pairs.shape[0]):
                a_exp_ix = self.traj_1_exp_marker_pairs[i][0]
                b_exp_ix = self.traj_1_exp_marker_pairs[i][1]
                a_sim_ix = self.exp_to_sim_markers[a_exp_ix]
                b_sim_ix = self.exp_to_sim_markers[b_exp_ix]

                a_exp = self.marker_position_exp[a_exp_ix]
                b_exp = self.marker_position_exp[b_exp_ix]
                a_sim = self.vitactip.deformed_markers[a_sim_ix]
                b_sim = self.vitactip.deformed_markers[b_sim_ix]

                dist_exp = self.dist(a_exp, b_exp)
                dist_sim = self.dist(a_sim, b_sim)

                self.squared_error_sum_2[None] += (dist_exp - dist_sim) ** 2
                self.error_sum_2[None] += abs(dist_exp - dist_sim)

    @ti.kernel
    def compute_marker_loss_4(self):
        if (
            self.interpolation_valid[None] == 1
            and self.trajectory_ix[None] == 1
            and self.cur_exp_frame[None] >= self.traj_1_critical_frames_exp[0]
            and self.cur_exp_frame[None] < self.traj_1_critical_frames_exp[1]
        ):
            rmse = ti.sqrt(self.squared_error_sum_2[None] / self.traj_1_exp_marker_pairs.shape[0])
            self.loss_2[None] += SYSTEM_PARAMS.optimisation.loss_2_weight * rmse
            self.mean_error_2[None] += self.error_sum_2[None] / self.traj_1_exp_marker_pairs.shape[0]
    
    @ti.kernel
    def compute_marker_loss_5(self):
        if (
            self.interpolation_valid[None] == 1
        ):
            self.loss[None] += -self.loss_1[None]
            self.loss[None] += -self.loss_2[None]

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
                    + SYSTEM_PARAMS.phantom.mass_eps
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

    def copy_frame(self):
        self.vitactip.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames - 1, 0)
        self.phantom.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames - 1, 0)

    def memory_to_cache(self, t):
        self.vitactip.memory_to_cache(t)
        self.phantom.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.vitactip.memory_from_cache(t)
        self.phantom.memory_from_cache(t)

    def pid_controller_1(self):
        self.vitactip.compute_current_orientation()
        current_ori = R.from_quat(self.vitactip.R_BA_quat.to_numpy())
        target = self.trajectories[self.trajectory_ix[None], self.current_target_idx[None]].to_numpy()
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
        self.vitactip.R_A_quat.from_numpy(ori_control_quat.reshape(4))
        self.ori_error_magnitude_degrees[None] = np.rad2deg(current_angle_radians)

    @ti.kernel
    def pid_controller_2(self, ts: ti.i32):
        current_pos = self.vitactip.vertices_undeformed_A[0, self.keypoint_indices[0]]
        target = self.trajectories[self.trajectory_ix[None], self.current_target_idx[None]]
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
                if self.current_target_idx[None] < self.trajectory_lengths[self.trajectory_ix[None]] - 1:
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
            R.from_quat(self.vitactip.R_A_quat.to_numpy()).as_matrix().reshape(3,3)
        )

    def clear_temp_images(self):
        folders = [
            SYSTEM_PARAMS.files.training_data_vein_full_folder,
            SYSTEM_PARAMS.files.training_data_contact_folder,
            SYSTEM_PARAMS.files.training_data_markers_folder,
            SYSTEM_PARAMS.files.training_data_segmentation_mask_folder,
        ]
        self.clear_training_data_folders_helper(folders)

    def clear_npz(self):
        folders = [
            SYSTEM_PARAMS.files.dataset_root
        ]
        self.clear_training_data_folders_helper(folders)

    def clear_training_data_folders_helper(self, folders):
        for folder in folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def record_training_data_point(self, training_iteration, ts):
        w = int(SYSTEM_PARAMS.fisheye_model.target_image_width)
        h = int(SYSTEM_PARAMS.fisheye_model.target_image_height)

        markers_file = SYSTEM_PARAMS.files.training_data_markers.format(training_iteration, ts)
        vein_file = SYSTEM_PARAMS.files.training_data_segmentation_mask.format(training_iteration, ts)
        vein_full_file = SYSTEM_PARAMS.files.training_data_vein_full.format(training_iteration, ts)
        contact_file = SYSTEM_PARAMS.files.training_data_contact.format(training_iteration, ts)

        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        r = SYSTEM_PARAMS.fisheye_model.circle_radius

        self.move_og_resolution()
        markers = self.sim_markers_deformed.to_numpy()
        self.move_ti_resolution()
        markers = SyntheticImageGenerator.filter_points(w, h, cx, cy, r, markers)
        self.marker_data.append(markers)
        markers_img = np.zeros((h, w), dtype=np.uint8)
        for point in markers:
            x, y = int(point[0]), int(point[1])
            cv2.circle(markers_img, (x, y), radius=1, color=255, thickness=-1)
        markers_mask = Contact.compute_mask(h, w, markers)

        vein = self.vein_2d_projection.to_numpy()
        vein_counts = self.phantom.vein_counts.to_numpy()
        vein_python_arr = []
        for i in range(vein.shape[0]):
            num_points = vein_counts[i]
            vein_python_arr.append(
                vein[i, :num_points]
            )
        vein = vein_python_arr
        vein_np, vein_mask = Contact.create_padded_array_with_mask(vein)
        self.vein_data.append(vein_np)
        self.vein_mask_data.append(vein_mask)
        vein_endpoints_python_arr = []
        for i in range(len(vein)):
            single_vein = vein[i]
            single_endpoints = Contact.fit_straight_line_and_return_endpoints(single_vein)
            vein_endpoints_python_arr.append(single_endpoints)
        vein_endpoints_np, vein_endpoints_mask = Contact.create_padded_array_with_mask(vein_endpoints_python_arr)
        self.vein_endpoints_data.append(vein_endpoints_np)
        self.vein_endpoints_mask_data.append(vein_endpoints_mask)

        if training_iteration == 0:
            cv2.imwrite(contact_file, markers_mask)
            cv2.imwrite(markers_file, markers_img)
    
    @staticmethod
    def fit_straight_line_and_return_endpoints(points):
        """
        Fit a straight line to points and return the endpoints.
        
        Args:
            points: numpy array of shape (num_points, 2)
            
        Returns:
            numpy array of shape (2, 2) containing the two endpoints
        """
        if len(points) < 2:
            foo = -np.ones(shape=(0, 2), dtype=float)
            return foo
        
        # Convert to numpy array if not already
        points = np.array(points, dtype=np.float64)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Use PCA to find the best fitting line direction
        # The first principal component gives us the line direction
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The eigenvector corresponding to the largest eigenvalue
        # gives us the direction of the line
        line_direction = eigenvectors[:, -1]
        
        # Project all points onto the line
        # For each point, compute the scalar projection onto the line direction
        projections_scalar = np.dot(centered_points, line_direction)
        
        # Find the indices of points with minimum and maximum projections
        min_idx = np.argmin(projections_scalar)
        max_idx = np.argmax(projections_scalar)
        
        # Compute the actual projected points on the line
        min_projection = centroid + projections_scalar[min_idx] * line_direction
        max_projection = centroid + projections_scalar[max_idx] * line_direction
        
        # Return the two endpoints
        endpoints = np.array([min_projection, max_projection])
        
        return endpoints

    @staticmethod
    def compute_mask(h, w, points):
        res = np.zeros((h, w), dtype=np.uint8)
        if len(points) > 0:
            contour_vein = SyntheticImageGenerator.alpha_shape(points).astype(np.int32)
            if len(contour_vein) > 0:
                contour_vein_cv = contour_vein.reshape((-1, 1, 2))
                cv2.fillPoly(res, [contour_vein_cv], color=255)
        
        return res
    
    @staticmethod
    def filter_using_mask(mask, points):
        res = []
        for point in points:
            x, y = int(point[0]), int(point[1])
            if mask[y, x] > 127:
                res.append(point)
        res = np.array(res)
        return res

    def write_training_data_to_file(self, training_iteration):
        directory = SYSTEM_PARAMS.files.dataset_root
        file = SYSTEM_PARAMS.files.dataset_data_point.format(
            training_iteration
        )
        path = f'{directory}/{file}'

        markers_array, markers_mask = Contact.create_padded_array_with_mask(self.marker_data)
        veins_array = np.array(self.vein_data)
        veins_mask = np.array(self.vein_mask_data)
        vein_endpoints_array = np.array(self.vein_endpoints_data)
        vein_endpoints_mask = np.array(self.vein_endpoints_mask_data)

        np.savez(
            path,
            markers=markers_array,
            markers_mask=markers_mask,
            labels=veins_array,
            labels_mask=veins_mask,
            vein_endpoints=vein_endpoints_array,
            vein_endpoints_mask=vein_endpoints_mask
        )
        
        self.marker_data = []
        self.vein_data = []
        self.vein_mask_data = []
        self.vein_endpoints_data = []
        self.vein_endpoints_mask_data = []

    @staticmethod
    def create_padded_array_with_mask(data_list):
        if not data_list:
            return np.array([]), np.array([])
        n = len(data_list)
        num_points_max = max(arr.shape[0] for arr in data_list)
        padded_array = np.zeros((n, num_points_max, 2), dtype=data_list[0].dtype)
        mask = np.zeros((n, num_points_max), dtype=bool)
        for i, arr in enumerate(data_list):
            num_points = arr.shape[0]
            if num_points > 0:
                padded_array[i, :num_points, :] = arr
                mask[i, :num_points] = True
        return padded_array, mask

    def take_2d_markers_snapshot(self, k):
        self.take_snapshot_1(k)
        self.take_snapshot_2(k)

    @ti.kernel
    def take_snapshot_1(self, k: ti.i32):
        for i in range(self.vitactip.num_markers):
            self.predict_markers_snapshots[k, i] = self.vitactip.deformed_markers[i]
            self.virtual_markers_snapshots[k, i] = self.vitactip.undeformed_markers[
                i
            ]

    @ti.kernel
    def take_snapshot_2(self, k: ti.i32):
        self.ground_truth_labels[k] = self.tumour_present_ground_truth_label[None]

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
        self.vein_exp_vis_all = ti.Vector.field(
            3, dtype=float, shape=(self.exp_vein_3d_coords_E_all.shape[0],), needs_grad=False
        )
        self.num_keypoints = 3
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
        self.vein_2d_projection = ti.Vector.field(
            2,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.meta.max_num_veins,
                self.phantom.actual_total_num_particles
            ),
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
        self.sim_markers_deformed_filtered_z = ti.field(
            dtype=float, shape=(self.marker_position_exp.shape[0],), needs_grad=False
        )
        self.sim_markers_deformed_z = ti.field(
            dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
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
        self.tactile_image_resolution = ti.Vector.field(2, dtype=float, shape=(), needs_grad=False)
        self.tactile_image_resolution[None] = ti.Vector([
            SYSTEM_PARAMS.fisheye_model.target_image_width,
            SYSTEM_PARAMS.fisheye_model.target_image_height
        ])
        self.fp_bp = ti.field(dtype=int, shape=(), needs_grad=False)
        self.vein_ix_base = ti.field(dtype=int, shape=(), needs_grad=False)
        self.vein_ix_offset = ti.field(dtype=float, shape=(), needs_grad=False)
        self.vein_exp_vis = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.interpolation_valid = ti.field(
            dtype=int, shape=(), needs_grad=False
        )
        self.interpolation_valid[None] = 1

    @ti.kernel
    def visualisation_reset_scene(self):
        self.healthy_tissue_points.fill(0)
        self.tumour_points.fill(0)
        self.vein_2d_projection.fill(-1)

    @ti.kernel
    def visualisation_draw_3d_scene(self, f: ti.i32):
        for p in range(self.phantom.actual_total_num_particles):
            if self.phantom.titles[p] == 0:
                self.healthy_tissue_points[p] = self.phantom.particles_A[f, p]
            elif self.phantom.titles[p] == 1:
                self.tumour_points[p] = self.phantom.particles_A[f, p]
        for p in range(self.vitactip.num_vertices):
            self.sensor_points[p] = self.vitactip.vertices_deformed_A[f, p]

    @ti.kernel
    def visualisation_project_vein_2d(self):
        for i in range(self.phantom.vein_counts.shape[0]):
            num_points = self.phantom.vein_counts[i]
            for j in range(num_points):
                vein_ix = self.phantom.vein_indices[i, j]
                point = self.phantom.particles_A[
                    SYSTEM_PARAMS.contact.num_sub_frames - 1,
                    vein_ix
                ]
                projection_2d = self.vitactip.project_A_point_2d(point)
                projection_2d[1] = self.tactile_image_resolution[None][1] - projection_2d[1]
                self.vein_2d_projection[i, j] = projection_2d

    @ti.kernel
    def visualisation_prepare_tactile_readout_data_fp(self):
        for i in range(self.vitactip.num_markers):
            undeformed = self.vitactip.undeformed_markers[i]
            deformed = self.vitactip.deformed_markers[i]
            deformed_z = self.vitactip.deformed_markers_z[i]
            undeformed[1] = self.tactile_image_resolution[None][1] - undeformed[1]
            deformed[1] = self.tactile_image_resolution[None][1] - deformed[1]
            undeformed = undeformed / self.tactile_image_resolution[None]
            deformed = deformed / self.tactile_image_resolution[None]
            offset = deformed - undeformed
            self.sim_markers_undeformed[i] = undeformed
            self.sim_markers_deformed[i] = deformed
            self.arrow_line_vertices[i * 2] = undeformed
            self.arrow_line_vertices[i * 2 + 1] = undeformed + offset
            self.sim_markers_deformed_z[i] = deformed_z

            exp_ix = self.sim_to_exp_markers[i]
            if exp_ix != -1:
                self.sim_markers_deformed_filtered[exp_ix] = deformed
                self.sim_markers_deformed_filtered_z[exp_ix] = deformed_z
    
    @ti.kernel
    def move_og_resolution(self):
        for i in range(self.sim_markers_deformed_filtered.shape[0]):
            if abs(self.sim_markers_deformed_filtered[i][0] - (-1.0)) > 1e-6:
                self.sim_markers_deformed_filtered[i] *= self.tactile_image_resolution[None]
        for i in range(self.sim_markers_deformed.shape[0]):
            self.sim_markers_deformed[i] *= self.tactile_image_resolution[None]

    @ti.kernel
    def move_ti_resolution(self):
        for i in range(self.sim_markers_deformed_filtered.shape[0]):
            if abs(self.sim_markers_deformed_filtered[i][0] - (-1.0)) > 1e-6:
                self.sim_markers_deformed_filtered[i] /= self.tactile_image_resolution[None]
        for i in range(self.sim_markers_deformed.shape[0]):
            self.sim_markers_deformed[i] /= self.tactile_image_resolution[None]

    @ti.kernel
    def visualisation_prepare_tactile_readout_data_bp(self):
        for i in range(self.marker_position_exp.shape[0]):
            point = self.marker_position_exp[i]
            if point[0] > 0 and point[1] > 0:
                point[1] = self.tactile_image_resolution[None][1] - point[1]
                self.exp_marker_points[i] = point / self.tactile_image_resolution[None]
            else:
                self.exp_marker_points[i] =  ti.Vector([-1.0, -1.0], dt=float)

    @ti.kernel
    def visualisation_prepare_clock_arm_points(self):
        for i in range(2):
            point = self.vitactip.projection_2d_clock_arms[i]
            point[1] = self.tactile_image_resolution[None][1] - point[1]
            self.clock_arm_points[i] = point / self.tactile_image_resolution[None]

    @staticmethod
    def line_equation(points):
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Calculate the coefficients of the line equation ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        return a, b, c

    @staticmethod
    def vector_point_to_line(a, b, c, p):
        p = np.array(p)
        if len(p.shape) == 1:  # Single point
            numerator = a * p[0] + b * p[1] + c
            denominator = np.sqrt(a * a + b * b)
            
            if abs(denominator) < 1e-10:
                return np.zeros(2)
                
            normal = np.array([a, b]) / denominator
            return -normal * numerator / denominator
        else:  # Multiple points
            numerator = a * p[..., 0] + b * p[..., 1] + c
            denominator = np.sqrt(a * a + b * b)
            
            if abs(denominator) < 1e-10:
                return np.zeros(p.shape)
                
            normal = np.array([a, b]) / denominator
            return -normal[None, :] * (numerator / denominator)[..., None]

    @staticmethod
    def line_point_to_line_pass_through_point(a, b, c, p):
        """
        Compute coefficients (a', b', c') of a line that:
        1. Passes through point p
        2. Is perpendicular to line ax + by + c = 0
        
        Args:
            a, b, c: Coefficients of original line ax + by + c = 0
            p: Point [x, y] that the perpendicular line should pass through
            
        Returns:
            a', b', c': Coefficients of perpendicular line a'x + b'y + c' = 0
        """
        # For line ax + by + c = 0, vector (-b, a) is perpendicular to the line
        # This will be our direction vector for the new line
        a_new = -b  # Use negative b as new a coefficient
        b_new = a   # Use a as new b coefficient
        
        # For the new line to pass through point p:
        # a_new * p[0] + b_new * p[1] + c_new = 0
        # Therefore:
        c_new = -(a_new * p[0] + b_new * p[1])
        
        return a_new, b_new, c_new

    def visualisation_draw_tactile_readout(self):
        self.visualisation_project_vein_2d()
        self.vitactip.extract_clock_arm_2d_projections(SYSTEM_PARAMS.contact.num_sub_frames - 1)
        self.visualisation_prepare_clock_arm_points()
        self.tactile_canvas.set_image(self.bg_image)
        self.visualisation_prepare_tactile_readout_data_fp()
        if (
            self.fp_bp[None] == 1
            and SYSTEM_PARAMS.visualisation.visualise_exp_markers_during_bp == 1
        ):
            self.visualisation_prepare_tactile_readout_data_bp()
            self.tactile_canvas.circles(
                self.exp_marker_points, radius=0.01, color=(0, 1, 0)
            )
        self.tactile_canvas.circles(
            self.sim_markers_deformed_filtered, radius=0.01, color=(1, 0, 0)
        )
        self.tactile_canvas.circles(
            self.clock_arm_points,
            radius=0.02,
            per_vertex_color=self.clock_arm_points_per_vertex_color,
        )
        self.tactile_window.show()

    def visualisation_set_up_gui(self):
        self.window = ti.ui.Window("high-level camera", (
            int(SYSTEM_PARAMS.visualisation.window_3d_width),
            int(SYSTEM_PARAMS.visualisation.window_3d_height)
        ))
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0, 0, 0))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        x, y, z = self.phantom_centroid_pose[:3]
        self.camera.position(x, y, z+SYSTEM_PARAMS.visualisation.camera_offset)
        self.camera.up(0, 1, 0)
        self.camera.lookat(x, y, z)
        self.camera.fov(10)
        self.tactile_window = ti.ui.Window("tactile readout", (
            int(SYSTEM_PARAMS.visualisation.tactile_readout_width),
            int(SYSTEM_PARAMS.visualisation.tactile_readout_height)
        ))
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
        key_points_per_vertex_color_npy[1:3, :] = clock_arm_points_per_vertex_color_npy
        self.key_points_per_vertex_color.from_numpy(key_points_per_vertex_color_npy)

    def create_transition_array_vectorized(self, n):
        t = np.linspace(0, 1, n)[:, np.newaxis]
        start = np.array([0, 1, 1])
        end = np.array([1, 0, 0])
        return (1 - t) * start + t * end

    def visualisation_update_gui(self, ts):
        vitactip_coords = self.vitactip.vertices_deformed_A.to_numpy()[0]
        if np.isnan(vitactip_coords).any():
            nan_count = np.any(np.isnan(vitactip_coords), axis=1).sum()
            print(
                f"ViTacTip contains {nan_count} / {vitactip_coords.shape[0]} nan vertices at ts: {ts}"
            )
        phantom_coords = self.phantom.particles_A.to_numpy()[0]
        if np.isnan(phantom_coords).any():
            nan_count = np.any(np.isnan(phantom_coords), axis=1).sum()
            print(
                f"phantom contains {nan_count} / {phantom_coords.shape[0]} nan vertices at ts: {ts}"
            )
        vitactip_bottom = self.vitactip.get_keypoint_coordinates(
            0, self.keypoint_indices[0].reshape((1,))
        )
        trajectory_keypoints = self.trajectories.to_numpy()[self.trajectory_ix[None], :, :3]
        vitactip_clock_arms = self.vitactip.get_keypoint_coordinates(
            f=0, keypoint_indices=self.vitactip.clock_arms_node_idxs.to_numpy()
        )
        if self.trajectory_ix[None] == 1:
            self.compute_vein_exp_vis()
            self.compute_vein_exp_vis_all()
            self.compute_validation_point()
        vein_exp_vis = self.vein_exp_vis.to_numpy()
        vein_exp_vis_all = self.vein_exp_vis_all.to_numpy()
        validation_point = self.validation_point_3d_A.to_numpy()
        self.keypoint_coords = np.vstack(
            (vitactip_bottom, vitactip_clock_arms)
        )
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        self.visualisation_draw_3d_scene(SYSTEM_PARAMS.contact.num_sub_frames - 1)
        self.visualisation_draw_tactile_readout()
        self.scene.particles(
            self.healthy_tissue_points,
            color=(0.0, 0.0, 1.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal/2,
        )
        self.scene.particles(
            self.tumour_points,
            color=(1.0, 1.0, 0.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        self.scene.particles(
            self.sensor_points,
            color=(0.0, 1.0, 0.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        assert self.keypoint_coords.shape[0] == self.key_points.shape[0], (
            f"Set self.key_points to shape ({self.keypoint_coords.shape[0]},)"
        )
        assert self.keypoint_coords.shape[0] == self.key_points_per_vertex_color.shape[0], (
            f"Set self.key_points_per_vertex_color to shape ({self.keypoint_coords.shape[0]},)"
        )
        if self.keypoint_coords is not None:
            self.key_points.from_numpy(self.keypoint_coords)
            self.scene.particles(
                self.key_points,
                color=(1.0, 0.0, 0.0),
                per_vertex_color=self.key_points_per_vertex_color,
                radius=SYSTEM_PARAMS.visualisation.particle_size_keypoint*2,
            )
        self.canvas.scene(self.scene)
        self.window.show()

    def forward_pass_common_part(self, ts):
        self.reset_state()
        self.set_optimisation_params_from_log()
        self.vitactip.set_control_vel(0)
        self.vitactip.set_vel(0)
        self.vitactip.set_up_system_params_2()
        self.phantom.set_stiffness()
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 1):
            self.update(ss)

    def backward_pass_common_part(self):
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 2, -1, -1):
            self.update_grad(ss)
        self.phantom.set_stiffness.grad()
        self.vitactip.set_up_system_params_2.grad()
        self.set_optimisation_params_from_log.grad()

    def save_gradients_for_calibration(self):
        if not self.gradients_printed:
            print(f"loss: {self.loss.grad[None]}")
            print(f"squared_error_sum_1 (grad): {self.squared_error_sum_1.grad[None]}")
            print(f"squared_error_sum_1 (val): {self.squared_error_sum_1[None]}")
            print(f"squared_error_sum_2 (grad): {self.squared_error_sum_2.grad[None]}")
            print(f"squared_error_sum_2 (val): {self.squared_error_sum_2[None]}")
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
            self.print_gradients_indexed("phantom.mu", self.phantom.mu, 0)
            self.print_gradients_indexed("phantom.lam", self.phantom.lam, 0)
            self.print_gradients_indexed(
                "phantom.youngs_modulus", self.phantom.youngs_modulus, 0
            )
            self.print_gradients_indexed("phantom.mu", self.phantom.mu, 1)
            self.print_gradients_indexed("phantom.lam", self.phantom.lam, 1)
            self.print_gradients_indexed(
                "phantom.youngs_modulus", self.phantom.youngs_modulus, 1
            )
            print()

            if False:
                base_gradient_data = {
                    "loss": self.loss.grad[None],
                    "squared_error_sum_1": self.squared_error_sum_1.grad[None],
                    "squared_error_sum_2": self.squared_error_sum_2.grad[None],
                    "vitactip_mu": self.vitactip.mu.grad[None],
                    "vitactip_lam": self.vitactip.lam.grad[None],
                    "vitactip_youngs_modulus": self.vitactip.youngs_modulus.grad[None],
                    "normal_stiffness": self.normal_stiffness.grad[None],
                    "normal_damping": self.normal_damping.grad[None],
                    "tangential_stiffness": self.tangential_stiffness.grad[None],
                    "coulomb_friction_coeff": self.coulomb_friction_coeff.grad[None],
                    "phantom_mu_0": self.phantom.mu.grad[0],
                    "phantom_mu_1": self.phantom.mu.grad[1],
                    "phantom_lam_0": self.phantom.lam.grad[0],
                    "phantom_lam_1": self.phantom.lam.grad[1],
                    "phantom_youngs_modulus_0": self.phantom.youngs_modulus.grad[0],
                    "phantom_youngs_modulus_1": self.phantom.youngs_modulus.grad[1]
                }
                
                gradient_data = {k: float(v) for k, v in base_gradient_data.items()}

                if SYSTEM_PARAMS.optimisation.calibrate_learning_rates == 1:
                    with open(SYSTEM_PARAMS.files.optimisation_loop_calibration, 'w') as f:
                        json.dump(gradient_data, f, indent=4)

            self.gradients_printed = True

    def print_gradients_single(self, name, ti_var):
        grad_npy = ti_var.grad.to_numpy()
        val_npy = ti_var.to_numpy()
        print(f"{name} (grad) min: {grad_npy.min()}")
        print(f"{name} (grad) max: {grad_npy.max()}")
        print(f"{name} (val) min: {val_npy.min()}")
        print(f"{name} (val) max: {val_npy.max()}")
        print()
    
    def print_gradients_indexed(self, name, ti_var, ix):
        grad_npy = ti_var.grad.to_numpy()[ix]
        val_npy = ti_var.to_numpy()[ix]
        print(f"{name}_{ix} (grad) min: {grad_npy.min()}")
        print(f"{name}_{ix} (grad) max: {grad_npy.max()}")
        print(f"{name}_{ix} (val) min: {val_npy.min()}")
        print(f"{name}_{ix} (val) max: {val_npy.max()}")
        print()
    
    def print_params_short(self):
        print(f'vitactip.youngs_modulus: {self.vitactip.youngs_modulus[None]:0.16e} ({SYSTEM_PARAMS.vitactip.single_material.youngs_modulus:0.16e})')
        print(f'phantom.youngs_modulus_0: {self.phantom.youngs_modulus[0]:0.16e} ({SYSTEM_PARAMS.phantom.silicone.youngs_modulus:0.16e})')
        print(f'phantom.youngs_modulus_1: {self.phantom.youngs_modulus[1]:0.16e} ({SYSTEM_PARAMS.phantom.hard_plastic.youngs_modulus:0.16e})')
        print(f'vitactip.poissons_ratio: {self.vitactip.poissons_ratio[None]:0.16e} ({SYSTEM_PARAMS.vitactip.single_material.poissons_ratio:0.16e})')
        print(f'phantom.poissons_ratio_0: {self.phantom.poissons_ratio[0]:0.16e} ({SYSTEM_PARAMS.phantom.silicone.poissons_ratio:0.16e})')
        print(f'phantom.poissons_ratio_1: {self.phantom.poissons_ratio[1]:0.16e} ({SYSTEM_PARAMS.phantom.hard_plastic.poissons_ratio:0.16e})')
        print(f'coulomb_friction_coeff: {self.coulomb_friction_coeff[None]:0.16e} ({SYSTEM_PARAMS.contact.coulomb_friction_coeff:0.16e})')
        print(f'normal_stiffness: {self.normal_stiffness[None]:0.16e} ({SYSTEM_PARAMS.contact.normal_stiffness:0.16e})')
        print(f'tangential_stiffness: {self.tangential_stiffness[None]:0.16e} ({SYSTEM_PARAMS.contact.tangential_stiffness:0.16e})')
        print(f'normal_damping: {self.normal_damping[None]:0.16e} ({SYSTEM_PARAMS.contact.normal_damping:0.16e})')
        
        assert not math.isnan(float(self.vitactip.youngs_modulus[None])), "vitactip.youngs_modulus is NaN"
        assert not math.isnan(float(self.phantom.youngs_modulus[0])), "phantom.youngs_modulus[0] is NaN"
        assert not math.isnan(float(self.phantom.youngs_modulus[1])), "phantom.youngs_modulus[1] is NaN"
        assert not math.isnan(float(self.vitactip.poissons_ratio[None])), "vitactip.poissons_ratio is NaN"
        assert not math.isnan(float(self.phantom.poissons_ratio[0])), "phantom.poissons_ratio[0] is NaN"
        assert not math.isnan(float(self.phantom.poissons_ratio[1])), "phantom.poissons_ratio[1] is NaN"
        assert not math.isnan(float(self.coulomb_friction_coeff[None])), "coulomb_friction_coeff is NaN"
        assert not math.isnan(float(self.normal_stiffness[None])), "normal_stiffness is NaN"
        assert not math.isnan(float(self.tangential_stiffness[None])), "tangential_stiffness is NaN"
        assert not math.isnan(float(self.normal_damping[None])), "normal_damping is NaN"
    
    def print_params_short_from_log(self):
        print(f'vitactip.youngs_modulus: {ti.exp(self.vitactip_youngs_modulus_log[None]):0.16e} ({SYSTEM_PARAMS.vitactip.single_material.youngs_modulus:0.16e})')
        print(f'phantom.youngs_modulus_0: {ti.exp(self.phantom_youngs_modulus_0_log[None]):0.16e} ({SYSTEM_PARAMS.phantom.silicone.youngs_modulus:0.16e})')
        print(f'phantom.youngs_modulus_1: {ti.exp(self.phantom_youngs_modulus_1_log[None]):0.16e} ({SYSTEM_PARAMS.phantom.hard_plastic.youngs_modulus:0.16e})')
        print(f'vitactip.poissons_ratio: {ti.exp(self.vitactip_poissons_ratio_log[None]):0.16e} ({SYSTEM_PARAMS.vitactip.single_material.poissons_ratio:0.16e})')
        print(f'phantom.poissons_ratio_0: {ti.exp(self.phantom_poissons_ratio_0_log[None]):0.16e} ({SYSTEM_PARAMS.phantom.silicone.poissons_ratio:0.16e})')
        print(f'phantom.poissons_ratio_1: {ti.exp(self.phantom_poissons_ratio_1_log[None]):0.16e} ({SYSTEM_PARAMS.phantom.hard_plastic.poissons_ratio:0.16e})')
        print(f'coulomb_friction_coeff: {ti.exp(self.coulomb_friction_coeff_log[None]):0.16e} ({SYSTEM_PARAMS.contact.coulomb_friction_coeff:0.16e})')
        print(f'normal_stiffness: {ti.exp(self.normal_stiffness_log[None]):0.16e} ({SYSTEM_PARAMS.contact.normal_stiffness:0.16e})')
        print(f'tangential_stiffness: {ti.exp(self.tangential_stiffness_log[None]):0.16e} ({SYSTEM_PARAMS.contact.tangential_stiffness:0.16e})')
        print(f'normal_damping: {ti.exp(self.normal_damping_log[None]):0.16e} ({SYSTEM_PARAMS.contact.normal_damping:0.16e})')
    
    def set_up_torch_params(self):
        self.vitactip_youngs_modulus_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.phantom_youngs_modulus_0_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.phantom_youngs_modulus_1_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.vitactip_poissons_ratio_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.phantom_poissons_ratio_0_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.phantom_poissons_ratio_1_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.coulomb_friction_coeff_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.normal_stiffness_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.tangential_stiffness_log = ti.field(dtype=float, shape=(), needs_grad=True)
        self.normal_damping_log = ti.field(dtype=float, shape=(), needs_grad=True)

        self.vitactip_youngs_modulus_log[None] = ti.log(self.vitactip.youngs_modulus[None])
        self.phantom_youngs_modulus_0_log[None] = ti.log(self.phantom.youngs_modulus[0])
        self.phantom_youngs_modulus_1_log[None] = ti.log(self.phantom.youngs_modulus[1])
        self.vitactip_poissons_ratio_log[None] = ti.log(self.vitactip.poissons_ratio[None])
        self.phantom_poissons_ratio_0_log[None] = ti.log(self.phantom.poissons_ratio[0])
        self.phantom_poissons_ratio_1_log[None] = ti.log(self.phantom.poissons_ratio[1])
        self.coulomb_friction_coeff_log[None] = ti.log(self.coulomb_friction_coeff[None])
        self.normal_stiffness_log[None] = ti.log(self.normal_stiffness[None])
        self.tangential_stiffness_log[None] = ti.log(self.tangential_stiffness[None])
        self.normal_damping_log[None] = ti.log(self.normal_damping[None])

        self.vitactip_youngs_modulus_torch = torch.tensor(self.vitactip_youngs_modulus_log[None], requires_grad=False)
        self.phantom_youngs_modulus_0_torch = torch.tensor(self.phantom_youngs_modulus_0_log[None], requires_grad=False)
        self.phantom_youngs_modulus_1_torch = torch.tensor(self.phantom_youngs_modulus_1_log[None], requires_grad=False)
        self.vitactip_poissons_ratio_torch = torch.tensor(self.vitactip_poissons_ratio_log[None], requires_grad=False)
        self.phantom_poissons_ratio_0_torch = torch.tensor(self.phantom_poissons_ratio_0_log[None], requires_grad=False)
        self.phantom_poissons_ratio_1_torch = torch.tensor(self.phantom_poissons_ratio_1_log[None], requires_grad=False)
        self.coulomb_friction_coeff_torch = torch.tensor(self.coulomb_friction_coeff_log[None], requires_grad=False)
        self.normal_stiffness_torch = torch.tensor(self.normal_stiffness_log[None], requires_grad=False)
        self.tangential_stiffness_torch = torch.tensor(self.tangential_stiffness_log[None], requires_grad=False)
        self.normal_damping_torch = torch.tensor(self.normal_damping_log[None], requires_grad=False)

        self.torch_params = [
            self.vitactip_youngs_modulus_torch,
            self.phantom_youngs_modulus_0_torch,
            self.phantom_youngs_modulus_1_torch,
            self.vitactip_poissons_ratio_torch,
            self.phantom_poissons_ratio_0_torch,
            self.phantom_poissons_ratio_1_torch,
            self.coulomb_friction_coeff_torch,
            self.normal_stiffness_torch,
            self.tangential_stiffness_torch,
            self.normal_damping_torch
        ]

        if False:
            self.optimiser = optim.Adam(self.torch_params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8)

        self.optimiser = optim.Adam([
            {'params': [self.vitactip_youngs_modulus_torch], 'lr': 0},
            {'params': [self.phantom_youngs_modulus_0_torch], 'lr': 0},
            {'params': [self.phantom_youngs_modulus_1_torch], 'lr': 0},
            {'params': [self.vitactip_poissons_ratio_torch], 'lr': 0},
            {'params': [self.phantom_poissons_ratio_0_torch], 'lr': 0},
            {'params': [self.phantom_poissons_ratio_1_torch], 'lr': 0},
            {'params': [self.coulomb_friction_coeff_torch]},
            {'params': [self.normal_stiffness_torch]},
            {'params': [self.tangential_stiffness_torch]},
            {'params': [self.normal_damping_torch]}
        ], lr=1e-1, betas=(0.9, 0.999), eps=1e-8)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=1, gamma=0.1)
    
    def update_params(self, ts):
        try:
            assert not math.isnan(float(self.vitactip_youngs_modulus_log.grad[None])), "vitactip_youngs_modulus gradient is NaN"
            assert not math.isnan(float(self.phantom_youngs_modulus_0_log.grad[None])), "phantom_youngs_modulus_0 gradient is NaN"
            assert not math.isnan(float(self.phantom_youngs_modulus_1_log.grad[None])), "phantom_youngs_modulus_1 gradient is NaN"
            assert not math.isnan(float(self.vitactip_poissons_ratio_log.grad[None])), "vitactip_poissons_ratio gradient is NaN"
            assert not math.isnan(float(self.phantom_poissons_ratio_0_log.grad[None])), "phantom_poissons_ratio_0 gradient is NaN"
            assert not math.isnan(float(self.phantom_poissons_ratio_1_log.grad[None])), "phantom_poissons_ratio_1 gradient is NaN"
            assert not math.isnan(float(self.coulomb_friction_coeff_log.grad[None])), "coulomb_friction_coeff gradient is NaN"
            assert not math.isnan(float(self.normal_stiffness_log.grad[None])), "normal_stiffness gradient is NaN"
            assert not math.isnan(float(self.tangential_stiffness_log.grad[None])), "tangential_stiffness gradient is NaN"
            assert not math.isnan(float(self.normal_damping_log.grad[None])), "normal_damping gradient is NaN"
        except:
            self.courant_number /= 2
            self.retry = True
            self.clear_grad()
            self.set_dt(verbose=True)
            return

        self.optimiser.zero_grad()

        self.vitactip_youngs_modulus_torch.grad = torch.tensor(self.vitactip_youngs_modulus_log.grad[None])
        self.phantom_youngs_modulus_0_torch.grad = torch.tensor(self.phantom_youngs_modulus_0_log.grad[None])
        self.phantom_youngs_modulus_1_torch.grad = torch.tensor(self.phantom_youngs_modulus_1_log.grad[None])
        self.vitactip_poissons_ratio_torch.grad = torch.tensor(self.vitactip_poissons_ratio_log.grad[None])
        self.phantom_poissons_ratio_0_torch.grad = torch.tensor(self.phantom_poissons_ratio_0_log.grad[None])
        self.phantom_poissons_ratio_1_torch.grad = torch.tensor(self.phantom_poissons_ratio_1_log.grad[None])
        self.coulomb_friction_coeff_torch.grad = torch.tensor(self.coulomb_friction_coeff_log.grad[None])
        self.normal_stiffness_torch.grad = torch.tensor(self.normal_stiffness_log.grad[None])
        self.tangential_stiffness_torch.grad = torch.tensor(self.tangential_stiffness_log.grad[None])
        self.normal_damping_torch.grad = torch.tensor(self.normal_damping_log.grad[None])
        
        self.optimiser.step()
        
        self.vitactip_youngs_modulus_log[None] = self.vitactip_youngs_modulus_torch.item()
        self.phantom_youngs_modulus_0_log[None] = self.phantom_youngs_modulus_0_torch.item()
        self.phantom_youngs_modulus_1_log[None] = self.phantom_youngs_modulus_1_torch.item()
        self.vitactip_poissons_ratio_log[None] = self.vitactip_poissons_ratio_torch.item()
        self.phantom_poissons_ratio_0_log[None] = self.phantom_poissons_ratio_0_torch.item()
        self.phantom_poissons_ratio_1_log[None] = self.phantom_poissons_ratio_1_torch.item()
        self.coulomb_friction_coeff_log[None] = self.coulomb_friction_coeff_torch.item()
        self.normal_stiffness_log[None] = self.normal_stiffness_torch.item()
        self.tangential_stiffness_log[None] = self.tangential_stiffness_torch.item()
        self.normal_damping_log[None] = self.normal_damping_torch.item()
    
    @ti.kernel
    def set_optimisation_params_from_log(self):
        self.vitactip.youngs_modulus[None] += ti.exp(self.vitactip_youngs_modulus_log[None])
        self.phantom.youngs_modulus[0] += ti.exp(self.phantom_youngs_modulus_0_log[None])
        self.phantom.youngs_modulus[1] += ti.exp(self.phantom_youngs_modulus_1_log[None])
        self.vitactip.poissons_ratio[None] += ti.exp(self.vitactip_poissons_ratio_log[None])
        self.phantom.poissons_ratio[0] += ti.exp(self.phantom_poissons_ratio_0_log[None])
        self.phantom.poissons_ratio[1] += ti.exp(self.phantom_poissons_ratio_1_log[None])
        self.coulomb_friction_coeff[None] += ti.exp(self.coulomb_friction_coeff_log[None])
        self.normal_stiffness[None] += ti.exp(self.normal_stiffness_log[None])
        self.tangential_stiffness[None] += ti.exp(self.tangential_stiffness_log[None])
        self.normal_damping[None] += ti.exp(self.normal_damping_log[None])
    
    def randomise_contact_params(self):
        ns = SYSTEM_PARAMS.contact.normal_stiffness
        nd = SYSTEM_PARAMS.contact.normal_damping
        ts = SYSTEM_PARAMS.contact.tangential_stiffness
        cfc = SYSTEM_PARAMS.contact.coulomb_friction_coeff

        self.normal_stiffness[None] = random.uniform(ns * 0.5, ns * 1.5)
        self.normal_damping[None] = random.uniform(nd * 0.5, nd * 1.5)
        self.tangential_stiffness[None] = random.uniform(ts * 0.5, ts * 1.5)
        self.coulomb_friction_coeff[None] = random.uniform(cfc * 0.5, cfc * 1.5)
    
    def print_contact_params(self):
        ns = SYSTEM_PARAMS.contact.normal_stiffness
        nd = SYSTEM_PARAMS.contact.normal_damping
        ts = SYSTEM_PARAMS.contact.tangential_stiffness
        cfc = SYSTEM_PARAMS.contact.coulomb_friction_coeff

        print(f"Normal Stiffness: {self.normal_stiffness[None] / ns}")
        print(f"Normal Damping: {self.normal_damping[None] / nd}")
        print(f"Tangential Stiffness: {self.tangential_stiffness[None] / ts}")
        print(f"Coulomb Friction Coefficient: {self.coulomb_friction_coeff[None] / cfc}")
    
    def save_final_params(self):
        results = {
            "vitactip": {
                "youngs_modulus": self.vitactip.youngs_modulus[None],
                "poissons_ratio": self.vitactip.poissons_ratio[None]
            },
            "phantom": {
                "youngs_modulus_0": self.phantom.youngs_modulus[0],
                "youngs_modulus_1": self.phantom.youngs_modulus[1],
                "poissons_ratio_0": self.phantom.poissons_ratio[0],
                "poissons_ratio_1": self.phantom.poissons_ratio[1]
            },
            "contact": {
                "coulomb_friction_coeff": self.coulomb_friction_coeff[None],
                "normal_stiffness": self.normal_stiffness[None],
                "tangential_stiffness": self.tangential_stiffness[None],
                "normal_damping": self.normal_damping[None]
            }
        }
        results = {
            k1: {
                k2: float(v2) 
                for k2, v2 in v1.items()
            }
            for k1, v1 in results.items()
        }
        with open(SYSTEM_PARAMS.files.domain_adaptation_results, "w") as f:
            json.dump(results, f, indent=4, cls=ScientificNotationEncoder)
    
    def set_dt(self, verbose=False):
        if len(self.state_dicts[self.trajectory_ix[None]]) > 0:
            tumour_modulus = self.phantom.youngs_modulus[1]
        else:
            tumour_modulus = self.phantom.youngs_modulus[0]
        dt = calculate_cfl_timestep(
            phantom_healthy_youngs_modulus=self.phantom.youngs_modulus[0],
            phantom_tumour_youngs_modulus=tumour_modulus,
            vitactip_youngs_modulus=self.vitactip.youngs_modulus[None],
            courant_number=self.courant_number,
            verbose=verbose,
        )
        self.dt[None] = dt
        self.phantom.dt[None] = dt
        self.vitactip.dt[None] = dt
        if verbose:
            print(f'dt={dt:0.3e} s')
    
    def get_keypoint_indices_and_validate(self):
        self.vitactip.test_mapping_from_global_space_to_camera_space()
        self.vitactip.extract_markers(0)
        self.compute_mapping_between_experimental_and_sim_markers()
        self.vitactip.save_predicted_markers_to_image()
        self.vitactip.get_keypoint_idxs()
        initial_markers = self.vitactip.deformed_markers.to_numpy()
        with open(SYSTEM_PARAMS.files.sim_markers_initial_positions, "wb") as f:
            pickle.dump(initial_markers, f)
        self.vitactip.compute_vertices_E()
        with open(SYSTEM_PARAMS.files.vitactip_points_E, "wb") as f:
            pickle.dump(self.vitactip.vertices_E.to_numpy(), f)

    @ti.kernel
    def log(self):
        if self.interpolation_valid[None] == 1:
            print(0)

    def domain_adaptation(self):
        losses_per_trajectory = []
        for opts in range(SYSTEM_PARAMS.contact.num_opt_steps):
            print(f"optimisation step: {opts} / {SYSTEM_PARAMS.contact.num_opt_steps - 1}")
            for i in range(self.trajectories.shape[0]):
            # for i in range(1, 2):
                print(f'trajectory {i}: {self.trajectory_names[i]}')
                self.trajectory_ix[None] = i
                self.set_up_initial_positions_state_and_trajectory()
                self.reset_pid_controller()
                self.visualisation_reset_scene()
                self.reset_exp_sim_traj()
                self.vitactip.extract_markers(0)
                self.compute_mapping_between_experimental_and_sim_markers()
                self.set_dt(verbose=True)
                self.fp()
                print("forward")
                ts = 0
                while self.last_target_reached[None] != 1:
                    self.pid_controller_1()
                    self.pid_controller_2(ts)
                    self.pid_controller_3()
                    self.vitactip.set_pose_control_1()
                    self.vitactip.set_pose_control_2()
                    self.vitactip.set_pose_control_3()
                    self.forward_pass_common_part(ts)
                    self.memory_to_cache(ts)
                    self.vitactip.extract_markers(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.visualisation_update_gui(ts)
                    self.maybe_save_tactile_sensor_mesh_to_pickle(ts)
                    ts += 1
                
                total_ts = ts
                self.bp()
                self.reset_loss()
                self.batch_loss.fill(0.0)
                self.clear_grad()
                self.prev_loss[None] = 0.0
                self.trajectory_loss[None] = 0.0
                # continue
                print("backward")
                passes = 0
                ts = total_ts-1
                while ts >= 0:
                    self.reset_loss()
                    self.memory_from_cache(ts)
                    self.forward_pass_common_part(ts)
                    self.vitactip.extract_markers(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.interpolate_experimental_frame(ts)
                    self.compute_marker_loss_1()
                    self.compute_marker_loss_2()
                    if SYSTEM_PARAMS.optimisation.enable_loss_2 == 1:
                        self.compute_marker_loss_3()
                        self.compute_marker_loss_4()
                    self.compute_marker_loss_5()
                    self.visualisation_update_gui(ts)
                    if False:
                        print(f"mean error 1: {self.mean_error_1[None]}")
                        print(f"mean error 2: {self.mean_error_2[None]}")
                        print(f"exp frame: {self.cur_exp_frame[None]}")
                        print(f"sim frame: {ts}")
                    self.loss.grad[None] = 1.0
                    self.compute_marker_loss_5.grad()
                    if SYSTEM_PARAMS.optimisation.enable_loss_2 == 1:
                        self.compute_marker_loss_4.grad()
                        self.compute_marker_loss_3.grad()
                    self.compute_marker_loss_2.grad()
                    self.compute_marker_loss_1.grad()
                    self.vitactip.extract_markers.grad(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.backward_pass_common_part()
                    passes += 1
                    if (
                        passes % SYSTEM_PARAMS.optimisation.mini_batch_size == 0
                        or (SYSTEM_PARAMS.optimisation.mini_batch_size > total_ts and ts == 0)
                    ):
                        if SYSTEM_PARAMS.optimisation.calibrate_learning_rates == 1:
                            self.save_gradients_for_calibration()
                        print(f'mini batch loss: {self.batch_loss[None]:0.3e}')
                        print(f'mini batch loss 1: {self.batch_loss_1[None]:0.3e}')
                        print(f'mini batch loss 2: {self.batch_loss_2[None]:0.3e}')
                        if opts > 0:
                            self.update_params(ts)
                        if self.retry:
                            break
                        self.reset_state()
                        self.set_optimisation_params_from_log()
                        self.print_params_short()
                        self.set_dt(verbose=True)
                        self.clear_grad()
                        self.reset_batch_loss()
                    if False:
                        print()
                    if not self.retry:
                        ts -= 1
                    else:
                        self.retry = False
                losses_per_trajectory.append(float(self.trajectory_loss[None]))
            previous_lr = self.optimiser.param_groups[-1]['lr']
            if opts > 0:
                self.scheduler.step()
            current_lr = self.optimiser.param_groups[-1]['lr']
            print(f"lr: {previous_lr:0.3e} -> {current_lr:0.3e}")
            print(
                f"optimisation step: {opts} / {SYSTEM_PARAMS.contact.num_opt_steps - 1} done"
            )
        print("optimisation loop done")
        print(f"courant_number: {self.courant_number}")
        losses_per_trajectory = np.array(losses_per_trajectory)
        losses_per_opt_step = losses_per_trajectory.reshape(-1, 4).sum(axis=1)
        xs = [
            losses_per_opt_step,
            losses_per_trajectory
        ]
        names = [
            "per_opt_step",
            "per_trajectory"
        ]
        for i in range(len(xs)):
            x = xs[i]
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(len(x))), x)
            plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.grid(True)
            plt.xlabel('batch index')
            plt.ylabel('batch loss')
            plt.title('batch loss over time')
            plt.savefig(SYSTEM_PARAMS.files.losses.format(names[i]))
            plt.show()
        self.save_final_params()
        print("all done")

    def collect_training_data(self):
        self.clear_temp_images()
        self.clear_npz()
        for j in range(SYSTEM_PARAMS.contact.num_training_trajectories):
            print(f"training trajectory: {j} / {SYSTEM_PARAMS.contact.num_training_trajectories - 1}")
            self.randomise_train_step()
            for i in range(0, 2):
                self.randomise_contact_params()
                self.trajectory_ix[None] = i
                self.set_up_initial_positions_state_and_trajectory()
                self.reset_pid_controller()
                self.visualisation_reset_scene()
                self.reset_exp_sim_traj()
                self.vitactip.extract_markers(0)
                self.compute_mapping_between_experimental_and_sim_markers()
                self.set_dt(verbose=True)
                self.fp()
                self.print_contact_params()
                for ts in range(SYSTEM_PARAMS.meta.max_timesteps_per_trajectory):
                    self.pid_controller_1()
                    self.pid_controller_2(ts)
                    self.pid_controller_3()
                    self.vitactip.set_pose_control_1()
                    self.vitactip.set_pose_control_2()
                    self.vitactip.set_pose_control_3()
                    self.forward_pass_common_part(ts)
                    self.copy_frame()
                    self.vitactip.extract_markers(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.vitactip.mark_surface_nodes_in_contact(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.visualisation_update_gui(ts)
                    if (
                        self.current_target_idx[None] == 4 
                        and ts % 2 == 0
                    ):
                        self.record_training_data_point(j, ts)
                self.write_training_data_to_file(j*2 + (i-1))
                # self.write_training_data_to_file(999)
                
                self.reset_loss()
                self.batch_loss.fill(0.0)
                self.clear_grad()
                self.prev_loss[None] = 0.0
                self.trajectory_loss[None] = 0.0
                
            print(
                f"training trajectory: {j} / {SYSTEM_PARAMS.contact.num_training_trajectories - 1} done"
            )
        print("training data collection done")
        print("all done")


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

    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()
    contact_model.set_up_torch_params()
    contact_model.collect_training_data()
