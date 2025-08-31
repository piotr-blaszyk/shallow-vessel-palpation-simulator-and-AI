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
import cProfile, pstats, sys
from tqdm import tqdm

from difftactile.main.constants import *
from difftactile.main.constants_bo_gp import *
from difftactile.object_model.vein import Vein
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.sensor_model.vitactip import ViTacTip
from difftactile.object_model.phantom import Phantom
from difftactile.main.cfl_and_contact_params_estimation import *
from difftactile.main.apply_scaling import ScientificNotationEncoder
from difftactile.main.synthetic_image_generator import *
from difftactile.data_analysis.experiment.adjacency import *
from difftactile.data_analysis.experiment.bo_gp import *

RUN_ON_LAB_MACHINE = True


@ti.data_oriented
class Contact:
    def __init__(self):
        self.vein = Vein()
        self.phantom = Phantom(vein=self.vein)
        self.vitactip = ViTacTip()
        self.compute_sensor_bounds()
        self.fisheye_model = FisheyeModelNoTaichi()
        self.set_up_system_params()
        self.load_system_identification_data()
        self.set_up_initial_positions_and_trajectory_first_init_only()
        self.set_up_trajectories_and_phantom_states()
        self.set_up_initial_positions_state_and_trajectory()
        self.set_up_collision_detection()
        self.set_up_pid()
        self.set_up_snapshot()
        self.set_up_loss_computation()
        self.visualisation_initialise()
        self.training_data_collection_initialise()
        self.foo()
        self.bo = BoGp()
    
    # def write_da_total_loss_to_file(self):
    #     target_data = {
    #         'target': sum(self.da_losses),
    #     }
    #     with open(self.target_path, "w") as f:
    #         json.dump(target_data, f, indent=4)
    #     self.da_losses = []
    
    def handle_da_loss(self, ts):
        trajectory_ix = self.trajectory_ix[None]
        trajectory_name = self.trajectory_names[trajectory_ix]
        if trajectory_name == 'slide':
            target_timestep = self.photo_timesteps[trajectory_name]
            act_now = ts == target_timestep
            if act_now:
                self.compute_da_loss()
        else:
            act_now = self.last_target_reached[None] == 1
            if act_now:
                self.compute_da_loss()
        return act_now
    
    def compute_da_loss(self):
        h = int(SYSTEM_PARAMS.fisheye_model.target_image_height)
        self.move_og_resolution()
        sim_points = self.sim_markers_deformed.to_numpy()
        self.move_ti_resolution()
        sim_points[:, 1] = h-sim_points[:, 1]
        _, sim_points_reordered, _ = Adjacency.get_graph_connectivity(sim_points)
        trajectory_ix = self.trajectory_ix[None]
        trajectory_name = self.trajectory_names[trajectory_ix]
        file_path = self.da_npz_paths[trajectory_name]
        data = np.load(file_path)
        exp_points = data['points']
        a = sim_points_reordered
        b = exp_points
        mae = np.linalg.norm(a-b, axis=1).mean()
        self.generate_validation_img(
            a, b,
            img_in=self.default_photo,
            img_out=self.da_overlay.format(trajectory_name),
        )
        self.da_losses.append(mae)
    
    def generate_validation_img(self, points1, points2, img_in, img_out):
        img = cv2.imread(img_in)
        points1 = points1.astype(np.int32)
        points2 = points2.astype(np.int32)
        for point in points1:
            cv2.circle(img, tuple(point), radius=3, color=(0, 0, 255), thickness=-1)  # BGR format: red
        for point in points2:
            cv2.circle(img, tuple(point), radius=3, color=(0, 255, 0), thickness=-1)  # BGR format: green
        cv2.imwrite(img_out, img)
    
    @ti.kernel
    def update_vitactip_tip_point(self):
        self.vitactip_tip_point[0] = self.vitactip.vertices_undeformed_A[
            self.num_sub_frames-1, 
            self.vitactip.tip_ix[None],
        ]
    
    @ti.kernel
    def update_clock_arm_points_3d(self):
        for i in range(self.vitactip.clock_arms_node_idxs.shape[0]):
            node_idx = self.vitactip.clock_arms_node_idxs[i]
            vertex = self.vitactip.vertices_undeformed_A[
                self.num_sub_frames-1,
                node_idx,
            ]
            self.clock_arm_points_3d[i] = vertex
    
    def foo(self):
        self.da_npz_paths = {
            'press': f'{SYSTEM_PARAMS.files.da_press_npz}',
            'twist_z': f'{SYSTEM_PARAMS.files.da_twist_z_npz}',
            'twist_x': f'{SYSTEM_PARAMS.files.da_twist_x_npz}',
            'slide': f'{SYSTEM_PARAMS.files.da_slide_npz}',
        }
        self.num_sub_frames = SYSTEM_PARAMS.contact.num_sub_frames
        self.max_ts = SYSTEM_PARAMS.meta.max_timesteps_per_trajectory
        self.vitactip_tip_point = ti.Vector.field(
            3,
            dtype=float,
            shape=(1,),
            needs_grad=False,
        )
        self.clock_arm_points_3d = ti.Vector.field(
            3,
            dtype=float,
            shape=(2,),
            needs_grad=False,
        )
        self.vitactip_vertices_temp = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.vitactip.vertices_deformed_A.shape[1],),
            needs_grad=False,
        )
        self.all_points = []
        self.collision_ixs = [0]
        self.collision_resolvers = [
            self.collision0,
            self.collision1,
            self.collision2,
        ]
        self.collision_detectors = [
            self.check_collision0,
            self.check_collision1,
            self.check_collision2,
        ]
    
    def detect_collisions(self, f):
        for i in range(len(self.collision_ixs)):
            ix = self.collision_ixs[i]
            self.collision_detectors[ix](f)
    
    def resolve_collisions(self, f):
        for i in NP_RNG.permutation(len(self.collision_ixs)):
            ix = self.collision_ixs[i]
            self.collision_resolvers[ix](f)

    def vein_sparse_to_dense_init(self):
        self.num_veins = SYSTEM_PARAMS.meta.max_num_veins
        self.vein_counts = ti.field(int, (self.num_veins,), needs_grad=False)
        self.vein_indices = ti.field(
            int, (self.num_veins, self.phantom.num_particles), needs_grad=False
        )
        self.max_vein_count = ti.field(int, (), needs_grad=False)
    
    def vein_sparse_to_dense(self):
        vein_titles = self.phantom.vein_titles.to_numpy()
        unique_titles, counts = np.unique(vein_titles, return_counts=True)
        vein_counts = np.zeros(shape=(self.num_veins,), dtype=int)
        for vein_ix, count in zip(unique_titles, counts):
            if vein_ix != -1:
                vein_counts[vein_ix] = count
        self.vein_counts.from_numpy(vein_counts)
        self.max_vein_count[None] = np.max(vein_counts)
        vein_counts_temp = np.zeros(shape=(self.num_veins,), dtype=int)
        vein_indices = -np.ones(shape=(self.num_veins, self.phantom.num_particles), dtype=int)
        for particle_ix in range(len(vein_titles)):
            vein_ix = vein_titles[particle_ix]
            if vein_ix != -1:
                vein_particle_ix = vein_counts_temp[vein_ix]
                vein_indices[vein_ix, vein_particle_ix] = particle_ix
                vein_counts_temp[vein_ix] += 1
        self.vein_indices.from_numpy(vein_indices)
    
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
        self.vein_polyline_data = []
        self.vein_polyline_mask_data = []
        self.target_id_data = []
        self.vein_cx_A = None
        self.target_3_ts = 12
        self.target_4_ts = 226
        # self.vein_sparse_to_dense_init()
        self.generate_tumour = False

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
        self.loss = ti.field(float, (), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.loss_1 = ti.field(float, (), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.loss_2 = ti.field(float, (), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.trajectory_loss = ti.field(float, (), needs_grad=False)
        self.batch_loss = ti.field(float, (), needs_grad=False)
        self.batch_loss_1 = ti.field(float, (), needs_grad=False)
        self.batch_loss_2 = ti.field(float, (), needs_grad=False)
        self.squared_error_sum_1 = ti.field(dtype=float, shape=(), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.squared_error_sum_2 = ti.field(dtype=float, shape=(), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.error_sum_1 = ti.field(dtype=float, shape=(), needs_grad=False)
        self.error_sum_2 = ti.field(dtype=float, shape=(), needs_grad=False)
        self.mean_error_1 = ti.field(dtype=float, shape=(), needs_grad=False)
        self.mean_error_2 = ti.field(dtype=float, shape=(), needs_grad=False)

    def set_up_collision_detection(self):
        self.triangle_ix_contact_0 = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.triangle_ix_contact_1 = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.triangle_ix_contact_2 = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.vein.particles_A.shape[0],
            ),
            needs_grad=False,
        )

    def set_up_system_params(self):
        self.collision2_contact_flat = ti.field(dtype=int, shape=(), needs_grad=False)
        self.target_path = SYSTEM_PARAMS.files.bo_gp_target_json
        self.use_bo = SYSTEM_PARAMS.meta.load_params_from_bo == 1
        self.da_overlay = SYSTEM_PARAMS.files.da_overlay
        self.photo_timesteps = {
            # 'press': 35,
            # 'twist_z': 180,
            # 'twist_x': 51,
            'slide': 327,
        }
        self.da_losses = []
        self.dist_sf = SYSTEM_PARAMS.meta.distance_scaling_factor
        self.sensor_r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        default_photo = SYSTEM_PARAMS.files.flat_sensor_default_state
        dir = SYSTEM_PARAMS.files.da_dir
        self.default_photo = f'{dir}{default_photo}'
        self.num_contact_pairs = SYSTEM_PARAMS.meta.num_contact_pairs
        self.trajectory_ix = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dt = ti.field(dtype=float, shape=(), needs_grad=False)
        self.dt[None] = SYSTEM_PARAMS.contact.dt_override
        self.normal_stiffness = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.normal_damping = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.tangential_stiffness = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.coulomb_friction_coeff = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=SYSTEM_PARAMS.meta.enable_grad)
        self.normal_stiffness.from_numpy(np.array(SYSTEM_PARAMS.contact.normal_stiffness))
        self.normal_damping.from_numpy(np.array(SYSTEM_PARAMS.contact.normal_damping))
        self.tangential_stiffness.from_numpy(np.array(SYSTEM_PARAMS.contact.tangential_stiffness))
        self.coulomb_friction_coeff.from_numpy(np.array(SYSTEM_PARAMS.contact.coulomb_friction_coeff))
        self.gradients_printed = False
        self.courant_number = SYSTEM_PARAMS.meta.target_courant_number
        self.retry = False
        phantom_closest_vertex = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        self.phantom_closest_vertex = np.array(phantom_closest_vertex, dtype=float)
        phantom_dimensions = SYSTEM_PARAMS_COMPUTED.phantom_dimensions
        self.phantom_dimensions = np.array(phantom_dimensions, dtype=float)
        self.gap = SYSTEM_PARAMS.geometry.gap

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
        self.exp_vein_3d_coords_E_np = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(self.exp_vein_2d_coords)
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
        self.validation_point_3d_E_np = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(validation_point_2d)
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

    def generate_trajectories(self):
        self.trajectory_names = [
            'press',
            'twist_z',
            'twist_x',
            'slide',
        ]
        trajectories_python_array = [
            self.get_press_trajectory(),
            self.get_twist_z_trajectory(),
            self.get_twist_x_trajectory(),
            self.get_slide_trajectory(),
        ]
        self.set_trajectories(trajectories_python_array)

        # self.state_dicts = []
        # for i in range(len(trajectories_python_array)):
        #     self.state_dicts.append(
        #         self.generate_random_state_dicts()
        #     )    
    
    def get_vitactip_orientation(self):
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        return og_r

    def get_random_slide_params(self):
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        xr = NP_RNG.uniform(-10, 10)
        yr = NP_RNG.uniform(-10, 10)
        zr = NP_RNG.uniform(0, 60)
        rand_r = R.from_euler(seq="xyz", angles=[0, 0, zr], degrees=True)
        og_r = og_r * rand_r
        slide_r = og_r
        srq = slide_r.as_quat()

        press_depth_surface = SYSTEM_PARAMS.geometry.gap
        press_depth_1 = SYSTEM_PARAMS.trajectory.press_depth_slide
        if True:
            k_0 = SYSTEM_PARAMS.trajectory.press_depth_offset_0
            k_1 = SYSTEM_PARAMS.trajectory.press_depth_offset_1
            press_depth_rand = NP_RNG.uniform(-k_0, k_1)
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
        d_single = NP_RNG.uniform(0.5 * r, 2 * r)
        trajectory = [
            [x, y, z, *srq],
            [x, y, z - press_depth_surface, *srq],
            [x, y, z - press_depth_1, *srq],
        ]
        return trajectory
        # 0,1,2,3,4!,5,6,7!,8,9,10!,11,12,13!,14,15,16!
        # x = 4 + 3*k, k >= 0
        # x >= 4 and (x - 4) % 3 == 0
        # ts >= 4 and (ts - 4) % 3 == 0
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
            a2 = a + x_dir * d_single
            b2 = b + y_dir * dy
            if xy_i == 0 or xy_i == 2:
                trajectory.append(
                    [a2, (b+b2)/2, c, *srq]
                )
            trajectory.append(
                [a2, b2, c, *srq]
            )
            xy_i += 1
            xy_i %= 4
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
        return trajectory
        
        # Calculate maximum possible magnitude based on sensor bounds
        x_min, x_max = self.sensor_x_range_world
        y_min, y_max = self.sensor_y_range_world
        max_dx = x_max - x_min
        max_dy = y_max - y_min
        max_magnitude = min(max_dx, max_dy) / 2  # Conservative estimate
        
        # Generate remaining trajectory points using polar coordinates
        current_x, current_y = x, y
        while len(trajectory) < self.trajectories.shape[1]:
            magnitude = NP_RNG.uniform(0, max_magnitude)
            
            # Keep trying angles until we find one that keeps point in bounds
            while True:
                angle = NP_RNG.uniform(0, 2 * math.pi)
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
    
    def get_straight_line_slide_trajectory(self):
        (
            srq,
            press_depth_surface,
            press_depth_1
        ) = self.get_random_slide_params()
        x, y, z = self.vitactip_tip_pose[:3]
        r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        y_span = SYSTEM_PARAMS.geometry.phantom_y_length
        y_final = y+r+y_span+r
        trajectory = [
            [x, y, z, *srq],
            [x, y_final, z, *srq],
        ]
        return trajectory
    
    def get_press_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        x = cvx+dx/2
        y = cvx+dy/2
        z = cvz+dz+self.gap
        press_depth = 0.004*self.dist_sf
        ori = ori.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y, z-self.gap, *ori],
            [x, y, z-self.gap-press_depth, *ori],
        ]
        return trajectory
    
    def get_twist_z_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        x = cvx+dx/2
        y = cvx+dy/2
        z = cvz+dz+self.gap
        press_depth = 0.004*self.dist_sf
        angle = 30
        z_rot = R.from_euler(seq="xyz", angles=[0, 0, -angle], degrees=True)
        ori2 = ori * z_rot
        ori = ori.as_quat()
        ori2 = ori2.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y, z-self.gap, *ori],
            [x, y, z-self.gap-press_depth, *ori],
            [x, y, z-self.gap-press_depth, *ori2],
        ]
        return trajectory
    
    def get_twist_x_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        x = cvx+dx/2
        y = cvx+dy/3
        z = cvz+dz+self.gap
        press_depth = 0.002*self.dist_sf
        angle = 20
        z_rot = R.from_euler(seq="xyz", angles=[angle, 0, 0], degrees=True)
        ori2 = ori * z_rot
        ori = ori.as_quat()
        ori2 = ori2.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y, z-self.gap, *ori],
            [x, y, z-self.gap-press_depth, *ori],
            [x, y, z-self.gap-press_depth, *ori2],
        ]
        return trajectory
    
    def get_slide_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        press_depth = 0.003*self.dist_sf
        r = self.sensor_r
        y_span = SYSTEM_PARAMS.geometry.phantom_y_length
        x = cvx+dx/2
        y = cvx-self.sensor_r
        z = cvz+dz-press_depth
        y2 = y+r+y_span+r
        ori = ori.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y2, z, *ori],
        ]
        return trajectory

    def set_up_initial_positions_state_and_trajectory(self):
        sensor_dome_tip_initial_pose = self.trajectories[self.trajectory_ix[None], 0].to_numpy()
        self.vitactip.set_up_pose(sensor_dome_tip_initial_pose)
        self.tactile_sensor_initial_position[0] = ti.Vector(
            sensor_dome_tip_initial_pose[:3]
        )
        self.phantom_initial_position[0] = ti.Vector(self.phantom_centroid_pose[:3])
        self.phantom.initialise_point_cloud()
    
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
        return []
        if not self.generate_tumour:
            return []

        state_dicts = []
        num_veins = SYSTEM_PARAMS.meta.max_num_veins
        placed_cy_values = []
        min_separation = SYSTEM_PARAMS.geometry.min_vein_separation
        for i in range(num_veins):
            theta_rand = NP_RNG.uniform(-10, 10)
            cz_offset = SYSTEM_PARAMS.geometry.phantom_z_length / 2 - SYSTEM_PARAMS.geometry.vein.depth_beneath_surface
            cx = self.sensor_x_range_phantom[0]

            while True:
                cy = NP_RNG.uniform(*self.sensor_y_range_phantom)
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
        self.detect_collisions(f)
        self.resolve_collisions(f)
        self.phantom.grid_op(f)
        self.phantom.g2p(f)
        self.vitactip.update_external_forces(f)

    def update_grad(self, f):
        self.vitactip.update_external_forces.grad(f)
        self.phantom.g2p.grad(f)
        self.phantom.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision0.grad(f)
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
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
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

    def reset_state(self):
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.triangle_ix_contact_0.fill(-1)
        self.triangle_ix_contact_1.fill(-1)
        self.triangle_ix_contact_2.fill(-1)
        self.collision2_contact_flat.fill(0)
        if False:
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
        self, 
        signed_distance, 
        surface_normal, 
        relative_velocity, 
        contact_pair_ix,
    ):
        i = contact_pair_ix
        tangential_force = ti.Vector([0.0, 0.0, 0.0])
        tangential_velocity = ti.Vector([0.0, 0.0, 0.0])
        contact_relative_velocity = relative_velocity
        normal_velocity_magnitude = ti.max(
            surface_normal.dot(contact_relative_velocity), 0
        )
        normal_force = (
            -(
                self.normal_stiffness[i]
                + self.normal_damping[i] * normal_velocity_magnitude
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
                    self.tangential_stiffness[i] * tangential_velocity_magnitude,
                    self.coulomb_friction_coeff[i]
                    * normal_force.norm(SYSTEM_PARAMS.contact.norm_eps),
                )
            )
        total_contact_force = normal_force + tangential_force
        return total_contact_force, normal_force, tangential_force

    @ti.kernel
    def check_collision0(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                closest_triangle_ix = self.vitactip.find_closest(
                    grid_node_position, frame
                )
                self.triangle_ix_contact_0[frame, i, j, k] = closest_triangle_ix
    
    @ti.kernel
    def check_collision1(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                closest_triangle_ix = self.vein.find_closest(
                    grid_node_position
                )
                self.triangle_ix_contact_1[frame, i, j, k] = closest_triangle_ix
    
    @ti.kernel
    def check_collision2(self, frame: ti.i32):
        for i in range(self.vein.particles_A.shape[0]):
            point = self.vein.particles_A[i]
            closest_triangle_ix = self.vitactip.find_closest(point, frame)
            self.triangle_ix_contact_2[frame, i] = closest_triangle_ix

    @ti.kernel
    def collision0(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
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
                closest_triangle_ix = self.triangle_ix_contact_0[frame, i, j, k]
                if closest_triangle_ix != -1:
                    (
                        penetration_depth,
                        surface_normal,
                        relative_velocity,
                        is_in_contact,
                    ) = self.vitactip.find_sdf(
                        grid_node_position,
                        grid_node_velocity,
                        closest_triangle_ix,
                        frame,
                    )
                    if is_in_contact:
                        total_contact_force, _, _ = self.calculate_contact_force(
                            penetration_depth,
                            -1*surface_normal,
                            -1*relative_velocity,
                            contact_pair_ix=0,
                        )
                        self.phantom.update_contact_impulse(
                            total_contact_force, frame, i, j, k
                        )
                        self.vitactip.update_contact_force(
                            closest_triangle_ix, -1*total_contact_force, frame
                        )
    
    @ti.kernel
    def collision1(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
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
                closest_triangle_ix = self.triangle_ix_contact_1[frame, i, j, k]
                if closest_triangle_ix != -1:
                    (
                        penetration_depth,
                        surface_normal,
                        relative_velocity,
                        is_in_contact,
                    ) = self.vein.find_sdf(
                        grid_node_position,
                        grid_node_velocity,
                        closest_triangle_ix,
                    )
                    if is_in_contact:
                        total_contact_force, _, _ = self.calculate_contact_force(
                            penetration_depth,
                            -1*surface_normal,
                            -1*relative_velocity,
                            contact_pair_ix=1,
                        )
                        self.phantom.update_contact_impulse(
                            total_contact_force, frame, i, j, k
                        )
    
    @ti.kernel
    def collision2(self, frame: ti.i32):
        for i in range(self.vein.particles_A.shape[0]):
            point = self.vein.particles_A[i]
            velocity = ti.Vector([0.0, 0.0, 0.0])
            closest_triangle_ix = self.triangle_ix_contact_2[frame, i]
            if closest_triangle_ix != -1:
                (
                    penetration_depth,
                    surface_normal,
                    relative_velocity,
                    is_in_contact,
                ) = self.vitactip.find_sdf(
                    point,
                    velocity,
                    closest_triangle_ix,
                    frame,
                )
                if is_in_contact:
                    self.collision2_contact_flat[None] = 1
                    total_contact_force, _, _ = self.calculate_contact_force(
                        penetration_depth, 
                        -1*surface_normal, 
                        -1*relative_velocity,
                        contact_pair_ix=2,
                    )
                    self.vitactip.update_contact_force(
                        closest_triangle_ix, 
                        -1*total_contact_force, 
                        frame,
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
        current_pos = self.vitactip.vertices_undeformed_A[0, self.vitactip.tip_ix[None]]
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
            # print(
            #     f"target {self.current_target_idx[None]} ({target}) reached at time step {ts}!"
            # )
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

    @ti.kernel
    def copy_vitactip_vertices(self):
        for i in range(self.vitactip.vertices_deformed_A.shape[1]):
            point = self.vitactip.vertices_deformed_A[
                self.num_sub_frames-1,
                i
            ]
            self.vitactip_vertices_temp[i] = point

    def record_vitactip_mesh(self):
        self.copy_vitactip_vertices()
        self.all_points.append(
            self.vitactip_vertices_temp.to_numpy()
        )
    
    def write_vitactip_mesh_to_file(self):
        all_points = np.stack(self.all_points, axis=0)
        path = SYSTEM_PARAMS.files.vitactip_mesh_npz
        np.savez(
            path,
            all_points=all_points,
        )
        self.all_points = []

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
        markers[:, 1] = h-markers[:, 1]
        self.marker_data.append(markers)
        markers_img = np.zeros((h, w), dtype=np.uint8)
        for point in markers:
            x, y = int(point[0]), int(point[1])
            cv2.circle(markers_img, (x, y), radius=1, color=255, thickness=-1)
        markers_mask = SyntheticImageGenerator.compute_mask(h, w, markers)

        vein = self.vein_2d_projection.to_numpy()
        vein[:, :, 1] = h-vein[:, :, 1]
        # vein_counts = self.vein_counts.to_numpy()
        # vein_python_arr = []
        # for i in range(vein.shape[0]):
        #     num_points = vein_counts[i]
        #     single_vein_points = vein[i, :num_points]
        #     vein_python_arr.append(
        #         single_vein_points
        #     )
        # vein = vein_python_arr
        # vein_polyline_python_arr = []
        # for i in range(len(vein)):
        #     single_vein = vein[i]
        #     single_vein = SyntheticImageGenerator.filter_using_mask(markers_mask, single_vein)
        #     polyline_points = SyntheticImageGenerator.fit_polynomial(single_vein)
        #     vein_polyline_python_arr.append(polyline_points)
        # vein_polyline_np, vein_polyline_mask = SyntheticImageGenerator.create_padded_array_with_mask(
        #     vein_polyline_python_arr, 
        #     k=SYSTEM_PARAMS.meta.polyline_num
        # )
        if self.collision2_contact_flat[None] == 1:
            vein_mask = np.ones(shape=(vein.shape[0], vein.shape[1]), dtype=bool)
        else:
            vein_mask = np.zeros(shape=(vein.shape[0], vein.shape[1]), dtype=bool)
        self.vein_polyline_data.append(vein)
        self.vein_polyline_mask_data.append(vein_mask)

        target_id_arr = np.array([
            self.current_target_idx[None]
        ])
        self.target_id_data.append(target_id_arr)
    
    @staticmethod
    def get_endpoints(points):
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

    def write_training_data_to_file(self, training_iteration, traj_ix):
        directory = SYSTEM_PARAMS.files.dataset_root
        file = SYSTEM_PARAMS.files.dataset_data_point.format(
            training_iteration
        )
        path = f'{directory}/{file}'

        markers_array, markers_mask = SyntheticImageGenerator.create_padded_array_with_mask(self.marker_data)
        vein_polyline_array = np.array(self.vein_polyline_data)
        vein_polyline_mask = np.array(self.vein_polyline_mask_data)
        target_id_array = np.array(self.target_id_data)
        trajectory_type = np.array([traj_ix], dtype=int)

        np.savez(
            path,
            markers=markers_array,
            markers_mask=markers_mask,
            vein_polyline=vein_polyline_array,
            vein_polyline_mask=vein_polyline_mask,
            target_id_array=target_id_array,
            trajectory_type=trajectory_type
        )
        
        self.marker_data = []
        self.vein_polyline_data = []
        self.vein_polyline_mask_data = []
        self.target_id_data = []

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

    def maybe_save_tactile_sensor_mesh_to_pickle(self, ts):
        if self.mesh_needs_to_be_saved[None] == 1:
            particles = self.vitactip.vertices_deformed_A.to_numpy()[0]
            with open(
                SYSTEM_PARAMS.files.deformed_node_coordinates.format(ts), "wb"
            ) as f:
                pickle.dump(particles, f)
            self.mesh_needs_to_be_saved[None] = 0
    
    def save_sensor_mesh_to_npz(self):
        particles = self.vitactip.vertices_deformed_A.to_numpy()[0]
        path = SYSTEM_PARAMS.files.sensor_mesh
        np.savez(
            path,
            particles=particles,
        )

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
        self.sensor_points = ti.Vector.field(
            3, dtype=float, shape=(self.vitactip.num_vertices), needs_grad=False
        )
        self.healthy_tissue_points = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.phantom.num_particles,),
            needs_grad=False,
        )
        self.vein_2d_projection = ti.Vector.field(
            2,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.meta.max_num_veins,
                self.vein.centerline_A.shape[0],
            ),
            needs_grad=False,
        )
        self.vein_2d_projection_flat = ti.Vector.field(
            2,
            dtype=float,
            shape=(
                self.vein.centerline_A.shape[0],
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
        self.vein_2d_projection.fill(-1)
        self.vein_2d_projection_flat.fill(-1)

    @ti.kernel
    def visualisation_draw_3d_scene(self, f: ti.i32):
        for p in range(self.phantom.num_particles):
            # if self.phantom.grid_node_vein_indices[p] == 0:
            self.healthy_tissue_points[p] = self.phantom.particles_A[f, p]
        for p in range(self.vitactip.num_vertices):
            self.sensor_points[p] = self.vitactip.vertices_deformed_A[f, p]

    @ti.kernel
    def visualisation_project_vein_2d(self):
        for i in range(self.vein.centerline_A.shape[0]):
            point = self.vein.centerline_A[i]
            projection_2d = self.vitactip.project_A_point_2d(point)
            projection_2d[1] = self.tactile_image_resolution[None][1] - projection_2d[1]
            self.vein_2d_projection[0, i] = projection_2d
            self.vein_2d_projection_flat[i] = projection_2d / self.tactile_image_resolution[None]

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
            self.sim_markers_deformed, radius=0.01, color=(1, 0, 0)
        )
        self.tactile_canvas.circles(
            self.clock_arm_points,
            radius=0.02,
            per_vertex_color=self.clock_arm_points_per_vertex_color,
        )
        self.tactile_canvas.circles(
            self.vein_2d_projection_flat,
            radius=0.01,
            color=(0, 0, 1)
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
        self.camera.position(x-SYSTEM_PARAMS.visualisation.camera_offset, y, z)
        self.camera.up(0, 0, 1)
        self.camera.lookat(x, y, z)
        self.camera.fov(3)
        self.tactile_window = ti.ui.Window("tactile readout", (
            int(SYSTEM_PARAMS.visualisation.tactile_readout_width),
            int(SYSTEM_PARAMS.visualisation.tactile_readout_height)
        ))
        self.tactile_canvas = self.tactile_window.get_canvas()
        self.bg_image = cv2.imread(self.default_photo)
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

    def create_transition_array_vectorized(self, n):
        t = np.linspace(0, 1, n)[:, np.newaxis]
        start = np.array([0, 1, 1])
        end = np.array([1, 0, 0])
        return (1 - t) * start + t * end
    
    def visualisation_update_gui(self, ts):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        self.visualisation_draw_3d_scene(SYSTEM_PARAMS.contact.num_sub_frames - 1)
        self.visualisation_draw_tactile_readout()
        self.scene.particles(
            self.healthy_tissue_points,
            color=(0.0, 0.0, 1.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        self.phantom.compute_grid_colours()
        # self.scene.particles(
        #     self.phantom.grid_positions,
        #     per_vertex_color=self.phantom.grid_colours,
        #     radius=SYSTEM_PARAMS.visualisation.particle_size_normal*5,
        # )
        self.scene.particles(
            self.vein.particles_A,
            color=(1.0, 1.0, 0.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        self.scene.particles(
            self.sensor_points,
            color=(0.0, 1.0, 0.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        self.update_vitactip_tip_point()
        self.update_clock_arm_points_3d()
        self.scene.particles(
            self.vitactip_tip_point,
            color=(1.0, 0.0, 0.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_keypoint,
        )
        self.scene.particles(
            self.clock_arm_points_3d,
            per_vertex_color=self.clock_arm_points_per_vertex_color,
            radius=SYSTEM_PARAMS.visualisation.particle_size_keypoint,
        )
        self.canvas.scene(self.scene)
        self.window.show()

    def forward_pass_common_part(self, ts):
        self.reset_state()
        # self.set_optimisation_params_from_log()
        self.vitactip.set_control_vel(0)
        self.vitactip.set_vel(0)
        # self.vitactip.set_up_system_params_2()
        # self.phantom.set_stiffness()
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 1):
            self.update(ss)

    def backward_pass_common_part(self):
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 2, -1, -1):
            self.update_grad(ss)
        # self.phantom.set_stiffness.grad()
        # self.vitactip.set_up_system_params_2.grad()
        # self.set_optimisation_params_from_log.grad()
    
    def randomise_contact_params(self):
        ns = SYSTEM_PARAMS.contact.normal_stiffness
        nd = SYSTEM_PARAMS.contact.normal_damping
        ts = SYSTEM_PARAMS.contact.tangential_stiffness
        cfc = SYSTEM_PARAMS.contact.coulomb_friction_coeff

        self.normal_stiffness[None] = NP_RNG.uniform(ns * 0.5, ns * 1.5)
        self.normal_damping[None] = NP_RNG.uniform(nd * 0.5, nd * 1.5)
        self.tangential_stiffness[None] = NP_RNG.uniform(ts * 0.5, ts * 1.5)
        self.coulomb_friction_coeff[None] = NP_RNG.uniform(cfc * 0.5, cfc * 1.25)
    
    def set_contact_params_from_bo(self):
        self.normal_stiffness[0] = self.bo.params['normal_stiffness']
        self.normal_damping[0] = self.bo.params['normal_damping']
        self.tangential_stiffness[0] = self.bo.params['tangential_stiffness']
        self.coulomb_friction_coeff[0] = self.bo.params['coulomb_friction_coeff']

        self.vitactip.youngs_modulus[None] = self.bo.params['vitactip_youngs_modulus']
        self.vitactip.poissons_ratio[None] = self.bo.params['vitactip_poissons_ratio']
        self.vitactip.set_up_system_params_2()

        # self.phantom.youngs_modulus[0] = self.bo.params['phantom_youngs_modulus']
        # self.phantom.poissons_ratio[0] = self.bo.params['phantom_poissons_ratio']
        # self.phantom.set_stiffness()
    
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
        # dt = calculate_cfl_timestep(
        #     phantom_healthy_youngs_modulus=self.phantom.youngs_modulus[0],
        #     vitactip_youngs_modulus=self.vitactip.youngs_modulus[None],
        #     courant_number=self.courant_number,
        #     verbose=verbose,
        # )
        dt = SYSTEM_PARAMS.contact.dt_override
        self.dt[None] = dt
        self.phantom.dt[None] = dt
        self.vitactip.dt[None] = dt
        # if verbose:
        #     print(f'dt={dt:0.3e} s')
    
    def get_keypoint_indices_and_validate(self):
        self.vitactip.test_mapping_from_global_space_to_camera_space()
        self.vitactip.extract_markers(0)
        self.compute_mapping_between_experimental_and_sim_markers()
        self.vitactip.save_predicted_markers_to_image()
        self.vitactip.compute_clock_arm_ixs()
        self.vitactip.compute_tip_ix()
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
                        # self.set_optimisation_params_from_log()
                        # self.print_params_short()
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
        # self.save_final_params()
        print("all done")

    def collect_training_data(self):
        # self.clear_temp_images()
        self.clear_npz()
        self.generate_trajectories()
        for j in range(SYSTEM_PARAMS.contact.num_training_trajectories):
            if self.use_bo:
                if j < 4:
                    self.bo.my_suggest_random()
                else:
                    self.bo.my_suggest_optimise()
                self.set_contact_params_from_bo()
            print(f"training trajectory: {j} / {SYSTEM_PARAMS.contact.num_training_trajectories - 1}")
            for i in range(0, 4):
                # self.randomise_contact_params()
                self.trajectory_ix[None] = i
                trajectory_name = self.trajectory_names[self.trajectory_ix[None]]
                # print(f'executing trajectory: {trajectory_name}')
                self.set_up_initial_positions_state_and_trajectory()
                # self.vein_sparse_to_dense()
                self.reset_pid_controller()
                self.visualisation_reset_scene()
                self.reset_exp_sim_traj()
                self.vitactip.extract_markers(0)
                # self.compute_mapping_between_experimental_and_sim_markers()
                self.set_dt(verbose=True)
                self.fp()
                # self.print_contact_params()
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
                    if ts % 10 == 0:
                        self.record_vitactip_mesh()
                    # target = self.current_target_idx[None]
                    # if (
                    #     target > 2
                    #     and ts % 4 == 0
                    # ):
                    #     self.record_training_data_point(j, ts)
                    should_break = self.handle_da_loss(ts)
                    if should_break:
                        break
                    if ts % 100 == 0:
                        self.save_sensor_mesh_to_npz()
                        # print(f"ts={ts}; sensor mesh saved")
                    # if self.last_target_reached[None] == 1:
                    #     break
                # self.write_training_data_to_file(file_num, i)
                self.write_vitactip_mesh_to_file()
                
                self.reset_loss()
                self.batch_loss.fill(0.0)
                # self.clear_grad()
                self.prev_loss[None] = 0.0
                self.trajectory_loss[None] = 0.0   
            print(
                f"training trajectory: {j} / {SYSTEM_PARAMS.contact.num_training_trajectories - 1} done"
            )
            if self.use_bo:
                print(f'domain adaptation losses: {self.da_losses}')
                print(f'domain adaptation loss sum: {sum(self.da_losses)}')
                # self.write_da_total_loss_to_file()
                self.bo.my_register(
                    sum(self.da_losses)
                )
                self.da_losses = []
                self.bo.write_to_file()
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
    # contact_model.set_up_torch_params()
    
    import time
    start_time = time.perf_counter()
    contact_model.collect_training_data()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training data collection took {elapsed_time:.2f} seconds")
    
    contact_model.bo.write_to_file()
    if False:
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            contact_model.collect_training_data()
            profiler.disable()
        finally:
            profiler.dump_stats("profile.out")

