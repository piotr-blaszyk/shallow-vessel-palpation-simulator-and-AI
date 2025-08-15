import taichi as ti
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from difftactile.object_model.obj_loader import ObjLoader
from difftactile.main.constants import *

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32


@ti.data_oriented
class Phantom:
    def __init__(self):
        self.set_up_system_params()
        self.load_obj()
        self.set_up_physical_state()
        self.set_up_domain_randomisation()
        self.cache = dict()

    def set_up_system_params(self):
        self.dt = ti.field(dtype=float, shape=(), needs_grad=False)
        self.dt[None] = SYSTEM_PARAMS.contact.dt
        self.rayleigh_damping_alpha = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.rayleigh_damping_beta = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.rayleigh_damping_alpha[None] = SYSTEM_PARAMS.phantom.rayleigh_damping_alpha
        self.rayleigh_damping_beta[None] = SYSTEM_PARAMS.phantom.rayleigh_damping_beta
        self.inverse_mpm_grid_cube_size = 1 / SYSTEM_PARAMS.phantom.mpm_grid_cube_size
        self.youngs_modulus = ti.field(dtype=ti.f32, shape=(2,), needs_grad=True)
        self.poissons_ratio = ti.field(dtype=ti.f32, shape=(2,), needs_grad=True)
        self.lam = ti.field(dtype=ti.f32, shape=(2,), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=(2,), needs_grad=True)
        self.youngs_modulus[0] += SYSTEM_PARAMS.phantom.silicone.youngs_modulus
        self.poissons_ratio[0] += SYSTEM_PARAMS.phantom.silicone.poissons_ratio
        self.youngs_modulus[1] += SYSTEM_PARAMS.phantom.hard_plastic.youngs_modulus
        self.poissons_ratio[1] += SYSTEM_PARAMS.phantom.hard_plastic.poissons_ratio
        self.set_stiffness()
        self.max_num_veins = SYSTEM_PARAMS.meta.max_num_veins

    def load_obj(self):
        obj_loader = ObjLoader(
            SYSTEM_PARAMS.files.phantom,
            num_particles_cube_1d=SYSTEM_PARAMS.phantom.num_particles_cube_1d,
        )
        obj_loader.generate_particles()
        self.actual_total_num_particles = len(obj_loader.particles)
        self.particles_B = ti.Vector.field(
            3, dtype=float, shape=(self.actual_total_num_particles,), needs_grad=False
        )
        self.particles_B.from_numpy((obj_loader.particles).astype(np.float32))
        self.titles = ti.field(
            dtype=int, shape=self.actual_total_num_particles, needs_grad=False
        )
        self.vein_titles = ti.field(
            dtype=int, shape=self.actual_total_num_particles, needs_grad=False
        )
        self.is_fixed = ti.field(
            dtype=int, shape=(self.actual_total_num_particles,), needs_grad=False
        )
        particles_np = self.particles_B.to_numpy()
        z_coords = particles_np[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_threshold = z_min + SYSTEM_PARAMS.phantom.fixed_points_z_ratio * (
            z_max - z_min
        )
        is_fixed_np = z_coords <= z_threshold
        # is_fixed_np = np.ones_like(z_coords, dtype=bool)
        self.is_fixed.from_numpy(is_fixed_np.astype(int))
        self.initial_particle_volume = (
            SYSTEM_PARAMS_COMPUTED.phantom_volume / self.actual_total_num_particles
        )
        self.healthy_tissue_particle_mass = (
            self.initial_particle_volume * SYSTEM_PARAMS.phantom.silicone.density
        )
        self.total_healthy_tissue_mass = (
            SYSTEM_PARAMS_COMPUTED.phantom_volume
            * SYSTEM_PARAMS.phantom.silicone.density
        )

    @ti.kernel
    def set_stiffness(self):
        for item in range(2):
            self.mu[item] += (
                self.youngs_modulus[item] / 2 / (1 + self.poissons_ratio[item])
            )
            self.lam[item] += (
                self.youngs_modulus[item]
                * self.poissons_ratio[item]
                / (1 + self.poissons_ratio[item])
                / (1 - 2 * self.poissons_ratio[item])
            )

    def set_up_physical_state(self):
        self.initial_position = ti.Vector.field(
            3, dtype=ti.f32, shape=(), needs_grad=False
        )
        self.initial_orientation = ti.Vector.field(
            3, dtype=ti.f32, shape=(), needs_grad=False
        )
        self.initial_velocity = ti.Vector.field(
            3, dtype=ti.f32, shape=(), needs_grad=False
        )
        self.rotation_matrix = ti.Matrix.field(3, 3, ti.f32, shape=(), needs_grad=False)
        self.T_BA = ti.Matrix.field(
            4, 4, ti.f32, shape=(), needs_grad=False
        )
        self.particles_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.velocities_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.affine_velocity_field = ti.Matrix.field(
            3,
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.trial_deformation_gradient = ti.Matrix.field(
            3,
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.deformation_gradient = ti.Matrix.field(
            3,
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.U_svd = ti.Matrix.field(
            3,
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.V_svd = ti.Matrix.field(
            3,
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.S_svd = ti.Matrix.field(
            3,
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.actual_total_num_particles,
            ),
            needs_grad=False,
        )
        self.grid_node_momentum_in = ti.Vector.field(
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS.phantom.n_grid_x,
                SYSTEM_PARAMS.phantom.n_grid_y,
                SYSTEM_PARAMS.phantom.n_grid_z,
            ),
            needs_grad=True,
        )
        self.grid_node_velocity_out = ti.Vector.field(
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS.phantom.n_grid_x,
                SYSTEM_PARAMS.phantom.n_grid_y,
                SYSTEM_PARAMS.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.grid_node_mass = ti.field(
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS.phantom.n_grid_x,
                SYSTEM_PARAMS.phantom.n_grid_y,
                SYSTEM_PARAMS.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.grid_node_external_impulse = ti.Vector.field(
            3,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS.phantom.n_grid_x,
                SYSTEM_PARAMS.phantom.n_grid_y,
                SYSTEM_PARAMS.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.grid_occupy = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS.phantom.n_grid_x,
                SYSTEM_PARAMS.phantom.n_grid_y,
                SYSTEM_PARAMS.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.total_surface_external_force = ti.Vector.field(
            3, float, shape=(SYSTEM_PARAMS.contact.num_sub_frames), needs_grad=False
        )
        self.keypoint_idx = ti.field(dtype=int, shape=(), needs_grad=False)
        self.keypoint_idx[None] = -1

    def set_up_domain_randomisation(self):
        self.group_cardinality = ti.field(dtype=int, shape=(2,), needs_grad=False)
        self.cylinder_cx = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_cy = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_cz = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_theta = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_h = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_r = ti.field(dtype=float, shape=(), needs_grad=False)

    @ti.kernel
    def fix_vein(self):
        return
        for i in range(self.actual_total_num_particles):
            if self.titles[i] == 1:
                self.is_fixed[i] = 1

    def set_state_from_outside(
        self, pos, ori, vel, state_dicts
    ):
        self.vein_titles.fill(-1)
        if len(state_dicts) > 0:
            self.titles.fill(0)
            self.group_cardinality.fill(0)
            for i in range(len(state_dicts)):
                state_dict = state_dicts[i]
                self.set_up_tumour_inclusion(state_dict)
                self.partition_point_cloud(i)
            self.compute_group_cardinality()
        else:
            self.group_cardinality[0] = self.actual_total_num_particles
            self.group_cardinality[1] = 0
            self.titles.fill(0)
        print(
            f"tumour_present: {len(state_dicts) > 0}, healthy: {self.group_cardinality[0]}, tumour: {self.group_cardinality[1]}"
        )
        self.set_pose_and_velocity(pos, ori, vel)
        self.initialise_point_cloud()

    def set_up_tumour_inclusion(self, state_dict):
        self.cylinder_cx[None] = state_dict['cx']
        self.cylinder_cy[None] = state_dict['cy']
        self.cylinder_cz[None] = state_dict['cz']
        self.cylinder_theta[None] = np.deg2rad(state_dict['theta'])
        self.cylinder_h[None] = state_dict['h']
        self.cylinder_r[None] = state_dict['r']

    @ti.kernel
    def compute_group_cardinality(self):
        for i in range(self.actual_total_num_particles):
            if self.titles[i] == 0:
                self.group_cardinality[0] += 1
            elif self.titles[i] == 1:
                self.group_cardinality[1] += 1

    @ti.kernel
    def partition_point_cloud(self, i: ti.i32):
        for item in range(self.actual_total_num_particles):
            pos = self.particles_B[item]
            px = pos[0] - self.cylinder_cx[None]
            py = pos[1] - self.cylinder_cy[None]
            pz = pos[2] - self.cylinder_cz[None]
            cost = ti.cos(-self.cylinder_theta[None])
            sint = ti.sin(-self.cylinder_theta[None])
            x_local = cost * px - sint * py
            y_local = sint * px + cost * py
            z_local = pz
            half_h = self.cylinder_h[None] / 2.0
            if (ti.abs(x_local) <= half_h) and (
                y_local * y_local + z_local * z_local
                <= self.cylinder_r[None] * self.cylinder_r[None]
            ):
                self.titles[item] = 1
                self.vein_titles[item] = i

    def set_pose_and_velocity(self, position, orientation, velocity):
        self.initial_position[None] = position
        self.initial_orientation[None] = orientation
        self.initial_velocity[None] = velocity
        rotation_object = R.from_rotvec(
            np.deg2rad([orientation[0], orientation[1], orientation[2]])
        )
        rotation_matrix = rotation_object.as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0, 3] = position[0]
        transformation_matrix[1, 3] = position[1]
        transformation_matrix[2, 3] = position[2]
        self.rotation_matrix[None] = rotation_matrix.tolist()
        self.T_BA[None] = transformation_matrix.tolist()

    @ti.kernel
    def initialise_point_cloud(self):
        for i in range(self.actual_total_num_particles):
            particle_position_local = self.particles_B[i]
            particle_position_global = self.T_BA[None] @ ti.Vector(
                [
                    particle_position_local[0],
                    particle_position_local[1],
                    particle_position_local[2],
                    1.0,
                ]
            )
            self.particles_A[0, i] = ti.Vector(
                [
                    particle_position_global[0],
                    particle_position_global[1],
                    particle_position_global[2],
                ]
            )
            self.velocities_A[0, i] = ti.Matrix(
                [
                    self.initial_velocity[None][0],
                    self.initial_velocity[None][1],
                    self.initial_velocity[None][2],
                ]
            )
            self.deformation_gradient[0, i] = ti.Matrix(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )

    @ti.kernel
    def get_external_force(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x,
            SYSTEM_PARAMS.phantom.n_grid_y,
            SYSTEM_PARAMS.phantom.n_grid_z,
        ):
            self.total_surface_external_force[f] += (
                self.grid_node_external_impulse[f, i, j, k] / self.dt[None]
            )

    @ti.func
    def update_contact_impulse(self, external_force, f, i, j, k):
        self.grid_node_external_impulse[f, i, j, k] += (
            external_force * self.dt[None]
        )

    @ti.kernel
    def compute_trial_deformation_gradient(self, f: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.trial_deformation_gradient[f, p] = (
                ti.Matrix.diag(dim=3, val=1)
                + self.dt[None] * self.affine_velocity_field[f, p]
            ) @ self.deformation_gradient[f, p]

    @ti.kernel
    def svd_of_trial_deformation_gradient(self, f: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.U_svd[f, p], self.S_svd[f, p], self.V_svd[f, p] = ti.svd(
                self.trial_deformation_gradient[f, p]
            )

    @ti.kernel
    def svd_of_trial_deformation_gradient_grad(self, f: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.trial_deformation_gradient.grad[f, p] += self.single_svd_grad(f, p)

    @ti.func
    def clamp(self, a: ti.f32):
        if a >= 0:
            a = ti.max(a, 1e-8)
        else:
            a = ti.min(a, -1e-8)
        return a

    @ti.func
    def single_svd_grad(self, f: ti.i32, p: ti.i32):
        vt = self.V_svd[f, p].transpose()
        ut = self.U_svd[f, p].transpose()
        s_term = self.U_svd[f, p] @ self.S_svd.grad[f, p] @ vt
        s = ti.Vector.zero(ti.f32, 3)
        s = (
            ti.Vector(
                [self.S_svd[f, p][0, 0], self.S_svd[f, p][1, 1], self.S_svd[f, p][2, 2]]
            )
            ** 2
        )
        ff = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            if i == j:
                ff[i, j] = 0
            else:
                ff[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = (
            self.U_svd[f, p]
            @ (
                (
                    ff
                    * (
                        ut @ self.U_svd.grad[f, p]
                        - self.U_svd.grad[f, p].transpose() @ self.U_svd[f, p]
                    )
                )
                @ self.S_svd[f, p]
            )
            @ vt
        )
        v_term = self.U_svd[f, p] @ (
            self.S_svd[f, p]
            @ (
                (
                    ff
                    * (
                        vt @ self.V_svd.grad[f, p]
                        - self.V_svd.grad[f, p].transpose() @ self.V_svd[f, p]
                    )
                )
                @ vt
            )
        )
        return u_term + v_term + s_term

    @ti.kernel
    def p2g(self, frame: ti.i32):
        for particle_id in range(self.actual_total_num_particles):
            shear_modulus = self.mu[self.titles[particle_id]]
            bulk_modulus = self.lam[self.titles[particle_id]]
            grid_base_index = (
                self.particles_A[frame, particle_id]
                * self.inverse_mpm_grid_cube_size
                - 0.5
            ).cast(int)
            particle_grid_diff = self.particles_A[
                frame, particle_id
            ] * self.inverse_mpm_grid_cube_size - grid_base_index.cast(float)
            weight_functions = [
                0.5 * (1.5 - particle_grid_diff) ** 2,
                0.75 - (particle_grid_diff - 1) ** 2,
                0.5 * (particle_grid_diff - 0.5) ** 2,
            ]
            volume_ratio = (self.S_svd[frame, particle_id]).determinant()
            rotation_matrix = (
                self.U_svd[frame, particle_id]
                @ self.V_svd[frame, particle_id].transpose()
            )
            cauchy_stress = 2 * shear_modulus * (
                self.trial_deformation_gradient[frame, particle_id] - rotation_matrix
            ) @ self.trial_deformation_gradient[
                frame, particle_id
            ].transpose() + ti.Matrix.identity(
                float, 3
            ) * bulk_modulus * volume_ratio * (volume_ratio - 1)
            force_term = cauchy_stress * self.initial_particle_volume
            impulse_term = force_term * -self.dt[None]
            impulse_term_scaled_quadratic_B_spline = (
                4 * self.inverse_mpm_grid_cube_size**2 * impulse_term
            )
            momentum_contribution = (
                impulse_term_scaled_quadratic_B_spline
                + self.healthy_tissue_particle_mass
                * self.affine_velocity_field[frame, particle_id]
            )
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                grid_offset = ti.Vector([i, j, k])
                dist_to_grid = (
                    grid_offset.cast(float) - particle_grid_diff
                ) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size
                weight = (
                    weight_functions[i][0]
                    * weight_functions[j][1]
                    * weight_functions[k][2]
                )
                self.grid_node_momentum_in[frame, grid_base_index + grid_offset] += (
                    weight
                    * (
                        self.healthy_tissue_particle_mass
                        * self.velocities_A[frame, particle_id]
                        + momentum_contribution @ dist_to_grid
                    )
                )
                self.grid_node_mass[frame, grid_base_index + grid_offset] += (
                    weight * self.healthy_tissue_particle_mass
                )
            self.deformation_gradient[frame + 1, particle_id] = (
                self.trial_deformation_gradient[frame, particle_id]
            )

    @ti.kernel
    def check_grid_occupy(self, f: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x,
            SYSTEM_PARAMS.phantom.n_grid_y,
            SYSTEM_PARAMS.phantom.n_grid_z,
        ):
            if self.grid_node_mass[f, i, j, k] > SYSTEM_PARAMS.phantom.mass_eps:
                self.grid_occupy[f, i, j, k] = 1

    @ti.kernel
    def grid_op(self, frame: ti.i32):
        for grid_x, grid_y, grid_z in ti.ndrange(
            SYSTEM_PARAMS.phantom.n_grid_x,
            SYSTEM_PARAMS.phantom.n_grid_y,
            SYSTEM_PARAMS.phantom.n_grid_z,
        ):
            if self.grid_occupy[frame, grid_x, grid_y, grid_z] == 1:
                inverse_mass = 1 / (
                    self.grid_node_mass[frame, grid_x, grid_y, grid_z]
                    + SYSTEM_PARAMS.phantom.mass_eps
                )
                grid_mass = self.grid_node_mass[frame, grid_x, grid_y, grid_z]
                grid_velocity = ti.Vector([0.0, 0.0, 0.0])
                grid_velocity += (
                    inverse_mass
                    * self.grid_node_momentum_in[frame, grid_x, grid_y, grid_z]
                )
                grid_velocity += (
                    inverse_mass
                    * self.grid_node_external_impulse[frame, grid_x, grid_y, grid_z]
                )
                grid_velocity += self.dt[None] * ti.Vector(
                    SYSTEM_PARAMS.phantom.gravity
                )
                elastic_force = (
                    self.grid_node_momentum_in[frame, grid_x, grid_y, grid_z]
                    / self.dt[None]
                )
                mass_damping = (
                    -self.rayleigh_damping_alpha[None] * grid_mass * grid_velocity
                )
                stiffness_damping = -self.rayleigh_damping_beta[None] * elastic_force
                grid_velocity += (
                    inverse_mass
                    * (mass_damping + stiffness_damping)
                    * self.dt[None]
                )
                if grid_x < SYSTEM_PARAMS.phantom.bound and grid_velocity[0] < 0:
                    grid_velocity[0] = 0
                if (
                    grid_x
                    > SYSTEM_PARAMS.phantom.n_grid_x - SYSTEM_PARAMS.phantom.bound
                    and grid_velocity[0] > 0
                ):
                    grid_velocity[0] = 0
                if grid_y < SYSTEM_PARAMS.phantom.bound and grid_velocity[1] < 0:
                    grid_velocity[1] = 0
                if (
                    grid_y
                    > SYSTEM_PARAMS.phantom.n_grid_y - SYSTEM_PARAMS.phantom.bound
                    and grid_velocity[1] > 0
                ):
                    grid_velocity[1] = 0
                if grid_z < SYSTEM_PARAMS.phantom.bound and grid_velocity[2] < 0:
                    grid_velocity[2] = 0
                if (
                    grid_z
                    > SYSTEM_PARAMS.phantom.n_grid_y - SYSTEM_PARAMS.phantom.bound
                    and grid_velocity[2] > 0
                ):
                    grid_velocity[2] = 0
                self.grid_node_velocity_out[frame, grid_x, grid_y, grid_z] = (
                    grid_velocity
                )

    @ti.kernel
    def g2p(self, frame: ti.i32):
        for particle_id in range(self.actual_total_num_particles):
            grid_base_index = (
                self.particles_A[frame, particle_id]
                * self.inverse_mpm_grid_cube_size
                - 0.5
            ).cast(int)
            particle_grid_diff = self.particles_A[
                frame, particle_id
            ] * self.inverse_mpm_grid_cube_size - grid_base_index.cast(float)
            weight_functions = [
                0.5 * (1.5 - particle_grid_diff) ** 2,
                0.75 - (particle_grid_diff - 1.0) ** 2,
                0.5 * (particle_grid_diff - 0.5) ** 2,
            ]
            updated_velocity = ti.Vector.zero(float, 3)
            updated_affine = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                grid_relative_offset = (
                    ti.Vector([i, j, k]).cast(float) - particle_grid_diff
                )
                grid_node_velocity = self.grid_node_velocity_out[
                    frame, grid_base_index + ti.Vector([i, j, k])
                ]
                weight = (
                    weight_functions[i][0]
                    * weight_functions[j][1]
                    * weight_functions[k][2]
                )
                updated_velocity += weight * grid_node_velocity
                updated_affine += (
                    4
                    * self.inverse_mpm_grid_cube_size
                    * weight
                    * grid_node_velocity.outer_product(grid_relative_offset)
                )
            if SYSTEM_PARAMS.phantom.fix_bottom_points == 1 and self.is_fixed[particle_id] == 1:
                self.velocities_A[frame + 1, particle_id] = ti.Vector(
                    [0.0, 0.0, 0.0]
                )
                self.affine_velocity_field[frame + 1, particle_id] = ti.Matrix.zero(
                    float, 3, 3
                )
                self.particles_A[frame + 1, particle_id] = self.particles_A[
                    frame, particle_id
                ]
            else:
                self.velocities_A[frame + 1, particle_id] = updated_velocity
                self.affine_velocity_field[frame + 1, particle_id] = updated_affine
                self.particles_A[frame + 1, particle_id] = (
                    self.particles_A[frame, particle_id]
                    + self.dt[None] * updated_velocity
                )

    @ti.func
    def reset_state(self):
        self.youngs_modulus.fill(0.0)
        self.poissons_ratio.fill(0.0)
        self.mu.fill(0.0)
        self.lam.fill(0.0)
        self.grid_node_momentum_in.fill(0.0)
        self.grid_node_velocity_out.fill(0.0)
        self.grid_node_mass.fill(0.0)
        self.grid_node_external_impulse.fill(0.0)
        self.grid_occupy.fill(0.0)
        self.total_surface_external_force.fill(0.0)

    @ti.kernel
    def clear_grad(self):
        self.grid_node_momentum_in.grad.fill(0.0)
        self.mu.grad.fill(0.0)
        self.lam.grad.fill(0.0)
        self.youngs_modulus.grad.fill(0.0)
        self.poissons_ratio.grad.fill(0.0)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.particles_A[target, p] = self.particles_A[source, p]
            self.velocities_A[target, p] = self.velocities_A[source, p]
            self.affine_velocity_field[target, p] = self.affine_velocity_field[
                source, p
            ]
            self.deformation_gradient[target, p] = self.deformation_gradient[source, p]

    @ti.kernel
    def load_step_from_cache(
        self,
        f: ti.i32,
        cache_x_0: ti.types.ndarray(),
        cache_v_0: ti.types.ndarray(),
        cache_C_0: ti.types.ndarray(),
        cache_F_0: ti.types.ndarray(),
    ):
        for p in range(self.actual_total_num_particles):
            for i in ti.static(range(3)):
                self.particles_A[f, p][i] = cache_x_0[p, i]
                self.velocities_A[f, p][i] = cache_v_0[p, i]
            for i, j in ti.ndrange(3, 3):
                self.affine_velocity_field[f, p][i, j] = cache_C_0[p, i, j]
                self.deformation_gradient[f, p][i, j] = cache_F_0[p, i, j]

    @ti.kernel
    def add_step_to_cache(
        self,
        f: ti.i32,
        cache_x_0: ti.types.ndarray(),
        cache_v_0: ti.types.ndarray(),
        cache_C_0: ti.types.ndarray(),
        cache_F_0: ti.types.ndarray(),
    ):
        for p in range(self.actual_total_num_particles):
            for i in ti.static(range(3)):
                cache_x_0[p, i] = self.particles_A[f, p][i]
                cache_v_0[p, i] = self.velocities_A[f, p][i]
            for i, j in ti.ndrange(3, 3):
                cache_C_0[p, i, j] = self.affine_velocity_field[f, p][i, j]
                cache_F_0[p, i, j] = self.deformation_gradient[f, p][i, j]

    def memory_to_cache(self, t):
        cur_step_name = f"{t:06d}"
        device = "cpu"
        self.cache[cur_step_name] = dict()
        self.cache[cur_step_name]["x_0"] = torch.zeros(
            (self.actual_total_num_particles, 3), dtype=TC_TYPE, device=device
        )
        self.cache[cur_step_name]["v_0"] = torch.zeros(
            (self.actual_total_num_particles, 3), dtype=TC_TYPE, device=device
        )
        self.cache[cur_step_name]["C_0"] = torch.zeros(
            (self.actual_total_num_particles, 3, 3), dtype=TC_TYPE, device=device
        )
        self.cache[cur_step_name]["F_0"] = torch.zeros(
            (self.actual_total_num_particles, 3, 3), dtype=TC_TYPE, device=device
        )
        self.add_step_to_cache(
            0,
            self.cache[cur_step_name]["x_0"],
            self.cache[cur_step_name]["v_0"],
            self.cache[cur_step_name]["C_0"],
            self.cache[cur_step_name]["F_0"],
        )
        self.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames - 1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f"{t:06d}"
        self.load_step_from_cache(
            0,
            self.cache[cur_step_name]["x_0"],
            self.cache[cur_step_name]["v_0"],
            self.cache[cur_step_name]["C_0"],
            self.cache[cur_step_name]["F_0"],
        )

    def get_keypoint_index(self) -> int:
        positions = self.particles_A.to_numpy()[0]
        centroid_x = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose[0]
        centroid_y = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose[1]
        x_mask = (
            np.abs(positions[:, 0] - centroid_x)
            < SYSTEM_PARAMS.phantom.keypoint_search_xy_threshold
        )
        y_mask = (
            np.abs(positions[:, 1] - centroid_y)
            < SYSTEM_PARAMS.phantom.keypoint_search_xy_threshold
        )
        xy_mask = x_mask & y_mask
        valid_points = positions[xy_mask]
        if len(valid_points) == 0:
            raise ValueError("No points found within 0.001 of centroid x,y coordinates")
        min_z_idx = np.argmin(valid_points[:, 2])
        keypoint_idx = np.where(xy_mask)[0][min_z_idx]
        self.keypoint_idx[None] = int(keypoint_idx)
        return np.array([keypoint_idx])

    def get_keypoint_coordinates(
        self, f: int, keypoint_indices: np.ndarray
    ) -> np.ndarray:
        positions = self.particles_A.to_numpy()[f]
        coordinates = positions[keypoint_indices]
        return coordinates
    