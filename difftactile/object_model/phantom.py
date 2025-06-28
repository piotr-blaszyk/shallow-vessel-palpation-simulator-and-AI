"""
a class to describe multi-material objects with mpm
"""
import os
import taichi as ti
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import json

from difftactile.object_model.obj_loader import ObjLoader

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
        self.cache = dict() # for grad backward

    def set_up_system_params(self):
        with open('../tasks/system-params.json', 'r') as f:
            self.params = json.load(f)
            self.phantom_params = self.params['phantom']
            self.contact_params = self.params['contact']
        
        with open('../tasks/initial-coordinates-and-geometry.json', 'r') as f:
            self.coordinates = json.load(f)
        
        self.target_total_num_particles = self.coordinates['target_total_num_particles']

        self.total_volume = self.coordinates['phantom_volume']

        self.sub_steps = self.contact_params['num_sub_frames']
        self.dt = self.contact_params['dt']
        self.bound = 3
        self.n_grid = 64
        self.space_scale = self.phantom_params['space_scale']
        self.obj_scale = self.phantom_params['object_scale']
        self.healthy_tissue_mass_density = self.phantom_params['silicone']['density']
        self.gravity = ti.Vector(self.phantom_params['gravity'])

        self.mpm_grid_cube_size = float(self.space_scale / self.n_grid)
        self.inverse_mpm_grid_cube_size =  1 / self.mpm_grid_cube_size
        self.eps = 1e-8
        
        self.youngs_modulus_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)
        self.poissons_ratio_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)
        self.lamda_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)
        self.mu_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)

        self.set_stiffness()

    def load_obj(self):
        self.obj_name = self.params['contact']['phantom_name']
        if self.obj_name is None:
            raise Exception("Please specify the name of the phantom object to load")
        
        data_path = os.path.join("..", "meshes", "objects", self.obj_name)
        obj_loader = ObjLoader(data_path, target_total_num_cube_particles = int(self.target_total_num_particles))
        obj_loader.generate_particles()
        self.actual_total_num_particles = len(obj_loader.particles)
        self.particles = ti.Vector.field(3, dtype=float, shape=(self.actual_total_num_particles,))
        self.particles.from_numpy((obj_loader.particles * self.obj_scale).astype(np.float32))
        self.titles = ti.field(dtype=int, shape=self.actual_total_num_particles)
        
        self.is_fixed = ti.field(dtype=int, shape=(self.actual_total_num_particles,))
        particles_np = self.particles.to_numpy()
        z_coords = particles_np[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_threshold = z_min + 0.25 * (z_max - z_min)
        is_fixed_np = (z_coords <= z_threshold)
        self.is_fixed.from_numpy(is_fixed_np.astype(int))

        self.initial_particle_volume = self.total_volume / self.actual_total_num_particles
        self.healthy_tissue_particle_mass = self.initial_particle_volume * self.healthy_tissue_mass_density

    def set_stiffness(self):
        healthy_tissue = self.phantom_params['silicone']
        tumour = self.phantom_params['hard_plastic']
        self.youngs_modulus_0[0] = healthy_tissue['youngs_modulus'] * self.space_scale
        self.poissons_ratio_0[0] = healthy_tissue['poissons_ratio']
        self.youngs_modulus_0[1] = tumour['youngs_modulus'] * self.space_scale
        self.poissons_ratio_0[1] = tumour['poissons_ratio']

        for item in range(2):
            self.mu_0[item] = self.youngs_modulus_0[item] / 2 / (1 + self.poissons_ratio_0[item])
            self.lamda_0[item] = self.youngs_modulus_0[item] * self.poissons_ratio_0[item] / (1 + self.poissons_ratio_0[item]) / (1 - 2 * self.poissons_ratio_0[item])

    def set_up_physical_state(self):
        # initial_position: position vector in m
        self.initial_position = ti.Vector.field(3, dtype=ti.f32, shape=())
        # initial_orientation: orientation vector in degrees
        self.initial_orientation = ti.Vector.field(3, dtype=ti.f32, shape=())
        # initial_velocity: velocity vector in m/s
        self.initial_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        # rotation_matrix: 3x3 rotation matrix (dimensionless)
        self.rotation_matrix = ti.Matrix.field(3, 3, ti.f32, shape=())
        # transformation_matrix: 4x4 transformation matrix (dimensionless)
        self.transformation_matrix = ti.Matrix.field(4, 4, ti.f32, shape=())

        # particle_position: position vectors in m
        self.particle_position = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)
        # particle_velocity: velocity vectors in m/s
        self.particle_velocity = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)

        # affine_velocity: affine velocity field matrix (m/s)
        self.affine_velocity_field = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)
        # trial_deformation_gradient: trial deformation gradient matrix (dimensionless)
        self.trial_deformation_gradient = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)
        # deformation_gradient: deformation gradient matrix (dimensionless)
        self.deformation_gradient = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)
        # U_svd: left singular vectors matrix (dimensionless)
        self.U_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)
        # V_svd: right singular vectors matrix (dimensionless)
        self.V_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)
        # S_svd: singular values matrix (dimensionless)
        self.S_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.actual_total_num_particles), needs_grad=False)

        # grid_node_momentum_in: momentum vectors in kg⋅m/s
        self.grid_node_momentum_in = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)
        # grid_node_velocity_out: velocity vectors in m/s
        self.grid_node_velocity_out = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)
        # grid_node_mass: mass values in kg
        self.grid_node_mass = ti.field(dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)
        # grid_node_external_force: force vectors in N
        self.grid_node_external_impulse = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)
        # grid_occupy: occupancy flags (dimensionless)
        self.grid_occupy = ti.field(dtype=int, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid))
        # total_surface_external_force: total surface force vector in N
        self.total_surface_external_force = ti.Vector.field(3, float, shape=(self.sub_steps), needs_grad=False)

        self.keypoint_idx = ti.field(dtype=int, shape=())
        self.keypoint_idx[None] = -1
    
    def set_up_domain_randomisation(self):
        self.group_cardinality = ti.field(dtype=int, shape=(2,), needs_grad=False)

        self.cylinder_cx = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_cy = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_cz = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_theta = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_h = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_r = ti.field(dtype=float, shape=(), needs_grad=False)

    def set_state_from_outside(self, pos, ori, vel, cylinder_tuple, stiffness_tuple, tumour_present):
        if tumour_present:
            self.titles.fill(-1)
            self.group_cardinality.fill(0)
            self.set_up_tumour_inclusion(cylinder_tuple)
            self.partition_point_cloud()
        else:
            self.group_cardinality[0] = self.actual_total_num_particles
            self.group_cardinality[1] = 0
            self.titles.fill(0)
        print(f'tumour_present: {tumour_present}, healthy: {self.group_cardinality[0]}, tumour: {self.group_cardinality[1]}')
        tumour_particles_absent = self.group_cardinality[1] == 0
        # if tumour_present and tumour_particles_absent:
        #     raise Exception("tumour present but no tumour particles generated!")
        tumour_present = not tumour_particles_absent
        self.set_pose_and_velocity(pos, ori, vel)
        self.initialise_point_cloud()

    def set_up_tumour_inclusion(self, cylinder_tuple):
        cx, cy, cz, theta, h, r = cylinder_tuple
        self.cylinder_cx[None] = cx
        self.cylinder_cy[None] = cy
        self.cylinder_cz[None] = cz
        self.cylinder_theta[None] = np.deg2rad(theta)
        self.cylinder_h[None] = h
        self.cylinder_r[None] = r

    @ti.kernel
    def partition_point_cloud(self):
        for item in range(self.actual_total_num_particles):
            pos = self.particles[item]
            # Translate to cylinder center
            px = pos[0] - self.cylinder_cx[None]
            py = pos[1] - self.cylinder_cy[None]
            pz = pos[2] - self.cylinder_cz[None]
            # Rotate by -theta about z-axis to align cylinder axis with x+
            cost = ti.cos(-self.cylinder_theta[None])
            sint = ti.sin(-self.cylinder_theta[None])
            x_local = cost * px - sint * py
            y_local = sint * px + cost * py
            z_local = pz
            # Check if within cylinder
            half_h = self.cylinder_h[None] / 2.0
            if (ti.abs(x_local) <= half_h) and (y_local * y_local + z_local * z_local <= self.cylinder_r[None] * self.cylinder_r[None]):
                self.titles[item] = 1
                self.group_cardinality[1] += 1
            else:
                self.titles[item] = 0
                self.group_cardinality[0] += 1

    def set_pose_and_velocity(self, position, orientation, velocity):
        self.initial_position[None] = position
        self.initial_orientation[None] = orientation
        self.initial_velocity[None] = velocity

        rotation_object = R.from_rotvec(np.deg2rad([orientation[0], orientation[1], orientation[2]]))
        rotation_matrix = rotation_object.as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3,0:3] = rotation_matrix
        transformation_matrix[0,3] = position[0]; transformation_matrix[1,3] = position[1]; transformation_matrix[2,3] = position[2]
        self.rotation_matrix[None] = rotation_matrix.tolist()
        self.transformation_matrix[None] = transformation_matrix.tolist()

    @ti.kernel
    def initialise_point_cloud(self):
        for i in range(self.actual_total_num_particles):
            # particle_position_reference: reference particle position in m
            current_particle_position = self.particles[i]
            # particle_position_transformed: transformed particle position in homogeneous coordinates
            target_particle_position = self.transformation_matrix[None] @ ti.Vector([current_particle_position[0], current_particle_position[1], current_particle_position[2], 1.0]) # 4 x 1 homogeneous
            self.particle_position[0,i] = ti.Vector([target_particle_position[0], target_particle_position[1], target_particle_position[2]])
            self.particle_velocity[0,i] = ti.Matrix([self.initial_velocity[None][0], self.initial_velocity[None][1], self.initial_velocity[None][2]])
            self.deformation_gradient[0,i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @ti.kernel
    def reset(self):
        self.grid_node_momentum_in.fill(0.0)
        self.grid_node_velocity_out.fill(0.0)
        self.grid_node_mass.fill(0.0)
        self.grid_node_external_impulse.fill(0.0)
        self.grid_occupy.fill(0.0)
        self.total_surface_external_force.fill(0.0)

    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            # external force in N
            self.total_surface_external_force[f] += self.grid_node_external_impulse[f, i, j, k] / self.dt

    @ti.func
    def update_contact_impulse(self, external_force, f, i, j, k):
        # extrnal impulse in N * s
        self.grid_node_external_impulse[f, i, j, k] += external_force * self.dt

    @ti.kernel
    def compute_trial_deformation_gradient(self, f: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.trial_deformation_gradient[f, p] = (ti.Matrix.diag(dim=3, val=1) + self.dt * self.affine_velocity_field[f, p]) @ self.deformation_gradient[f, p]

    @ti.kernel
    def svd_of_trial_deformation_gradient(self, f: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.U_svd[f, p], self.S_svd[f, p], self.V_svd[f, p] = ti.svd(self.trial_deformation_gradient[f, p])

    @ti.kernel
    def p2g(self, frame:ti.i32):
        """
        Particle to grid (P2G) step in MPM.
        Transfers particle properties to grid nodes using interpolation kernels.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for particle_id in range(self.actual_total_num_particles):
            # shear_modulus: shear modulus in Pa
            shear_modulus = self.mu_0[self.titles[particle_id]]
            # bulk_modulus: bulk modulus in Pa
            bulk_modulus = self.lamda_0[self.titles[particle_id]]
            # grid_base_index: grid cell indices (dimensionless)
            grid_base_index = (self.particle_position[frame, particle_id] * self.inverse_mpm_grid_cube_size - 0.5).cast(int)
            # particle_grid_diff: fractional position within grid cell (dimensionless)
            particle_grid_diff = self.particle_position[frame, particle_id] * self.inverse_mpm_grid_cube_size - grid_base_index.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=particle_grid_diff, particle_grid_diff-1,particle_grid_diff-2]
            # weight_functions: interpolation weights (dimensionless)
            weight_functions = [0.5 * (1.5 - particle_grid_diff) ** 2, 0.75 - (particle_grid_diff - 1) ** 2, 0.5 * (particle_grid_diff - 0.5) ** 2]

            # volume_ratio: volume ratio (dimensionless)
            volume_ratio = (self.S_svd[frame, particle_id]).determinant()
            # rotation_matrix: rotation matrix (dimensionless)
            rotation_matrix = self.U_svd[frame, particle_id] @ self.V_svd[frame, particle_id].transpose()
            # cauchy_stress: stress tensor in Pa
            cauchy_stress = 2 * shear_modulus * (self.trial_deformation_gradient[frame, particle_id] - rotation_matrix) @ self.trial_deformation_gradient[frame, particle_id].transpose() + ti.Matrix.identity(float, 3) * bulk_modulus * volume_ratio * (volume_ratio - 1)

            # force_term: force contribution in N
            force_term = cauchy_stress * self.initial_particle_volume
            # impulse_term: kg * m / s
            impulse_term = force_term * -self.dt
            # impulse_term_scaled: kg * m / s
            impulse_term_scaled_quadratic_B_spline = 4 * self.inverse_mpm_grid_cube_size ** 2 * impulse_term

            # momentum_contrib: momentum contribution in kg⋅m/s
            momentum_contribution = impulse_term_scaled_quadratic_B_spline + self.healthy_tissue_particle_mass * self.affine_velocity_field[frame, particle_id]

            # Loop over 3x3 grid node neighborhood
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                # grid_offset: grid offset vector (dimensionless)
                grid_offset = ti.Vector([i, j, k])
                # dist_to_grid: distance to grid node in m
                dist_to_grid = (grid_offset.cast(float) - particle_grid_diff) * self.mpm_grid_cube_size
                # weight: interpolation weight (dimensionless)
                weight = weight_functions[i][0] * weight_functions[j][1] * weight_functions[k][2]
                self.grid_node_momentum_in[frame, grid_base_index + grid_offset] += weight * (self.healthy_tissue_particle_mass * self.particle_velocity[frame, particle_id] + momentum_contribution @ dist_to_grid)
                self.grid_node_mass[frame, grid_base_index + grid_offset] += weight * self.healthy_tissue_particle_mass

            self.deformation_gradient[frame+1, particle_id] = self.trial_deformation_gradient[frame, particle_id]

    @ti.kernel
    def check_grid_occupy(self, f:ti.i32):
        """
        Check which grid nodes are occupied by particles.
        
        Args:
            f: current simulation frame (dimensionless)
        """
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.grid_node_mass[f, i, j, k] > self.eps:
                self.grid_occupy[f, i, j, k] = 1

    @ti.kernel
    def grid_op(self, frame:ti.i32):
        """
        Grid operations: update grid velocities and apply boundary conditions.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for grid_x, grid_y, grid_z in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.grid_occupy[frame, grid_x, grid_y, grid_z] == 1:
                # inverse_mass: inverse mass in kg^-1
                inverse_mass = 1 / (self.grid_node_mass[frame, grid_x, grid_y, grid_z] + self.eps)

                # grid_velocity: grid node velocity in m/s
                grid_velocity = ti.Vector([0.0, 0.0, 0.0])
                grid_velocity += inverse_mass * self.grid_node_momentum_in[frame, grid_x, grid_y, grid_z] # Momentum to velocity
                grid_velocity += inverse_mass * self.grid_node_external_impulse[frame, grid_x, grid_y, grid_z]
                grid_velocity += self.dt * self.gravity  # gravity

                # Apply boundary conditions at domain edges
                if grid_x < self.bound and grid_velocity[0] < 0:
                    grid_velocity[0] = 0  # Left boundary
                if grid_x > self.n_grid - self.bound and grid_velocity[0] > 0:
                    grid_velocity[0] = 0  # Right boundary
                if grid_y < self.bound and grid_velocity[1] < 0:
                    grid_velocity[1] = 0  # Bottom boundary
                if grid_y > self.n_grid - self.bound and grid_velocity[1] > 0:
                    grid_velocity[1] = 0  # Top boundary
                if grid_z < self.bound and grid_velocity[2] < 0:
                    grid_velocity[2] = 0  # Front boundary
                if grid_z > self.n_grid - self.bound and grid_velocity[2] > 0:
                    grid_velocity[2] = 0  # Back boundary

                self.grid_node_velocity_out[frame, grid_x, grid_y, grid_z] = grid_velocity

    @ti.kernel
    def g2p(self, frame:ti.i32):
        """
        Grid to particle (G2P) step in MPM.
        Updates particle properties from grid node values using interpolation.
        
        Args:
            frame: current simulation frame (dimensionless)
        """
        for particle_id in range(self.actual_total_num_particles): # grid to particle (G2P)
            # grid_base_index: grid cell indices (dimensionless)
            grid_base_index = (self.particle_position[frame, particle_id] * self.inverse_mpm_grid_cube_size - 0.5).cast(int)
            # particle_grid_diff: fractional position within grid cell (dimensionless)
            particle_grid_diff = self.particle_position[frame, particle_id] * self.inverse_mpm_grid_cube_size - grid_base_index.cast(float)
            # weight_functions: interpolation weights (dimensionless)
            weight_functions = [0.5 * (1.5 - particle_grid_diff) ** 2, 0.75 - (particle_grid_diff - 1.0) ** 2, 0.5 * (particle_grid_diff - 0.5) ** 2]
            # updated_velocity: updated particle velocity in m/s
            updated_velocity = ti.Vector.zero(float, 3)
            # updated_affine: updated affine velocity field (dimensionless)
            updated_affine = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                # loop over 3x3 grid node neighborhood
                # grid_relative_offset: relative offset to grid node (dimensionless)
                grid_relative_offset = ti.Vector([i, j, k]).cast(float) - particle_grid_diff
                # grid_node_velocity: grid node velocity in m/s
                grid_node_velocity = self.grid_node_velocity_out[frame, grid_base_index + ti.Vector([i, j, k])]
                # weight: interpolation weight (dimensionless)
                weight = weight_functions[i][0] * weight_functions[j][1] * weight_functions[k][2]
                updated_velocity += weight * grid_node_velocity
                updated_affine += 4 * self.inverse_mpm_grid_cube_size * weight * grid_node_velocity.outer_product(grid_relative_offset)

            # fixed_particle: flag for fixed particles (dimensionless)
            # fixed_particle = self.is_fixed[particle_id] == 1
            fixed_particle = False
            if fixed_particle:
                self.particle_velocity[frame+1, particle_id] = ti.Vector([0.0, 0.0, 0.0])
                self.affine_velocity_field[frame+1, particle_id] = ti.Matrix.zero(float, 3, 3)
                self.particle_position[frame+1, particle_id] = self.particle_position[frame, particle_id]
            else:
                self.particle_velocity[frame+1, particle_id] = updated_velocity
                self.affine_velocity_field[frame+1, particle_id] = updated_affine
                self.particle_position[frame+1, particle_id] = self.particle_position[frame, particle_id] + self.dt * updated_velocity  # advection

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.actual_total_num_particles):
            self.particle_position[target, p] = self.particle_position[source, p]
            self.particle_velocity[target, p] = self.particle_velocity[source, p]
            self.affine_velocity_field[target, p] = self.affine_velocity_field[source, p]
            self.deformation_gradient[target, p] = self.deformation_gradient[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray(), cache_C_0: ti.types.ndarray(), cache_F_0: ti.types.ndarray()):
        for p in range(self.actual_total_num_particles):
            for i in ti.static(range(3)):
                self.particle_position[f, p][i] = cache_x_0[p,i]
                self.particle_velocity[f, p][i] = cache_v_0[p,i]

            for i, j in ti.ndrange(3, 3):
                self.affine_velocity_field[f, p][i, j] = cache_C_0[p, i, j]
                self.deformation_gradient[f, p][i, j] = cache_F_0[p, i, j]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray(), cache_C_0: ti.types.ndarray(), cache_F_0: ti.types.ndarray()):
        for p in range(self.actual_total_num_particles):
            for i in ti.static(range(3)):
                cache_x_0[p,i] = self.particle_position[f, p][i]
                cache_v_0[p,i] = self.particle_velocity[f, p][i]

            for i, j in ti.ndrange(3, 3):
                cache_C_0[p, i, j] = self.affine_velocity_field[f, p][i, j]
                cache_F_0[p, i, j] = self.deformation_gradient[f, p][i, j]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['x_0'] = torch.zeros((self.actual_total_num_particles, 3), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['v_0'] = torch.zeros((self.actual_total_num_particles, 3), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['C_0'] = torch.zeros((self.actual_total_num_particles, 3, 3), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['F_0'] = torch.zeros((self.actual_total_num_particles, 3, 3), dtype=TC_TYPE, device=device)

        self.add_step_to_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'], self.cache[cur_step_name]['C_0'], self.cache[cur_step_name]['F_0'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)

        self.load_step_from_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'], self.cache[cur_step_name]['C_0'], self.cache[cur_step_name]['F_0'])

    def get_keypoint_index(self) -> int:
        """
        Get the index of the point closest to phantom_centroid_pose in x,y plane
        and with minimum z coordinate.
        
        Returns:
            int: Index of the keypoint
        """
        # Get particle positions as numpy array for frame 0
        positions = self.particle_position.to_numpy()[0]
        
        # Get phantom centroid x,y coordinates from coordinates
        centroid_x = self.coordinates['phantom_centroid_pose'][0]
        centroid_y = self.coordinates['phantom_centroid_pose'][1]
        
        # Create mask for points within 0.01 of centroid x,y
        x_mask = np.abs(positions[:, 0] - centroid_x) < 0.01
        y_mask = np.abs(positions[:, 1] - centroid_y) < 0.01
        xy_mask = x_mask & y_mask
        
        # Among points that satisfy xy criteria, find one with minimum z
        valid_points = positions[xy_mask]
        if len(valid_points) == 0:
            raise ValueError("No points found within 0.001 of centroid x,y coordinates")
            
        min_z_idx = np.argmin(valid_points[:, 2])
        # Get the original index
        keypoint_idx = np.where(xy_mask)[0][min_z_idx]
        self.keypoint_idx[None] = int(keypoint_idx)
        
        return np.array([keypoint_idx])

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
        positions = self.particle_position.to_numpy()[f]
        
        # Extract coordinates for the specified indices
        coordinates = positions[keypoint_indices]
        
        return coordinates
