"""
a class to describe multi-material objects with mpm
"""
import os
import taichi as ti
import numpy as np
from scipy.spatial.transform import Rotation as R
from difftactile.object_model.obj_loader import ObjLoader
import torch

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class MultiObj:
    def __init__(self, dt=5e-5, sub_steps=80, obj_name=None, space_scale = 1.0, obj_scale = 1.0, mesh_density = 2, mass_density = 1):
        self.sub_steps = sub_steps
        self.dt = dt
        self.init_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.init_ori = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.init_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.rot_h = ti.Matrix.field(3, 3, ti.f32, shape=())
        self.trans_h = ti.Matrix.field(4, 4, ti.f32, shape=())

        self.bound = 3
        self.dim = 3
        self.n_grid = 64
        self.space_scale = space_scale
        self.obj_scale = obj_scale
        self.mass_density = mass_density * self.obj_scale
        self.particle_density = self.n_grid * mesh_density * obj_scale / space_scale
        self.gravity = ti.Vector([0.0, 0.0, 0.0])

        ## parameters for object
        self.obj_name = obj_name
        if self.obj_name is not None:
            data_path = os.path.join("..", "meshes", "objects", self.obj_name)
            obj_loader = ObjLoader(data_path, particle_density = int(self.particle_density))
            obj_loader.generate_particles()
            self.n_particles = len(obj_loader.particles)
            self.particles = ti.Vector.field(3, dtype=float, shape=self.n_particles)
            self.particles.from_numpy((obj_loader.particles * self.obj_scale).astype(np.float32))
            self.titles = ti.field(dtype=int, shape=self.n_particles)
            # print("Object model is loaded!")
        else:
            print("ERR on loading object model")

        self.grid_node_length = float(self.space_scale / self.n_grid)
        self.inverse_grid_node_length =  1 / self.grid_node_length
        self.particle_volume = (self.grid_node_length * self.obj_scale) ** 3
        self.particle_mass_density = self.mass_density * 1.0
        self.particle_mass = self.particle_volume * self.particle_mass_density
        self.eps = 1e-5
        
        self.E_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)
        self.nu_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)
        self.lamda_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)
        self.mu_0 = ti.field(dtype=ti.f32, shape=(2,), needs_grad=False)

        self.set_stiffness((5e3, 5e4))

        self.particle_position = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)  # position
        self.particle_velocity = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)  # velocity

        self.affine_velocity = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)  # affine velocity field
        self.F_new = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)
        self.deformation_gradient = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)  # deformation gradient
        self.U_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)
        self.V_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)
        self.S_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=False)

        self.grid_node_momentum_in = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)  # grid node momentum/velocity
        self.grid_node_momentum_out = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)  # grid node momentum/velocity
        self.grid_node_mass = ti.field(dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)  # grid node mass
        self.grid_node_external_force = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=False)  # grid node external force
        self.grid_occupy = ti.field(dtype=int, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid))
        self.total_surface_external_force = ti.Vector.field(3, float, shape=(self.sub_steps), needs_grad=False)
        
        self.cache = dict() # for grad backward

        self.group_cardinality = ti.field(dtype=int, shape=(2,), needs_grad=False)

        self.cylinder_cx = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_cy = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_cz = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_theta = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_h = ti.field(dtype=float, shape=(), needs_grad=False)
        self.cylinder_r = ti.field(dtype=float, shape=(), needs_grad=False)

    def set_stiffness(self, stiffness_tuple):
        # Material A (normal tissue/fat)
        self.E_0[0], self.nu_0[0] = stiffness_tuple[0] * self.space_scale, 0.48  # Young's modulus ~5 kPa, nearly incompressible
        # Material B (tumor)
        self.E_0[1], self.nu_0[1] = stiffness_tuple[1] * self.space_scale, 0.48  # Young's modulus ~50 kPa, nearly incompressible

        for item in range(2):
            self.mu_0[item] = self.E_0[item] / 2 / (1 + self.nu_0[item])
            self.lamda_0[item] = self.E_0[item] * self.nu_0[item] / (1 + self.nu_0[item]) / (1 - 2 * self.nu_0[item])
    
    def set_tumour_cylinder(self, cylinder_tuple):
        cx, cy, cz, theta, h, r = cylinder_tuple
        self.cylinder_cx[None] = cx
        self.cylinder_cy[None] = cy
        self.cylinder_cz[None] = cz
        self.cylinder_theta[None] = np.deg2rad(theta)
        self.cylinder_h[None] = h
        self.cylinder_r[None] = r

    @ti.kernel
    def preprocess_obj(self):
        for item in range(self.n_particles):
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

    def init(self, pos, ori, vel, cylinder_tuple, stiffness_tuple, tumour_present):
        self.set_stiffness(stiffness_tuple)
        if tumour_present:
            self.titles.fill(-1)
            self.group_cardinality.fill(0)
            self.set_tumour_cylinder(cylinder_tuple)
            self.preprocess_obj()
        else:
            self.group_cardinality[0] = self.n_particles
            self.group_cardinality[1] = 0
            self.titles.fill(0)
        print(f'tumour_present: {tumour_present}, healthy: {self.group_cardinality[0]}, tumour: {self.group_cardinality[1]}')
        tumour_particles_absent = self.group_cardinality[1] == 0
        if tumour_present and tumour_particles_absent:
            raise Exception("tumour present but no tumour particles generated!")
        tumour_present = not tumour_particles_absent
        self.set_object_params(pos, ori, vel)
        self.init_object()

    def set_object_params(self, position, orientation, velocity):
        self.init_pos[None] = position
        self.init_ori[None] = orientation
        self.init_vel[None] = velocity

        rot = R.from_rotvec(np.deg2rad([orientation[0], orientation[1], orientation[2]]))
        rot_mat = rot.as_matrix()
        trans_mat = np.eye(4)
        trans_mat[0:3,0:3] = rot_mat
        trans_mat[0,3] = position[0]; trans_mat[1,3] = position[1]; trans_mat[2,3] = position[2]
        self.rot_h[None] = rot_mat.tolist()
        self.trans_h[None] = trans_mat.tolist()

    @ti.kernel
    def init_object(self):
        for i in range(self.n_particles):
            before_t_pos = self.particles[i]
            after_t_pos = self.trans_h[None] @ ti.Vector([before_t_pos[0], before_t_pos[1], before_t_pos[2], 1.0]) # 4 x 1 homogeneous
            self.particle_position[0,i] = ti.Vector([after_t_pos[0], after_t_pos[1], after_t_pos[2]])
            self.particle_velocity[0,i] = ti.Matrix([self.init_vel[None][0], self.init_vel[None][1], self.init_vel[None][2]])
            self.deformation_gradient[0,i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @ti.kernel
    def reset(self):
        self.grid_node_momentum_in.fill(0.0)
        self.grid_node_momentum_out.fill(0.0)
        self.grid_node_mass.fill(0.0)
        self.grid_node_external_force.fill(0.0)
        self.grid_occupy.fill(0.0)
        self.total_surface_external_force.fill(0.0)

    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            self.total_surface_external_force[f] += self.grid_node_external_force[f, i, j, k] /self.dt

    @ti.func
    def update_contact_force(self, external_force, f, i, j, k):
        self.grid_node_external_force[f, i, j, k] += external_force * self.dt

    @ti.kernel
    def compute_new_F(self, f: ti.i32):
        for p in range(self.n_particles):
            self.F_new[f, p] = (ti.Matrix.diag(dim=3, val=1) + self.dt * self.affine_velocity[f, p]) @ self.deformation_gradient[f, p]

    @ti.kernel
    def svd(self, f: ti.i32):
        for p in range(self.n_particles):
            self.U_svd[f, p], self.S_svd[f, p], self.V_svd[f, p] = ti.svd(self.F_new[f, p])

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            self.F_new.grad[f, p] += self.single_svd_grad(f, p)

    @ti.func
    def clamp(self, a: ti.f32):
        if a>=0:
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
        s = ti.Vector([self.S_svd[f, p][0, 0], self.S_svd[f, p][1, 1], self.S_svd[f, p][2, 2]]) ** 2
        ff = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            if i == j:
                ff[i, j] = 0
            else:
                ff[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = self.U_svd[f, p] @ ((ff * (ut @ self.U_svd.grad[f, p] - self.U_svd.grad[f, p].transpose() @ self.U_svd[f, p])) @ self.S_svd[f, p]) @ vt
        v_term = self.U_svd[f, p] @ (self.S_svd[f, p] @ ((ff * (vt @ self.V_svd.grad[f, p] - self.V_svd.grad[f, p].transpose() @ self.V_svd[f, p])) @ vt))
        return u_term + v_term + s_term

    @ti.func
    def H_svd_grad(self, f: ti.i32):
        vt = self.V[f].transpose()
        ut = self.U[f].transpose()
        s_term = self.U[f] @ self.S.grad[f] @ vt

        s = ti.Vector.zero(ti.f32, 3)
        s = ti.Vector([self.S[f][0, 0], self.S[f][1, 1], self.S[f][2, 2]]) ** 2
        ff = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            if i == j:
                ff[i, j] = 0
            else:
                ff[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = self.U[f] @ ((ff * (ut @ self.U.grad[f] - self.U.grad[f].transpose() @ self.U[f])) @ self.S[f]) @ vt
        v_term = self.U[f] @ (self.S[f] @ ((ff * (vt @ self.V.grad[f] - self.V.grad[f].transpose() @ self.V[f])) @ vt))
        return u_term + v_term + s_term

    @ti.kernel
    def p2g(self, frame:ti.i32):
        for particle_id in range(self.n_particles):
            shear_modulus, bulk_modulus = self.mu_0[self.titles[particle_id]], self.lamda_0[self.titles[particle_id]]
            grid_base_index = (self.particle_position[frame, particle_id] * self.inverse_grid_node_length - 0.5).cast(int)
            particle_grid_diff = self.particle_position[frame, particle_id] * self.inverse_grid_node_length - grid_base_index.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=particle_grid_diff, particle_grid_diff-1,particle_grid_diff-2]
            weight_functions = [0.5 * (1.5 - particle_grid_diff) ** 2, 0.75 - (particle_grid_diff - 1) ** 2, 0.5 * (particle_grid_diff - 0.5) ** 2]

            volume_ratio = (self.S_svd[frame, particle_id]).determinant()
            rotation_matrix = self.U_svd[frame, particle_id] @ self.V_svd[frame, particle_id].transpose()
            cauchy_stress = 2 * shear_modulus * (self.F_new[frame, particle_id] - rotation_matrix) @ self.F_new[frame, particle_id].transpose() + ti.Matrix.identity(float, 3) * bulk_modulus * volume_ratio * (volume_ratio - 1)

            force_term = (-self.dt * self.particle_volume * 4 * self.inverse_grid_node_length * self.inverse_grid_node_length) * cauchy_stress

            momentum_contrib = force_term + self.particle_mass * self.affine_velocity[frame, particle_id]

            # Loop over 3x3 grid node neighborhood
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                grid_offset = ti.Vector([i, j, k])
                dist_to_grid = (grid_offset.cast(float) - particle_grid_diff) * self.grid_node_length
                weight = weight_functions[i][0] * weight_functions[j][1] * weight_functions[k][2]
                self.grid_node_momentum_in[frame, grid_base_index + grid_offset] += weight * (self.particle_mass * self.particle_velocity[frame, particle_id] + momentum_contrib @ dist_to_grid)
                self.grid_node_mass[frame, grid_base_index + grid_offset] += weight * self.particle_mass

            self.deformation_gradient[frame+1, particle_id] = self.F_new[frame, particle_id]

    @ti.kernel
    def check_grid_occupy(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.grid_node_mass[f, i, j, k] > self.eps:
                self.grid_occupy[f, i, j, k] = 1

    @ti.kernel
    def grid_op(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.grid_occupy[f, i, j, k] == 1:
                inv_m = 1 / (self.grid_node_mass[f, i, j, k]+self.eps)

                v_out = ti.Vector([0.0, 0.0, 0.0])
                v_out += inv_m * self.grid_node_momentum_in[f, i, j, k] # Momentum to velocity
                v_out += inv_m * self.grid_node_external_force[f, i, j, k]
                v_out += self.dt * self.gravity  # gravity

                if i < self.bound and v_out[0] < 0:
                    v_out[0] = 0  # Boundary conditions
                if i > self.n_grid - self.bound and v_out[0] > 0:
                    v_out[0] = 0
                if j < self.bound and v_out[1] < 0: # < 3
                    v_out[1] = 0
                if j > self.n_grid - self.bound and v_out[1] > 0:
                    v_out[1] = 0
                if k < self.bound and v_out[2] < 0: # < 3
                    v_out[2] = 0
                if k > self.n_grid - self.bound and v_out[2] > 0:
                    v_out[2] = 0

                self.grid_node_momentum_out[f, i, j, k] = v_out

    @ti.kernel
    def g2p(self, f:ti.i32):
        for p in range(self.n_particles): # grid to particle (G2P)
            base = (self.particle_position[f, p] * self.inverse_grid_node_length - 0.5).cast(int)
            fx = self.particle_position[f, p] * self.inverse_grid_node_length - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_node_momentum_out[f, base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * self.inverse_grid_node_length * weight * g_v.outer_product(dpos)

            self.particle_velocity[f+1, p], self.affine_velocity[f+1, p] = new_v, new_C
            self.particle_position[f+1, p] = self.particle_position[f, p] + self.dt * new_v  # advection

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.particle_position[target, p] = self.particle_position[source, p]
            self.particle_velocity[target, p] = self.particle_velocity[source, p]
            self.affine_velocity[target, p] = self.affine_velocity[source, p]
            self.deformation_gradient[target, p] = self.deformation_gradient[source, p]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.particle_position.grad[target, p] = self.particle_position.grad[source, p]
            self.particle_velocity.grad[target, p] = self.particle_velocity.grad[source, p]
            self.affine_velocity.grad[target, p] = self.affine_velocity.grad[source, p]
            self.deformation_gradient.grad[target, p] = self.deformation_gradient.grad[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray(), cache_C_0: ti.types.ndarray(), cache_F_0: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                self.particle_position[f, p][i] = cache_x_0[p,i]
                self.particle_velocity[f, p][i] = cache_v_0[p,i]

            for i, j in ti.ndrange(self.dim, self.dim):
                self.affine_velocity[f, p][i, j] = cache_C_0[p, i, j]
                self.deformation_gradient[f, p][i, j] = cache_F_0[p, i, j]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray(), cache_C_0: ti.types.ndarray(), cache_F_0: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                cache_x_0[p,i] = self.particle_position[f, p][i]
                cache_v_0[p,i] = self.particle_velocity[f, p][i]

            for i, j in ti.ndrange(self.dim, self.dim):
                cache_C_0[p, i, j] = self.affine_velocity[f, p][i, j]
                cache_F_0[p, i, j] = self.deformation_gradient[f, p][i, j]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['x_0'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['v_0'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['C_0'] = torch.zeros((self.n_particles, self.dim, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['F_0'] = torch.zeros((self.n_particles, self.dim, self.dim), dtype=TC_TYPE, device=device)

        self.add_step_to_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'], self.cache[cur_step_name]['C_0'], self.cache[cur_step_name]['F_0'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)

        self.load_step_from_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'], self.cache[cur_step_name]['C_0'], self.cache[cur_step_name]['F_0'])

    @ti.kernel
    def clear_loss_grad(self):
        self.E_0.grad.fill(0.0)
        self.nu_0.grad.fill(0.0)
        self.mu_0.grad.fill(0.0)
        self.lamda_0.grad.fill(0.0)

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        self.grid_node_momentum_in.grad.fill(0.0)
        self.grid_node_momentum_out.grad.fill(0.0)
        self.grid_node_mass.grad.fill(0.0)
        self.grid_node_external_force.grad.fill(0.0)
        self.F_new.grad.fill(0.0)
        self.U_svd.grad.fill(0.0)
        self.V_svd.grad.fill(0.0)
        self.S_svd.grad.fill(0.0)
        for p in range(self.n_particles):
            for t in range(f):
                self.particle_position.grad[t,p].fill(0.0)
                self.particle_velocity.grad[t,p].fill(0.0)
                self.affine_velocity.grad[t,p].fill(0.0)
                self.deformation_gradient.grad[t,p].fill(0.0)
