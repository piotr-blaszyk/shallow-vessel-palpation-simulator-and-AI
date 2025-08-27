import taichi as ti


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
def p2g(self, frame: ti.i32):
    for particle_id in range(self.actual_total_num_particles):
        shear_modulus = self.mu[self.titles[particle_id]]
        bulk_modulus = self.lam[self.titles[particle_id]]
        grid_base_index = (
            self.particle_position[frame, particle_id] * self.inverse_mpm_grid_cube_size
            - 0.5
        ).cast(int)
        particle_grid_diff = self.particle_position[
            frame, particle_id
        ] * self.inverse_mpm_grid_cube_size - grid_base_index.cast(float)
        weight_functions = [
            0.5 * (1.5 - particle_grid_diff) ** 2,
            0.75 - (particle_grid_diff - 1) ** 2,
            0.5 * (particle_grid_diff - 0.5) ** 2,
        ]
        volume_ratio = (self.S_svd[frame, particle_id]).determinant()
        rotation_matrix = (
            self.U_svd[frame, particle_id] @ self.V_svd[frame, particle_id].transpose()
        )
        cauchy_stress = 2 * shear_modulus * (
            self.trial_deformation_gradient[frame, particle_id] - rotation_matrix
        ) @ self.trial_deformation_gradient[
            frame, particle_id
        ].transpose() + ti.Matrix.identity(float, 3) * bulk_modulus * volume_ratio * (
            volume_ratio - 1
        )
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
                weight_functions[i][0] * weight_functions[j][1] * weight_functions[k][2]
            )
            self.grid_node_momentum_in[frame, grid_base_index + grid_offset] += (
                weight
                * (
                    self.healthy_tissue_particle_mass
                    * self.particle_velocity[frame, particle_id]
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
def update_internal_forces(self, frame: ti.i32):
    for tetra_idx in range(self.num_tetrahedra):
        vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx = self.tetrahedra[tetra_idx]
        vertex1_pos = self.vertices_deformed_A[frame, vertex1_idx]
        vertex2_pos = self.vertices_deformed_A[frame, vertex2_idx]
        vertex3_pos = self.vertices_deformed_A[frame, vertex3_idx]
        vertex4_pos = self.vertices_deformed_A[frame, vertex4_idx]
        deformation_matrix = ti.Matrix.cols(
            [
                vertex1_pos - vertex4_pos,
                vertex2_pos - vertex4_pos,
                vertex3_pos - vertex4_pos,
            ]
        )
        tetra_volume = ti.abs(deformation_matrix.determinant()) / 6
        deformation_gradient = (
            deformation_matrix @ self.initial_deformation_gradient_inverse[tetra_idx]
        )
        mu = self.mu[None]
        lam = self.lam[None]
        jacobian = deformation_gradient.determinant()
        first_invariant = (
            deformation_gradient.transpose() @ deformation_gradient
        ).trace()
        dJ_dF0 = deformation_gradient[:, 1].cross(deformation_gradient[:, 2])
        dJ_dF1 = deformation_gradient[:, 2].cross(deformation_gradient[:, 0])
        dJ_dF2 = deformation_gradient[:, 0].cross(deformation_gradient[:, 1])
        jacobian_derivative = ti.Matrix.cols([dJ_dF0, dJ_dF1, dJ_dF2])
        alpha = 1 + 0.75 * mu / lam
        stress_tensor = (
            mu * (1 - 1 / (first_invariant + 1)) * deformation_gradient
            + lam * (jacobian - alpha) * jacobian_derivative
        )
        force_matrix = (
            -tetra_volume
            * stress_tensor
            @ self.initial_deformation_gradient_inverse[tetra_idx].transpose()
        )
        vertex_indices = ti.Vector([vertex1_idx, vertex2_idx, vertex3_idx, vertex4_idx])
        for k in ti.static(range(3)):
            vertex_force = ti.Vector([force_matrix[j, k] for j in range(3)])
            self.vertex_velocities[frame, vertex_indices[k]] += (
                self.dt[None] * vertex_force / self.vertex_mass[vertex_indices[k]]
            )
            self.vertex_velocities[frame, vertex_indices[3]] += (
                -self.dt[None] * vertex_force / self.vertex_mass[vertex_indices[3]]
            )


@ti.kernel
def check_grid_occupy(self, f: ti.i32):
    for i, j, k in ti.ndrange(
        SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
        SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
        SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
    ):
        if self.grid_node_mass[f, i, j, k] > SYSTEM_PARAMS.phantom.mass_eps:
            self.grid_occupy[f, i, j, k] = 1


@ti.kernel
def check_collision(self, frame: ti.i32):
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
            closest_sensor_vertex_idx = self.vitactip.find_closest(
                grid_node_position, frame
            )
            self.contact_idx[frame, i, j, k] = closest_sensor_vertex_idx


@ti.kernel
def collision(self, frame: ti.i32):
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
            grid_node_velocity = self.phantom.grid_node_momentum_in[frame, i, j, k] / (
                self.phantom.grid_node_mass[frame, i, j, k] + SYSTEM_PARAMS.phantom.mass_eps
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


@ti.kernel
def grid_op(self, frame: ti.i32):
    for grid_x, grid_y, grid_z in ti.ndrange(
        SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
        SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
        SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
    ):
        if self.grid_occupy[frame, grid_x, grid_y, grid_z] == 1:
            inverse_mass = 1 / (
                self.grid_node_mass[frame, grid_x, grid_y, grid_z]
                + SYSTEM_PARAMS.phantom.mass_eps
            )
            grid_velocity = ti.Vector([0.0, 0.0, 0.0])
            grid_velocity += (
                inverse_mass * self.grid_node_momentum_in[frame, grid_x, grid_y, grid_z]
            )
            grid_velocity += (
                inverse_mass
                * self.grid_node_external_impulse[frame, grid_x, grid_y, grid_z]
            )
            grid_velocity += self.dt[None] * ti.Vector(SYSTEM_PARAMS.phantom.gravity)
            if grid_x < SYSTEM_PARAMS.phantom.bound and grid_velocity[0] < 0:
                grid_velocity[0] = 0
            if (
                grid_x > SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x - SYSTEM_PARAMS.phantom.bound
                and grid_velocity[0] > 0
            ):
                grid_velocity[0] = 0
            if grid_y < SYSTEM_PARAMS.phantom.bound and grid_velocity[1] < 0:
                grid_velocity[1] = 0
            if (
                grid_y > SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y - SYSTEM_PARAMS.phantom.bound
                and grid_velocity[1] > 0
            ):
                grid_velocity[1] = 0
            if grid_z < SYSTEM_PARAMS.phantom.bound and grid_velocity[2] < 0:
                grid_velocity[2] = 0
            if (
                grid_z > SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y - SYSTEM_PARAMS.phantom.bound
                and grid_velocity[2] > 0
            ):
                grid_velocity[2] = 0
            self.grid_node_velocity_out[frame, grid_x, grid_y, grid_z] = grid_velocity


@ti.kernel
def g2p(self, frame: ti.i32):
    for particle_id in range(self.actual_total_num_particles):
        grid_base_index = (
            self.particle_position[frame, particle_id] * self.inverse_mpm_grid_cube_size
            - 0.5
        ).cast(int)
        particle_grid_diff = self.particle_position[
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
            grid_relative_offset = ti.Vector([i, j, k]).cast(float) - particle_grid_diff
            grid_node_velocity = self.grid_node_velocity_out[
                frame, grid_base_index + ti.Vector([i, j, k])
            ]
            weight = (
                weight_functions[i][0] * weight_functions[j][1] * weight_functions[k][2]
            )
            updated_velocity += weight * grid_node_velocity
            updated_affine += (
                4
                * self.inverse_mpm_grid_cube_size
                * weight
                * grid_node_velocity.outer_product(grid_relative_offset)
            )
        if SYSTEM_PARAMS.phantom.fix_bottom_points == 1:
            self.particle_velocity[frame + 1, particle_id] = ti.Vector([0.0, 0.0, 0.0])
            self.affine_velocity_field[frame + 1, particle_id] = ti.Matrix.zero(
                float, 3, 3
            )
            self.particle_position[frame + 1, particle_id] = self.particle_position[
                frame, particle_id
            ]
        else:
            self.particle_velocity[frame + 1, particle_id] = updated_velocity
            self.affine_velocity_field[frame + 1, particle_id] = updated_affine
            self.particle_position[frame + 1, particle_id] = (
                self.particle_position[frame, particle_id]
                + self.dt[None] * updated_velocity
            )


@ti.kernel
def update_external_forces(self, frame: ti.i32):
    for vertex_idx in range(self.num_vertices):
        updated_velocity = ti.Vector([0.0, 0.0, 0.0])
        updated_velocity += self.vertex_velocities[frame, vertex_idx]
        updated_velocity += (
            self.dt[None]
            * self.contact_forces_on_vertices[frame, vertex_idx]
            / self.vertex_mass[vertex_idx]
        )
        is_fixed_layer = self.is_fixed_layer[vertex_idx] == 1
        if is_fixed_layer:
            updated_velocity = self.vertex_control_velocities[vertex_idx]
        self.vertex_velocities[frame + 1, vertex_idx] += updated_velocity
        self.vertices_deformed_A[frame + 1, vertex_idx] += self.vertices_deformed_A[
            frame, vertex_idx
        ]
        self.vertices_deformed_A[frame + 1, vertex_idx] += (
            self.dt[None] * updated_velocity
        )
        self.vertices_undeformed_A[frame + 1, vertex_idx] = (
            self.vertices_undeformed_A[frame, vertex_idx]
            + self.dt[None] * self.vertex_control_velocities[vertex_idx]
        )
