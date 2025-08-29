import numpy as np
import math
import json
import pickle

from difftactile.main.constants import *

class PreMain:
    def __init__(self):
        self.sensor_r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        self.press_depth_slide = SYSTEM_PARAMS.trajectory.press_depth_slide

    @staticmethod
    def arr_str(xs):
        if len(xs.shape) == 1:
            return "[" + ", ".join([f'{x:.6f}' for x in xs]) + "]"
        else:
            res = []
            res.append('[')
            for i in range(xs.shape[0]):
                res.append(PreMain.arr_str(xs[i]))
                res.append(',\n')
            res.append(']')
            return ''.join(res)

    @staticmethod
    def compute_learning_rate(
            gradient, 
            value, 
            relative_step_size=0.5
        ):
        if abs(gradient) > 1e-6:
            return abs(relative_step_size * value / gradient)
        else:
            return 0.0

    @staticmethod
    def compute_and_print_learning_rates():
        with open(SYSTEM_PARAMS.files.optimisation_loop_calibration, 'r') as f:
            gradients = json.load(f)
        
        return {
            'learning_rates': {
                'vitactip': {
                    'youngs_modulus': PreMain.compute_learning_rate(
                        gradients['vitactip_youngs_modulus'],
                        SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
                    )
                },
                'phantom': {
                    'youngs_modulus_0': PreMain.compute_learning_rate(
                        gradients['phantom_youngs_modulus_0'],
                        SYSTEM_PARAMS.phantom.silicone.youngs_modulus
                    ),
                    'youngs_modulus_1': PreMain.compute_learning_rate(
                        gradients['phantom_youngs_modulus_1'],
                        SYSTEM_PARAMS.phantom.hard_plastic.youngs_modulus
                    )
                },
                'contact': {
                    'coulomb_friction_coeff': PreMain.compute_learning_rate(
                        gradients['coulomb_friction_coeff'],
                        SYSTEM_PARAMS.contact.coulomb_friction_coeff
                    ),
                    'normal_stiffness': PreMain.compute_learning_rate(
                        gradients['normal_stiffness'],
                        SYSTEM_PARAMS.contact.normal_stiffness
                    ),
                    'tangential_stiffness': PreMain.compute_learning_rate(
                        gradients['tangential_stiffness'],
                        SYSTEM_PARAMS.contact.tangential_stiffness
                    ),
                    'normal_damping': PreMain.compute_learning_rate(
                        gradients['normal_damping'],
                        SYSTEM_PARAMS.contact.normal_damping
                    )
                }
            }
        }

    def get_num_mpm_grid_nodes(self):
        num_nodes = []
        layouts = []
        for i in range(self.p_dims.shape[0]):
            n, layout = self.get_num_mpm_grid_nodes_single_dim(i)
            num_nodes.append(n)
            layouts.append(layout)
        self.num_nodes = np.array(num_nodes, dtype=int)
        self.layouts = layouts

    def get_num_mpm_grid_nodes_single_dim(self, i):
        l = self.p_dims[i]
        n_phantom = l / self.d
        n_phantom = int(np.ceil(n_phantom))
        n_pad_single = self.pad_single / self.d
        n_pad_single = int(np.ceil(n_pad_single))
        n_v0_single = 3+1
        if i == 0 or i == 1:
            arr = np.array([
                n_v0_single,
                n_pad_single,
                n_phantom,
                n_pad_single,
                n_v0_single,
            ], dtype=int)
            n = arr.sum()+1
            return n, arr
        else:
            arr = np.array([
                n_v0_single,
                n_phantom,
                n_pad_single,
                n_v0_single,
            ], dtype=int)
            n = arr.sum()+1
            return n, arr
    
    def min_coords_single_dim(self, i):
        if i == 0 or i == 1:
            return sum(self.layouts[i][:2])*self.d
        else:
            return sum(self.layouts[i][:1])*self.d

    def get_min_coords(self):
        res = []
        for i in range(len(self.layouts)):
            res.append(
                self.min_coords_single_dim(i)
            )
        self.min_coords = np.array(res, dtype=float)

    def compute_mpm_grid_node_vein_mask(
        self,
    ):
        vy = SYSTEM_PARAMS.geometry.grid_vein.vy
        vz = SYSTEM_PARAMS.geometry.grid_vein.vz
        r0 = SYSTEM_PARAMS.geometry.grid_vein.r0
        d = self.d
        r1 = 1.5*d
        r = r0+r1
        nx, ny, nz = self.num_nodes
        grid_node_positions = np.zeros(shape=(ny, nz, 2), dtype=float)
        for i in range(grid_node_positions.shape[0]):
            for j in range(grid_node_positions.shape[1]):
                grid_node_positions[i, j] = np.array([i*d, j*d], dtype=float)
        grid_node_positions[:, :, 0] -= self.min_coords[1]
        grid_node_positions[:, :, 1] -= self.min_coords[2]
        distances = np.sqrt(np.sum((grid_node_positions - np.array([vy, vz]))**2, axis=2))
        grid_node_v0_mask = distances <= r
        r_interm_outer = r+1.5*d
        path = SYSTEM_PARAMS.files.grid_node_v0_mask
        np.savez(
            path,
            grid_node_v0_mask=grid_node_v0_mask,
        )

    def go(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
            mesh_data = pickle.load(f)

        mesh_data['mean_particle_spacing'] /= 1000

        px = SYSTEM_PARAMS.geometry.phantom_x_length
        py = SYSTEM_PARAMS.geometry.phantom_y_length
        pz = SYSTEM_PARAMS.geometry.phantom_z_length
        self.p_dims = np.array([px, py, pz], dtype=float)
        self.d = SYSTEM_PARAMS.phantom.mpm_grid_cube_size
        self.pad_single = SYSTEM_PARAMS.geometry.mpm_pad_single

        self.get_num_mpm_grid_nodes()
        nx, ny, nz = self.num_nodes

        d = SYSTEM_PARAMS.phantom.mpm_grid_cube_size
        self.get_min_coords()
        min_coords = self.min_coords

        self.compute_mpm_grid_node_vein_mask()

        print(f'SYSTEM_PARAMS.phantom.mpm_grid_cube_size: {d}')
        print(f'min_coord: {min_coords}')

        phantom_closest_vertex = min_coords.copy()

        dist_from_floor = phantom_closest_vertex[2] - min_coords[2]

        phantom_dimensions = np.array([
            SYSTEM_PARAMS.geometry.phantom_x_length, 
            SYSTEM_PARAMS.geometry.phantom_y_length, 
            SYSTEM_PARAMS.geometry.phantom_z_length
        ], dtype=float)
        phantom_volume = phantom_dimensions[0] * phantom_dimensions[1] * phantom_dimensions[2]

        contact_surface_area = math.pi * (SYSTEM_PARAMS.gmsh_mm.radii[0] / 1_000) ** 2

        phantom_furthest_vertex = phantom_closest_vertex + phantom_dimensions

        print(f'phantom_dimensions: {phantom_dimensions}')
        print(f'phantom_closest_vertex: {phantom_closest_vertex}')
        print(f'phantom_furthest_vertex: {phantom_furthest_vertex}')
        print(f'phantom_volume: {phantom_volume}')
        print(f'contact_surface_area: {contact_surface_area}')
        print(f'dist_from_floor: {dist_from_floor:0.3e}')

        phantom_max_dim = np.max(phantom_dimensions)
        object_scale = phantom_max_dim
        phantom_normalised_spans = phantom_dimensions / phantom_max_dim / 2
        phantom_scaled_spans = phantom_normalised_spans * object_scale

        pps = SYSTEM_PARAMS.phantom.particle_spacing
        phantom_num_1d_particles = int(np.ceil(phantom_max_dim/pps)+1)

        phantom_min_max_particle_spacing = pps

        print(f'phantom_normalised_spans: {phantom_normalised_spans}')
        print(f'phantom_scaled_spans: {phantom_scaled_spans}')
        print(f'phantom_min_max_particle_spacing: {phantom_min_max_particle_spacing}')
        print(f"mesh_data['mean_particle_spacing']: {mesh_data['mean_particle_spacing']}")

        phantom_difftactile_position = phantom_closest_vertex + phantom_dimensions/2
        vitactip_tip_position = phantom_closest_vertex.copy()
        vitactip_tip_position[0] += phantom_dimensions[0]/2
        vitactip_tip_position[1] += -self.sensor_r
        vitactip_tip_position[2] += phantom_dimensions[2]-self.press_depth_slide

        system_params_computed = {
            "phantom_closest_vertex": phantom_closest_vertex.tolist(),
            "phantom_centroid_pose": phantom_difftactile_position.tolist() + SYSTEM_PARAMS.geometry.phantom_orientation,
            "phantom_top_surface_z": phantom_furthest_vertex[2],
            "phantom_dimensions": phantom_dimensions.tolist(),
            "vitactip_tip_position": vitactip_tip_position.tolist(),
            "vitactip_tip_pose": vitactip_tip_position.tolist() + SYSTEM_PARAMS.geometry.sensor_orientation,
            "phantom_volume": phantom_volume,
            "object_scale": object_scale,
            "phantom_min_max_particle_spacing": phantom_min_max_particle_spacing,
            "vitactip_mean_particle_spacing": mesh_data['mean_particle_spacing'],
            "contact_surface_area": contact_surface_area,
            "phantom": {
                "num_particles_cube_1d": int(phantom_num_1d_particles),
                "n_grid_x": int(nx),
                "n_grid_y": int(ny),
                "n_grid_z": int(nz)
            },
            "min_coords": min_coords.tolist()
        }

        if False:
            learning_rates_dict = compute_and_print_learning_rates()
            res = system_params_computed | learning_rates_dict
        
        res = system_params_computed
        with open(SYSTEM_PARAMS.files.system_params_computed, 'w') as f:
            json.dump(res, f, indent=2)

        print('difftactile coordinates')
        print(f'phantom_closest_vertex: {PreMain.arr_str(phantom_closest_vertex)}')
        print(f'phantom centre of mass: {PreMain.arr_str(phantom_difftactile_position)}')
        print(f'sensor dome tip: {PreMain.arr_str(vitactip_tip_position)}')


def main():
    pm = PreMain()
    pm.go()


if __name__ == '__main__':
    main()
