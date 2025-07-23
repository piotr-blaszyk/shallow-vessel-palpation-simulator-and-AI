import numpy as np
import math
import json
import pickle

from difftactile.main.constants import *

def arr_str(xs):
    if len(xs.shape) == 1:
        return "[" + ", ".join([f'{x:.6f}' for x in xs]) + "]"
    else:
        res = []
        res.append('[')
        for i in range(xs.shape[0]):
            res.append(arr_str(xs[i]))
            res.append(',\n')
        res.append(']')
        return ''.join(res)

def compute_learning_rate(
        gradient, 
        value, 
        percentage_update=1.0
    ):
    return abs(percentage_update * value / gradient)

def compute_and_print_learning_rates():
    with open(SYSTEM_PARAMS.files.optimisation_loop_calibration, 'r') as f:
        gradients = json.load(f)
    
    return {
        'learning_rates': {
            'vitactip': {
                'youngs_modulus': compute_learning_rate(
                    gradients['vitactip_youngs_modulus'],
                    SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
                )
            },
            'phantom': {
                'youngs_modulus_0': compute_learning_rate(
                    gradients['phantom_youngs_modulus_0'],
                    SYSTEM_PARAMS.phantom.silicone.youngs_modulus
                ),
                'youngs_modulus_1': compute_learning_rate(
                    gradients['phantom_youngs_modulus_1'],
                    SYSTEM_PARAMS.phantom.hard_plastic.youngs_modulus
                )
            },
            'contact': {
                'coulomb_friction_coeff': compute_learning_rate(
                    gradients['coulomb_friction_coeff'],
                    SYSTEM_PARAMS.contact.coulomb_friction_coeff
                ),
                'normal_stiffness': compute_learning_rate(
                    gradients['normal_stiffness'],
                    SYSTEM_PARAMS.contact.normal_stiffness
                ),
                'tangential_stiffness': compute_learning_rate(
                    gradients['tangential_stiffness'],
                    SYSTEM_PARAMS.contact.tangential_stiffness
                ),
                'normal_damping': compute_learning_rate(
                    gradients['normal_damping'],
                    SYSTEM_PARAMS.contact.normal_damping
                )
            }
        }
    }

def main():
    with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
        mesh_data = pickle.load(f)

    mesh_data['min_particle_spacing'] = {key: value/1000 for key, value in mesh_data['min_particle_spacing'].items()}
    mesh_data['max_particle_spacing'] = {key: value/1000 for key, value in mesh_data['max_particle_spacing'].items()}

    min_coord = SYSTEM_PARAMS.phantom.mpm_grid_cube_size * 3
    max_coord_x = (SYSTEM_PARAMS.phantom.n_grid_x - 3) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size
    max_coord_y = (SYSTEM_PARAMS.phantom.n_grid_y- 3) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size
    max_coord_z = (SYSTEM_PARAMS.phantom.n_grid_z - 3) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size
    print(f'SYSTEM_PARAMS.phantom.mpm_grid_cube_size: {SYSTEM_PARAMS.phantom.mpm_grid_cube_size}')
    print(f'min_coord: {min_coord}')
    print(f'max_coord_x: {max_coord_x}')
    print(f'max_coord_y: {max_coord_y}')
    print(f'max_coord_z: {max_coord_z}')

    phantom_closest_vertex = np.array([
        SYSTEM_PARAMS.phantom.mpm_grid_cube_size*int(SYSTEM_PARAMS.phantom.n_grid_x * 1/4), 
        SYSTEM_PARAMS.phantom.mpm_grid_cube_size*int(SYSTEM_PARAMS.phantom.n_grid_y * 1/4), 
        SYSTEM_PARAMS.phantom.mpm_grid_cube_size*3
    ], dtype=float)

    dist_from_floor = phantom_closest_vertex[2] - min_coord

    phantom_dimensions = np.array([
        SYSTEM_PARAMS.geometry.phantom_x_length, 
        SYSTEM_PARAMS.geometry.phantom_y_length, 
        SYSTEM_PARAMS.geometry.phantom_z_length
    ], dtype=float)
    phantom_volume = phantom_dimensions[0] * phantom_dimensions[1] * phantom_dimensions[2]

    contact_surface_area = math.pi * (SYSTEM_PARAMS.gmsh_mm.stem_wall_radius_outer / 1_000) ** 2

    phantom_furthest_vertex = phantom_closest_vertex + phantom_dimensions

    print(f'phantom_dimensions: {phantom_dimensions}')
    print(f'phantom_closest_vertex: {phantom_closest_vertex}')
    print(f'phantom_furthest_vertex: {phantom_furthest_vertex}')
    print(f'phantom_volume: {phantom_volume}')
    print(f'contact_surface_area: {contact_surface_area}')
    print(f'dist_from_floor: {dist_from_floor:0.3e}')

    assert np.min(phantom_closest_vertex) >= min_coord, "the phantom is outside of the manipulation cube"
    assert phantom_furthest_vertex[0] <= max_coord_x, "the phantom is outside of the manipulation cube"
    assert phantom_furthest_vertex[1] <= max_coord_y, "the phantom is outside of the manipulation cube"
    assert phantom_furthest_vertex[2] <= max_coord_z, "the phantom is outside of the manipulation cube"

    phantom_max_dim = np.max(phantom_dimensions)
    object_scale = phantom_max_dim
    phantom_normalised_spans = phantom_dimensions / phantom_max_dim / 2
    phantom_scaled_spans = phantom_normalised_spans * object_scale

    phantom_min_max_particle_spacing = phantom_max_dim / (SYSTEM_PARAMS.phantom.num_particles_cube_1d-1)

    print(f'phantom_normalised_spans: {phantom_normalised_spans}')
    print(f'phantom_scaled_spans: {phantom_scaled_spans}')
    print(f'phantom_min_max_particle_spacing: {phantom_min_max_particle_spacing}')
    print(f"mesh_data['min_particle_spacing']: {mesh_data['min_particle_spacing']}")

    phantom_difftactile_position = phantom_closest_vertex + phantom_scaled_spans

    vitactip_tip_position = np.array([
        phantom_difftactile_position[0] - SYSTEM_PARAMS.geometry.sensor_xy_radius,
        phantom_difftactile_position[1],
        phantom_difftactile_position[2] + phantom_dimensions[2] / 2 + SYSTEM_PARAMS.geometry.gap,
    ])

    for key in mesh_data['min_particle_spacing'].keys():
        mesh_data['min_particle_spacing_for_material'] = mesh_data['min_particle_spacing'][key]
        mesh_data['max_particle_spacing_for_material'] = mesh_data['max_particle_spacing'][key]
        particle_min_min_spacing_ratio = max(
            phantom_min_max_particle_spacing / mesh_data['min_particle_spacing_for_material'],
            mesh_data['min_particle_spacing_for_material'] / phantom_min_max_particle_spacing,
        )
        if key == 'all':
            if False:
                assert particle_min_min_spacing_ratio < 1.1, f"ratio of minimum particle spacing in phantom and ViTacTip is too high: {particle_min_min_spacing_ratio:0.2f}; phantom: {phantom_min_max_particle_spacing:0.2e}; ViTacTip: {mesh_data['min_particle_spacing_for_material']:0.2e}"
            particle_min_max_spacing_ratio = max(
                phantom_min_max_particle_spacing / mesh_data['min_particle_spacing_for_material'],
                phantom_min_max_particle_spacing / mesh_data['max_particle_spacing_for_material'],
                mesh_data['max_particle_spacing_for_material'] / mesh_data['min_particle_spacing_for_material'],
            )
            particle_min_max_spacing_ratio = max(particle_min_max_spacing_ratio, 1 / particle_min_max_spacing_ratio)
            if False:
                assert particle_min_min_spacing_ratio < 10.0, f"ratio of minimum and maximum global particle spacing is too high: {particle_min_max_spacing_ratio:0.2f}"

    # Create coordinates dictionary
    system_params_computed = {
        "phantom_closest_vertex": phantom_closest_vertex.tolist(),
        "phantom_centroid_pose": phantom_difftactile_position.tolist() + SYSTEM_PARAMS.geometry.phantom_orientation,
        "vitactip_tip_position": vitactip_tip_position.tolist(),
        "vitactip_tip_pose": vitactip_tip_position.tolist() + SYSTEM_PARAMS.geometry.sensor_orientation,
        "phantom_volume": phantom_volume,
        "min_coord": min_coord,
        "max_coord_x": max_coord_x,
        "max_coord_y": max_coord_y,
        "max_coord_z": max_coord_z,
        "object_scale": object_scale,
        "phantom_min_max_particle_spacing": phantom_min_max_particle_spacing,
        "vitactip_min_particle_spacing": mesh_data['min_particle_spacing'],
        "contact_surface_area": contact_surface_area,
    }

    learning_rates_dict = compute_and_print_learning_rates()
    res = system_params_computed | learning_rates_dict

    # Save coordinates to JSON file
    with open(SYSTEM_PARAMS.files.system_params_computed, 'w') as f:
        json.dump(res, f, indent=2)

    print('difftactile coordinates')
    print(f'phantom_closest_vertex: {arr_str(phantom_closest_vertex)}')
    print(f'phantom centre of mass: {arr_str(phantom_difftactile_position)}')
    print(f'sensor dome tip: {arr_str(vitactip_tip_position)}')