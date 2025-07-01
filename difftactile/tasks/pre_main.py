import numpy as np
import math
import json
import pickle

from ..tasks.constants import *

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

with open('../tasks/system-params.json', 'r') as f:
    params = json.load(f)
with open('../tasks/output/gmsh-mesh.pkl', 'rb') as f:
    mesh_data = pickle.load(f)
phantom_params = params['phantom']
contact_params = params['contact']
geometry_params = params['geometry']

vitactip_min_particle_spacing = mesh_data['min_particle_spacing']
vitactip_max_particle_spacing = mesh_data['max_particle_spacing']
vitactip_min_particle_spacing = {key: value/1000 for key, value in vitactip_min_particle_spacing.items()}
vitactip_max_particle_spacing = {key: value/1000 for key, value in vitactip_max_particle_spacing.items()}
phantom_num_particles_cube_1d = phantom_params['num_particles_cube_1d']
gap = geometry_params['gap']
phantom_orientation = geometry_params['phantom_orientation']
sensor_orientation = geometry_params['sensor_orientation']
bound = phantom_params['bound']
n_grid_x = phantom_params['n_grid_x']
n_grid_y = phantom_params['n_grid_y']
n_grid_z = phantom_params['n_grid_z']

mpm_grid_cube_size = phantom_params['mpm_grid_cube_size']
min_coord = mpm_grid_cube_size * 3
max_coord_x = (n_grid_x - 3) * mpm_grid_cube_size
max_coord_y = (n_grid_y- 3) * mpm_grid_cube_size
max_coord_z = (n_grid_z - 3) * mpm_grid_cube_size
print(f'mpm_grid_cube_size: {mpm_grid_cube_size}')
print(f'min_coord: {min_coord}')
print(f'max_coord_x: {max_coord_x}')
print(f'max_coord_y: {max_coord_y}')
print(f'max_coord_z: {max_coord_z}')

phantom_closest_vertex = np.array([mpm_grid_cube_size*16, mpm_grid_cube_size*16, mpm_grid_cube_size*3], dtype=float)

dist_from_floor = phantom_closest_vertex[2] - min_coord

phantom_h = geometry_params['phantom_h']
phantom_r = geometry_params['phantom_r']
phantom_d = phantom_r * 2
phantom_dimensions = np.array([phantom_d, phantom_d, phantom_h], dtype=float)
phantom_volume = math.pi * phantom_r ** 2 * phantom_h

contact_surface_area = math.pi * phantom_r ** 2

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

phantom_min_max_particle_spacing = phantom_max_dim / (phantom_num_particles_cube_1d-1)
phantom_to_cube_volume_ratio = phantom_volume / (np.max(phantom_scaled_spans) * 2) ** 3

print(f'phantom_normalised_spans: {phantom_normalised_spans}')
print(f'phantom_scaled_spans: {phantom_scaled_spans}')
print(f'phantom_min_max_particle_spacing: {phantom_min_max_particle_spacing}')
print(f'vitactip_min_particle_spacing: {vitactip_min_particle_spacing}')

phantom_difftactile_position = phantom_closest_vertex + phantom_scaled_spans

vitactip_tip_position = np.array([
    phantom_difftactile_position[0],
    phantom_difftactile_position[1],
    phantom_difftactile_position[2] + phantom_dimensions[2] / 2 + gap,
])

for key in vitactip_min_particle_spacing.keys():
    vitactip_min_particle_spacing_for_material = vitactip_min_particle_spacing[key]
    vitactip_max_particle_spacing_for_material = vitactip_max_particle_spacing[key]
    particle_min_min_spacing_ratio = max(
        phantom_min_max_particle_spacing / vitactip_min_particle_spacing_for_material,
        vitactip_min_particle_spacing_for_material / phantom_min_max_particle_spacing,
    )
    if key == 'all':
        assert particle_min_min_spacing_ratio < 1.1, f"ratio of minimum particle spacing in phantom and ViTacTip is too high: {particle_min_min_spacing_ratio:0.2f}; phantom: {phantom_min_max_particle_spacing:0.2e}; ViTacTip: {vitactip_min_particle_spacing_for_material:0.2e}"
        particle_min_max_spacing_ratio = max(
            phantom_min_max_particle_spacing / vitactip_min_particle_spacing_for_material,
            phantom_min_max_particle_spacing / vitactip_max_particle_spacing_for_material,
            vitactip_max_particle_spacing_for_material / vitactip_min_particle_spacing_for_material,
        )
        particle_min_max_spacing_ratio = max(particle_min_max_spacing_ratio, 1 / particle_min_max_spacing_ratio)
        assert particle_min_min_spacing_ratio < 10.0, f"ratio of minimum and maximum global particle spacing is too high: {particle_min_max_spacing_ratio:0.2f}"

# Create coordinates dictionary
coordinates = {
    "phantom_closest_vertex": phantom_closest_vertex.tolist(),
    "phantom_centroid_pose": phantom_difftactile_position.tolist() + phantom_orientation,
    "vitactip_tip_position": vitactip_tip_position.tolist(),
    "vitactip_tip_pose": vitactip_tip_position.tolist() + sensor_orientation,
    "phantom_volume": phantom_volume,
    "min_coord": min_coord,
    "max_coord_x": max_coord_x,
    "max_coord_y": max_coord_y,
    "max_coord_z": max_coord_z,
    "object_scale": object_scale,
    "phantom_min_max_particle_spacing": phantom_min_max_particle_spacing,
    "vitactip_min_particle_spacing": vitactip_min_particle_spacing,
    "contact_surface_area": contact_surface_area,
}

# Save coordinates to JSON file
with open('../tasks/system-params-computed.json', 'w') as f:
    json.dump(coordinates, f, indent=2)

print('difftactile coordinates')
print(f'phantom_closest_vertex: {arr_str(phantom_closest_vertex)}')
print(f'phantom centre of mass: {arr_str(phantom_difftactile_position)}')
print(f'sensor dome tip: {arr_str(vitactip_tip_position)}')