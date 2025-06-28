import numpy as np
import math
import json

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
    phantom_params = params['phantom']
    contact_params = params['contact']

# Constants for poses
gap = 0.010
phantom_orientation = [0, 0, 0]  # Roll, pitch, yaw in degrees
sensor_orientation = [-90, 0, 0]  # Roll, pitch, yaw in degrees

space_scale = phantom_params['space_scale']
obj_scale = phantom_params['object_scale']
n_grid = 64
mpm_grid_node_size = space_scale / n_grid
min_coord = mpm_grid_node_size * 3
max_coord = (n_grid - 3) * mpm_grid_node_size
print(f'mpm_grid_node_size: {mpm_grid_node_size}')
print(f'min_coord: {min_coord}')
print(f'max_coord: {max_coord}')

phantom_closest_vertex = np.array([mpm_grid_node_size*6, mpm_grid_node_size*6, mpm_grid_node_size*3], dtype=float)

dist_from_floor = phantom_closest_vertex[2] - min_coord

phantom_h = 0.022
phantom_r = 0.040
phantom_d = phantom_r * 2
phantom_dimensions = np.array([phantom_d, phantom_d, phantom_h], dtype=float)
phantom_volume = math.pi * phantom_r ** 2 * phantom_h

phantom_furthest_vertex = phantom_closest_vertex + phantom_dimensions

print(f'phantom_closest_vertex: {phantom_closest_vertex}')
print(f'phantom_furthest_vertex: {phantom_furthest_vertex}')
print(f'phantom_volume: {phantom_volume}')
print(f'dist_from_floor: {dist_from_floor:0.3e}')

assert np.min(phantom_closest_vertex) > min_coord, "the phantom is outside of the manipulation cube"
assert np.max(phantom_furthest_vertex) < max_coord, "the phantom is outside of the manipulation cube"

phantom_max_dim = np.max(phantom_dimensions)
assert obj_scale == phantom_max_dim
phantom_normalised_spans = phantom_dimensions / phantom_max_dim / 2
phantom_scaled_spans = phantom_normalised_spans * obj_scale

phantom_to_cube_volume_ratio = phantom_volume / (np.max(phantom_scaled_spans) * 2) ** 3

print(f'phantom_normalised_spans: {phantom_normalised_spans}')
print(f'phantom_scaled_spans: {phantom_scaled_spans}')

phantom_difftactile_position = phantom_closest_vertex + phantom_scaled_spans

vitactip_tip_position = np.array([
    phantom_difftactile_position[0],
    phantom_difftactile_position[1],
    phantom_difftactile_position[2] + phantom_dimensions[2] / 2 + gap,
])

# Create coordinates dictionary
coordinates = {
    "phantom_closest_vertex": phantom_closest_vertex.tolist(),
    "phantom_centroid_pose": phantom_difftactile_position.tolist() + phantom_orientation,
    "vitactip_tip_position": vitactip_tip_position.tolist(),
    "vitactip_tip_pose": vitactip_tip_position.tolist() + sensor_orientation,
    "gap": gap,
    "phantom_volume": phantom_volume,
    "min_coord": min_coord,
    "max_coord": max_coord,
    "mpm_grid_node_size": mpm_grid_node_size,
}

# Save coordinates to JSON file
with open('../tasks/initial-coordinates-and-geometry.json', 'w') as f:
    json.dump(coordinates, f, indent=2)

print('difftactile coordinates')
print(f'phantom_closest_vertex: {arr_str(phantom_closest_vertex)}')
print(f'phantom centre of mass: {arr_str(phantom_difftactile_position)}')
print(f'sensor dome tip: {arr_str(vitactip_tip_position)}')