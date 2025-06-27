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
gap = 0.01
phantom_orientation = [0, 0, 0]  # Roll, pitch, yaw in degrees
sensor_orientation = [-90, 0, 0]  # Roll, pitch, yaw in degrees

space_scale = phantom_params['space_scale']
obj_scale = phantom_params['object_scale']
n_grid = 64
dx_0 = space_scale / n_grid
min_z = dx_0 * 3
max_z = (n_grid - 3) * dx_0
print(f'dx_0: {dx_0}')
print(f'min_z: {min_z}')
print(f'max_z: {max_z}')

phantom_closest_vertex = np.array([min_z*3, min_z*3, min_z*3], dtype=float)

phantom_h = 0.022
phantom_r = 0.040
phantom_d = phantom_r * 2
phantom_dimensions = np.array([phantom_d, phantom_d, phantom_h], dtype=float)

phantom_max_dim = np.max(phantom_dimensions)
assert obj_scale == phantom_max_dim
phantom_normalised_spans = phantom_dimensions / phantom_max_dim / 2
phantom_scaled_spans = phantom_normalised_spans * obj_scale

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
    "gap": gap
}

# Save coordinates to JSON file
with open('../tasks/initial-coordinates.json', 'w') as f:
    json.dump(coordinates, f, indent=2)

print('difftactile coordinates')
print(f'phantom_closest_vertex: {arr_str(phantom_closest_vertex)}')
print(f'phantom centre of mass: {arr_str(phantom_difftactile_position)}')
print(f'sensor dome tip: {arr_str(vitactip_tip_position)}')