import numpy as np
import pickle

from difftactile.tasks.constants import *

# Load the 2D numpy array from the pickle file
with open(SYSTEM_PARAMS.files.tactile_sensor_all_nodes, 'rb') as f:
    arr = pickle.load(f)

# Ensure it's a numpy array
arr = np.array(arr)

def print_point_cloud(arr):
    # Print the shape for verification
    print('Shape:', arr.shape)

    # Print min and max along axis 1
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    print('Min along axis 0:', min_vals)
    print('Max along axis 0:', max_vals)
