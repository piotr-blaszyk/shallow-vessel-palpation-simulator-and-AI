import numpy as np

def create_flat_indices_array(shape):
    i, j, k = np.meshgrid(np.arange(shape[0]), 
                         np.arange(shape[1]), 
                         np.arange(shape[2]), 
                         indexing='ij')
    flat_ixs = np.ravel_multi_index((i, j, k), shape)
    return flat_ixs

# Example usage
shape = (4, 3, 2)  # Example dimensions
flat_ixs = create_flat_indices_array(shape)

# Verify
print(f"Shape of flat_ixs: {flat_ixs.shape}")
print(f"Value at (2,1,1): {flat_ixs[2,1,1]}")
print(f"Should equal: {np.ravel_multi_index((2,1,1), shape)}")