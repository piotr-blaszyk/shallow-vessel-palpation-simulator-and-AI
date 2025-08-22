import numpy as np

# Example array of shape (n, k)
# Let's create a sample array for demonstration
n, k = 4, 3
array = np.array([
    [1, 5, 3],
    [4, 2, 6],
    [7, 1, 4],
    [2, 3, 5]
])  # Shape: (4, 3)

# Find minimum values along axis 0 (across rows)
min_values = np.min(array, axis=0)  # Shape: (3,)

print("Original array shape:", array.shape)
print("Original array:\n", array)
print("\nMinimum values shape:", min_values.shape)
print("Minimum values:", min_values) 