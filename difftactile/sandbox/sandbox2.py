import numpy as np

num_time_steps = 10
data_points = np.arange(num_time_steps, dtype=float)

# Differences
diffs = np.diff(data_points)  # shape (num_time_steps - 1,)

# Example kernel to "spread" each diff across two time steps
kernel = np.array([0.5, 0.5])  

# Convolve with 'full' and then trim/pad
reconstructed = np.convolve(diffs, kernel, mode="full")[:num_time_steps]

print("Original shape:", data_points.shape)
print("Diffs shape:", diffs.shape)
print("Reconstructed shape:", reconstructed.shape)