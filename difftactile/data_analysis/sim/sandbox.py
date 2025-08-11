import numpy as np

valid_frames_mask = np.array([
    False,
    False,
    False
])
        
# Find runs of True values
# Add sentinel values to handle edge cases
padded = np.concatenate(([False], valid_frames_mask, [False]))
runs = np.where(np.diff(padded))[0]

# runs[::2] are starts of True runs, runs[1::2] are ends of True runs
# Length of each run is end - start
run_lengths = runs[1::2] - runs[::2]

if len(run_lengths) > 0:
    longest_run_idx = np.argmax(run_lengths)
    start_idx = runs[::2][longest_run_idx]
    end_idx = runs[1::2][longest_run_idx]
else:
    start_idx = 0
    end_idx = 0

foo = 7
