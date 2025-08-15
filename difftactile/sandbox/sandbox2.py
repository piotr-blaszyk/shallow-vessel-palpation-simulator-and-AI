import numpy as np
import time

npz_path = 'difftactile/output/training_data/pickle/trajectory_0000.npz'

# Cold load
start_time = time.perf_counter()
data = np.load(npz_path)
end_time = time.perf_counter()
cold_load_time = end_time - start_time
print(f"Cold load time: {cold_load_time:.4f} seconds")

# Hot load
start_time = time.perf_counter()
data = np.load(npz_path)
end_time = time.perf_counter()
hot_load_time = end_time - start_time
print(f"Hot load time: {hot_load_time:.4f} seconds")

print(f"Hot load was {cold_load_time/hot_load_time:.1f}x faster")
