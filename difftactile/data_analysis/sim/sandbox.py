import time
import numpy as np

path1 = "difftactile/output/training_data/pickle/trajectory_0000.npz"
path2 = "difftactile/output/training_data/pickle/trajectory_0001.npz"
path3 = "difftactile/output/training_data/pickle/trajectory_0002.npz"

# Cold read (after flushing cache)
start = time.perf_counter()
data = np.load(path1)
cold_time = time.perf_counter() - start
print(f"Cold read: {cold_time:.6f} s")

data = np.load(path2)
data = np.load(path3)

# Hot read (file should now be in cache)
start = time.perf_counter()
data = np.load(path1)
hot_time = time.perf_counter() - start
print(f"Hot read: {hot_time:.6f} s")
