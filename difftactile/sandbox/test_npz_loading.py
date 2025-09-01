import gc
import os
import time

import numpy as np


def measure_loading_time(path, num_iterations=5):
    """Measure the loading time of an NPZ file."""
    times = []
    
    for i in range(num_iterations):
        # Clear any cached data and force garbage collection
        gc.collect()
        
        # Measure loading time
        start_time = time.perf_counter()
        data = np.load(path)
        end_time = time.perf_counter()
        
        loading_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(loading_time)
        
        # Close the file to ensure it's not kept in memory
        data.close()
        
    return times

def main():
    # Use the path from sandbox2.py
    path = 'difftactile/output/training_data/pickle/trajectory_0001.npz'
    
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return
    
    print(f"Testing NPZ file loading times for: {path}")
    print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
    print("\nMeasuring loading times...")
    
    # First load (cold)
    times = measure_loading_time(path, num_iterations=1)
    print(f"\nCold load time: {times[0]:.2f} ms")
    
    # Subsequent loads (hot)
    times = measure_loading_time(path, num_iterations=5)
    avg_hot_time = sum(times) / len(times)
    print(f"Hot load times (average of 5 runs): {avg_hot_time:.2f} ms")
    print(f"Hot load times (individual runs): {[f'{t:.2f}' for t in times]} ms")

if __name__ == "__main__":
    main() 