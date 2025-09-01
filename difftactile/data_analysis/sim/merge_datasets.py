import os
import re
import shutil


def extract_trajectory_number(file_path):
    """Extract the trajectory number from the file path.
    Args:
        file_path (str): Path like 'path/to/trajectory_0001.npz'
    Returns:
        int: The trajectory number (e.g., 1 for 'trajectory_0001.npz')
    """
    match = re.search(r"trajectory_(\d+)\.npz$", file_path)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract trajectory number from {file_path}")

def merge_datasets():
    # Define paths
    base_dir = os.path.join("difftactile", "output", "training_data")
    source_dir_1 = os.path.join(base_dir, "pickle_2025_08_24")
    source_dir_2 = os.path.join(base_dir, "pickle_2025_08_25")
    target_dir = os.path.join(base_dir, "pickle_2025_08_24-25_merged")

    # Check if source directories exist
    if not os.path.exists(source_dir_1):
        raise ValueError(f"Source directory {source_dir_1} does not exist")
    if not os.path.exists(source_dir_2):
        raise ValueError(f"Source directory {source_dir_2} does not exist")

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Copy files from first directory as-is
    for filename in os.listdir(source_dir_1):
        if filename.endswith('.npz'):
            src_path = os.path.join(source_dir_1, filename)
            dst_path = os.path.join(target_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename} from {source_dir_1}")

    # Copy and rename files from second directory
    for filename in os.listdir(source_dir_2):
        if filename.endswith('.npz'):
            src_path = os.path.join(source_dir_2, filename)
            traj_num = extract_trajectory_number(filename)
            new_traj_num = traj_num + 256
            new_filename = f"trajectory_{new_traj_num:04d}.npz"
            dst_path = os.path.join(target_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied and renamed {filename} to {new_filename}")

if __name__ == "__main__":
    merge_datasets()