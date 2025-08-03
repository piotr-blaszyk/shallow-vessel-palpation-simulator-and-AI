import os
import shutil
import random
from pathlib import Path
import re
from typing import List, Tuple

from difftactile.main.constants import *

def get_trajectory_numbers(source_dir: Path) -> List[int]:
    """Get all trajectory numbers from the source directory."""
    trajectory_pattern = re.compile(r'trajectory_(\d{4})')
    trajectory_numbers = set()
    
    for item in source_dir.iterdir():
        if item.is_dir():
            match = trajectory_pattern.match(item.name)
            if match:
                trajectory_numbers.add(int(match.group(1)))
    
    return sorted(list(trajectory_numbers))

def create_split_directories(root_dir: Path) -> None:
    """Create the directory structure for splits."""
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            (root_dir / 'splits' / split / subdir).mkdir(parents=True, exist_ok=True)

def split_trajectories(trajectory_numbers: List[int], train_ratio=0.7, val_ratio=0.15) -> Tuple[List[int], List[int], List[int]]:
    """Split trajectory numbers into train, validation and test sets."""
    random.shuffle(trajectory_numbers)
    
    n_trajectories = len(trajectory_numbers)
    n_train = int(n_trajectories * train_ratio)
    n_val = int(n_trajectories * val_ratio)
    
    train_trajectories = trajectory_numbers[:n_train]
    val_trajectories = trajectory_numbers[n_train:n_train + n_val]
    test_trajectories = trajectory_numbers[n_train + n_val:]
    
    return train_trajectories, val_trajectories, test_trajectories

def copy_trajectory_data(root_dir: Path, trajectory_num: int, split: str) -> None:
    """Copy a single trajectory's data to the new split location."""
    for data_type in ['images', 'labels']:
        src_dir = root_dir / 'pickle' / data_type / f'trajectory_{trajectory_num:04d}'
        dst_dir = root_dir / 'splits' / split / data_type / f'trajectory_{trajectory_num:04d}'
        
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Setup paths
    root_dir = Path(SYSTEM_PARAMS.files.dataset_root)  # Adjust this path as needed
    
    # Create new directory structure
    create_split_directories(root_dir)
    
    # Get all trajectory numbers from images directory
    trajectory_numbers = get_trajectory_numbers(root_dir / 'pickle' / 'images')
    
    # Split trajectories
    train_trajectories, val_trajectories, test_trajectories = split_trajectories(trajectory_numbers)
    
    # Copy data for each split
    for traj_num in train_trajectories:
        copy_trajectory_data(root_dir, traj_num, 'train')
    
    for traj_num in val_trajectories:
        copy_trajectory_data(root_dir, traj_num, 'val')
    
    for traj_num in test_trajectories:
        copy_trajectory_data(root_dir, traj_num, 'test')
    
    # Print split statistics
    print(f"Dataset split complete:")
    print(f"Train trajectories: {len(train_trajectories)}")
    print(f"Validation trajectories: {len(val_trajectories)}")
    print(f"Test trajectories: {len(test_trajectories)}")

if __name__ == '__main__':
    main()
