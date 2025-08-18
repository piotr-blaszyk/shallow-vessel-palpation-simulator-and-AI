import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors
import pickle
import cv2
from tqdm import tqdm
import time

from difftactile.cnn.dataset import *
from difftactile.cnn.gnn import *
from difftactile.data_analysis.experiment.adjacency import Adjacency

def main():
    dataset = MyDataset(
        data_dir=SYSTEM_PARAMS.files.dataset_root_test
    )

    start_time = time.perf_counter()
    for i in tqdm(range(100), desc="Benchmarking dataset"):
        foo = dataset[i]
    end_time = time.perf_counter()
    print(f"Time taken to iterate over the dataset: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()

