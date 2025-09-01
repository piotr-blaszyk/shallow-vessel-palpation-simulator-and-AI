import time

from tqdm import tqdm

from difftactile.cnn.dataset import *
from difftactile.cnn.gnn import *


def main():
    dataset = MyDataset(
        data_dir=SYSTEM_PARAMS.files.dataset_root_reordered
    )

    start_time = time.perf_counter()
    for i in tqdm(range(100), desc="Benchmarking dataset"):
        foo = dataset[i]
    end_time = time.perf_counter()
    print(f"Time taken to iterate over the dataset: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()

