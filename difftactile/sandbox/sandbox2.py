import numpy as np
from tqdm import tqdm

for i in tqdm(range(1_000_000_000)):
    res = np.random.uniform(0, 1)
