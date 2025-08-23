# sitecustomize.py
import numpy as np
import random as _random

# ------------------------
# One seed for everything
# ------------------------
MASTER_SEED = 12345  # <- change this for reproducibility, or None for system entropy

# ------------------------
# NumPy random setup
# ------------------------
_rng = np.random.default_rng(MASTER_SEED)

np.random.seed = lambda *a, **k: None  # disable reseeding
np.random.rand = _rng.random
np.random.randn = _rng.standard_normal
np.random.randint = _rng.integers
np.random.uniform = _rng.uniform
np.random.normal = _rng.normal
np.random.choice = _rng.choice
np.random.shuffle = _rng.shuffle
np.random.permutation = _rng.permutation

# ------------------------
# Python's random setup
# ------------------------
_random_gen = _random.Random(MASTER_SEED)  # same seed as NumPy

_random.seed = lambda *a, **k: None   # disable reseeding
_random.random = _random_gen.random
_random.uniform = _random_gen.uniform
_random.randint = _random_gen.randint
_random.randrange = _random_gen.randrange
_random.choice = _random_gen.choice
_random.shuffle = _random_gen.shuffle
_random.sample = _random_gen.sample
_random.betavariate = _random_gen.betavariate
_random.gammavariate = _random_gen.gammavariate
_random.gauss = _random_gen.gauss
_random.lognormvariate = _random_gen.lognormvariate
_random.normalvariate = _random_gen.normalvariate
_random.vonmisesvariate = _random_gen.vonmisesvariate
_random.paretovariate = _random_gen.paretovariate
_random.weibullvariate = _random_gen.weibullvariate
