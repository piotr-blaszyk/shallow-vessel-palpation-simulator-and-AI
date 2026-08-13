"""One place to make a training run reproducible.

Training was previously stochastic in four independent ways, so two runs of the
same configuration gave different numbers and neither could be reproduced:

  1. **Model initialisation** - torch's global RNG, unseeded.
  2. **Batch order** - `SubsetRandomSampler`. This one was already seeded
     (`GNNDataModule(seed=42)` feeds a `torch.Generator`), and is the reason the
     variation looked smaller than it was.
  3. **Data augmentation and subset selection** - `NP_RNG` in
     `difftactile/main/constants.py`, which was `np.random.default_rng()` with
     no seed. This drives the augmentation rotations in `dataset.py`, the
     dataset shuffles, and the per-epoch train/val subset choice in
     `_select_new_subset()`.
  4. **DataLoader workers** - with `num_workers_large=16` the augmentation runs
     in forked worker processes, each of which inherits a *copy* of `NP_RNG`.
     Seeding the parent alone would make every worker draw the identical
     sequence, which is worse than useless: the augmentation would repeat
     16-fold across a batch. `seed_worker()` below gives each worker a distinct
     but *derived* stream, so runs stay reproducible without collapsing the
     augmentation diversity.

`seed_everything()` covers all four. Call it once at the top of a training
entrypoint, before datasets or models are constructed - the model's weights are
drawn at construction time, so seeding afterwards would not reach them.

THE SEED IS NOT A HYPERPARAMETER. Do not tune it. A result that depends on which
seed was used is not a result, and the honest way to report seed sensitivity is
to run several and give the spread - see `DIFFTACTILE_SEED` below.
"""

import os
import random

import numpy as np


# Default seed for every stochastic component. 42 matches the value
# `GNNDataModule` already used for its sampler, so the batch order of a default
# run is unchanged by the introduction of central seeding.
DEFAULT_SEED = 42

# Environment override, so a seed sweep needs no source edit:
#
#     for s in 0 1 2 3 4; do DIFFTACTILE_SEED=$s ./run_pipeline.sh C-to-B; done
#
# THIS MATTERS MORE HERE THAN IT USUALLY DOES. Measured on C-to-B, seven seeds
# gave AUROC from 0.604 to 0.779 (AP 0.239 to 0.342) - a spread far larger than
# any difference the paper's configurations claim between one another. The meat
# training set is small (139 clips), so which subset the run happens to favour
# dominates. Consequences:
#
#   * A single-seed number is not a result. Report mean +/- spread over several
#     seeds, or the whole set.
#   * Never pick the seed that scores best. That is fitting the seed to the test
#     set, and it is why this constant is not a tunable hyperparameter.
#   * When comparing two configurations, compare distributions over seeds, not
#     one run each - a 0.05 gap between single runs here means nothing.
SEED_ENV_VAR = "DIFFTACTILE_SEED"


def resolve_seed(seed=None):
    """The seed to use: explicit argument, else $DIFFTACTILE_SEED, else default."""
    if seed is not None:
        return int(seed)
    return int(os.environ.get(SEED_ENV_VAR, DEFAULT_SEED))


def _reseed_shared_rng(seed):
    """Reseed the project's shared `NP_RNG` **in place**.

    Deliberately mutates the existing Generator's bit-generator state rather
    than rebinding `constants.NP_RNG` to a fresh object. `dataset.py` and
    friends do `from difftactile.main.constants import *`, which binds NP_RNG
    into each module's own namespace at import time - so rebinding the name in
    `constants` would leave every one of those modules still holding, and
    drawing from, the original unseeded generator. Mutating the shared object
    reaches all of them, whatever name they hold it under.
    """
    from difftactile.main import constants
    constants.NP_RNG.bit_generator.state = (
        np.random.default_rng(seed).bit_generator.state
    )


def seed_everything(seed=None, deterministic_torch=True):
    """Seed every RNG a training run touches. Returns the seed actually used.

    Covers Python's `random`, numpy's legacy global RNG, the project's `NP_RNG`,
    and torch (CPU and all CUDA devices). Safe to call when torch is absent -
    the import is done here rather than at module scope so that the annotator's
    small environment can still import this module.

    `deterministic_torch` asks torch for deterministic kernels, which is what
    actually makes two runs agree - seeding alone is NOT enough for this model.
    Seeding fixes the inputs (weights, batch order, augmentation); it does
    nothing about the GPU summing them in a different order each time. This
    network is built from `GINEConv` and `global_add_pool`, whose scatter
    reductions use CUDA atomics, and atomic float addition is not associative,
    so the same batch can produce slightly different gradients run to run. Those
    differences compound over training into visibly different final metrics -
    measured here as AUROC drifting by ~0.005 between two seeded runs.

    Three switches are needed, and all three are set together:
      * `cudnn.deterministic` / `benchmark=False` - deterministic convolution
        algorithms, and no autotuning that picks a different one per run.
      * `use_deterministic_algorithms` - makes torch use deterministic scatter
        variants, and *raise* rather than silently proceed if an op has no
        deterministic implementation.
      * `CUBLAS_WORKSPACE_CONFIG` - required by cuBLAS for reproducible GEMMs on
        CUDA >= 10.2; without it the call above raises at the first matmul.

    It costs some speed - deterministic kernels are slower than the autotuned
    ones, and the ban on atomics can hurt scatter-heavy models like this one.
    Pass False to trade reproducibility back for that speed.

    Note what this does NOT promise: identical results across different GPU
    models, CUDA versions, or worker counts. Floating-point reduction order
    varies with hardware, so cross-machine agreement is not on offer here -
    same machine, same code, same seed is.
    """
    seed = resolve_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    _reseed_shared_rng(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Must be set before the first cuBLAS handle is created, i.e. before any
        # CUDA matmul runs - which is why seed_everything() belongs at the very
        # top of an entrypoint, before models and datasets are built.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: some scatter paths in torch-geometric have no deterministic
        # CUDA implementation. Raising there would make training impossible
        # rather than reproducible, so those warn and stay nondeterministic -
        # the run is then "as deterministic as this model allows", and the
        # warning names exactly which op is responsible.
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed


def seed_worker(worker_id):
    """`worker_init_fn` for a DataLoader: give each worker its own stream.

    Forked workers inherit a copy of the parent's RNG state, so without this
    every worker would draw the SAME augmentation sequence - identical rotations
    repeated across a batch, which quietly destroys augmentation diversity while
    looking fine.

    torch derives a distinct `initial_seed()` per worker per epoch from the base
    seed, so taking it as the source keeps the streams both distinct and
    reproducible.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    _reseed_shared_rng(worker_seed)
