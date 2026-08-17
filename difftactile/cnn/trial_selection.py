"""Choosing WHICH trials a demonstration shows - shared by the prediction viewer
and the bird's-eye vessel maps, so the two show the same trials in the same order.

The project-page (github.io) Sim -> Sim demonstration shows ten held-out
simulated trajectories: seven with the vein under the sweep ("vessel-present")
and three without ("vessel-absent"), interleaved for display as

    a a b a a b a a b a        (a = vessel-present, b = vessel-absent)

Both categories are drawn at random, without replacement, from the SORTED list
of test trajectories with a fixed seed (DIFFTACTILE_VIEW_TRIALS_SEED, default
0). Sorting first matters: the pickled simulated test set is stored in its
training-time shuffled order, and sampling from that order would make the draw
depend on which loader pickle is open. Sorting makes it a function of the trial
NAMES alone, which is what lets `main.py::vessel_map_test_trajectories_main()`
(which re-simulates exactly these trajectories with sensor poses for the maps)
and `visualise.py::Visualisation._match_trials()` (the `interleaved:P:A`
`--trials` token behind the video) agree without sharing any state.

Numpy only - no torch - so it is importable from the simulator side too.
"""

import os
import random

import numpy as np


def trial_has_vessel(path):
    """True if any frame of the trial at `path` is labelled vessel-present.

    `path` is a meat trial directory (`marker_labels.npz` inside) or a
    simulated / silicone trajectory `.npz` (`vein_classification` key).
    """
    if os.path.isdir(path):
        labels = np.load(os.path.join(path, "marker_labels.npz"))["marker_labels"]
        return bool(np.any(labels))
    with np.load(path) as d:
        return bool(np.any(d["vein_classification"]))


def interleave(present, absent):
    """Spread `absent` evenly through `present`: with 7 and 3, `a a b a a b a a b a`.

    Every group of `len(present) // len(absent)` present items is followed by one
    absent item; the remainder of the present items closes the list. With more
    absent than present items the roles are the same and the pattern simply
    starts with the leftover absent items at the end (`a b b a b b ...` is not
    a case this project uses).
    """
    present, absent = list(present), list(absent)
    if not absent:
        return present
    if not present:
        return absent
    group = max(1, len(present) // len(absent))
    out = []
    p = 0
    for b in absent:
        out += present[p:p + group]
        p += group
        out.append(b)
    out += present[p:]
    return out


def select_interleaved(trial_ids, trial_path, n_present, n_absent, seed=None):
    """`n_present` vessel-present and `n_absent` vessel-absent trials, interleaved.

    `trial_ids` are the candidate trial names (any order - they are sorted
    here); `trial_path(trial_id)` gives the file/directory `trial_has_vessel()`
    reads. The draw is `random.Random(seed).sample()` over each sorted category
    in turn (present first), with `seed` defaulting to DIFFTACTILE_VIEW_TRIALS_SEED
    (0). Returns the ordered list of trial ids; raises if a category is short.
    """
    if seed is None:
        seed = int(os.environ.get("DIFFTACTILE_VIEW_TRIALS_SEED", "0"))
    ids = sorted(trial_ids)
    present = [t for t in ids if trial_has_vessel(trial_path(t))]
    absent = [t for t in ids if t not in set(present)]
    if len(present) < n_present or len(absent) < n_absent:
        raise ValueError(
            f"asked for {n_present} vessel-present + {n_absent} vessel-absent trials, "
            f"but only {len(present)} present / {len(absent)} absent are available"
        )
    rng = random.Random(seed)
    chosen_present = rng.sample(present, n_present)
    chosen_absent = rng.sample(absent, n_absent)
    return interleave(chosen_present, chosen_absent)
