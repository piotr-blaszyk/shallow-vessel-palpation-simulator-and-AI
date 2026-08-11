"""
Regenerate `difftactile/output/test_loader_gnn_icra.pickle` without retraining.

That pickle is normally a by-product of `difftactile/cnn/gnn.py::main()`, which writes it
*before* fitting the model (gnn.py:743). Its consumers -- `gnn.evaluate_and_plot_roc()`,
`cnn/visualise.py` and `cnn/threshold_gnn.py` -- only ever read the `dataset` and
`dataset_stats` entries, so the file can be rebuilt from the datasets alone. This script
reproduces exactly the construction gnn.main() performs up to that write, so a checkpoint
trained earlier stays compatible with the stats used to normalise its inputs.

Requires (see README "Reproducibility"):
  - SYSTEM_PARAMS.files.sim_data_endgame  (simulated `_reordered_dense` trajectories)
  - SYSTEM_PARAMS.files.exp_data_endgame  (silicone `_dense` dataset)
  - difftactile/output/base-graph-connectivity.npz

Run from the repository root:
    python -m difftactile.scripts.script_generate_test_loader_icra
"""

import os
import pickle

from difftactile.cnn.gnn import compute_stats
from difftactile.cnn.dataset import MyDataset
from difftactile.main.constants import SYSTEM_PARAMS

# `dataset_stats` is a dict keyed by curriculum difficulty. gnn.main() as it currently stands
# computes only difficulty 1.0 (the `if i / 10 != target_difficulty` guard at gnn.py:710 skips
# the rest), but its consumers disagree about which key to read: gnn.evaluate_and_plot_roc()
# looks up 0.0 (gnn.py:880) and visualise.py looks up other values. The original pickle predates
# that guard and held all 11 levels, so we populate all of them -- it is cheap (a few seconds
# each) and keeps every consumer working.
DIFFICULTIES = [i / 10 for i in range(11)]

# The difficulty the datasets are left set to, matching gnn.main().
TARGET_DIFFICULTY = 1.0


def main():
    batch_size = SYSTEM_PARAMS.gnn.batch_size
    num_workers = SYSTEM_PARAMS.gnn.num_workers

    full_dataset = MyDataset(
        scheme="single_dataset",
        sim_exp="sim",
        data_dir=SYSTEM_PARAMS.files.sim_data_endgame,
        apply_augmentations=True,
    )
    train_dataset, val_dataset, test_dataset = full_dataset.create_splits(
        train_size=0.7, val_size=0.15, test_size=0.15
    )

    # Stats are computed on the *training* split, as in gnn.main().
    all_stats = {}
    for difficulty in DIFFICULTIES:
        train_dataset.set_difficulty_level(difficulty)
        stats = compute_stats(train_dataset, batch_size)
        all_stats[difficulty] = stats
        print(
            f"difficulty: {difficulty}; "
            f"pos:neg = {stats['alpha_neg']:.2f}:{stats['alpha_pos']:.2f}"
        )

    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.set_difficulty_level(TARGET_DIFFICULTY)
        dataset.set_stats(all_stats[TARGET_DIFFICULTY])

    test_data = {
        "dataset": test_dataset,
        "num_workers": num_workers,
        "dataset_stats": all_stats,
    }
    out_path = SYSTEM_PARAMS.files.test_loader_gnn_icra
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(test_data, f)
    print(f"Wrote {out_path} (test split: {len(test_dataset)} clips)")


if __name__ == "__main__":
    main()
