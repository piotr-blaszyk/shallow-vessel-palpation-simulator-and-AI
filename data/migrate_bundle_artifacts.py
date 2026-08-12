#!/usr/bin/env python3
"""Rewrite pre-rename names inside the published binary artifacts.

Handles two kinds of file; the argument's extension selects which.

`.pt` checkpoints
-----------------
A torch checkpoint is a zip archive whose internal entries are named after the
file's name at save time, so a checkpoint saved as
`final_segmentation_model_gnn_iros.pt` carries that string as its internal
directory prefix regardless of what the file is later renamed to. Re-saving the
state dict under the current filename rewrites those entries. The tensors are
untouched -- verified bit-for-bit identical before and after.

`.pickle` test loaders
----------------------

The published test-loader pickles were serialised before the naming cleanup, so
they still carry the old conference-derived names *inside* the file:

  * the top-level flag key            'iros'          -> 'meat'
  * the pickled MyDataset attributes  iros_data       -> meat_data
                                      dilation_iros   -> dilation_meat
  * the dataset's scheme string       'iros'          -> 'meat'

`pickle.load` restores `__dict__` verbatim without calling `__init__`, so an
un-migrated object keeps the old attribute names and its `scheme` no longer
matches the renamed `"meat"` branch -- `len(dataset)` silently returns 0 instead
of the true clip count. The three paper configurations never touch the embedded
dataset (they read only `dataset_stats`, which is why their metrics are
unaffected), but `cnn/visualise.py` does, and the stale names would otherwise
ship inside a published artifact.

Usage:
    python data/migrate_bundle_artifacts.py FILE [FILE ...]

Rewrites each file in place. Safe to re-run: already-migrated files are left
untouched and reported as such.
"""
import os
import pickle
import re
import sys

# Any of these appearing inside a shipped artifact is what this script removes.
STALE = re.compile(r"iros|icra|endgame", re.I)

# Old name -> new name, for both dict keys and instance attributes.
ATTR_RENAMES = {
    "iros_data": "meat_data",
    "dilation_iros": "dilation_meat",
}
FLAG_RENAMES = {"iros": "meat"}
SCHEME_RENAMES = {"iros": "meat"}


def migrate(path):
    """Rewrite one pickle in place. Returns a list of changes made."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    changes = []

    if isinstance(data, dict):
        for old, new in FLAG_RENAMES.items():
            if old in data:
                data[new] = data.pop(old)
                changes.append(f"key {old!r} -> {new!r}")

    dataset = data.get("dataset") if isinstance(data, dict) else None
    if dataset is not None:
        for old, new in ATTR_RENAMES.items():
            if hasattr(dataset, old):
                setattr(dataset, new, getattr(dataset, old))
                delattr(dataset, old)
                changes.append(f"attr {old} -> {new}")
        scheme = getattr(dataset, "scheme", None)
        if scheme in SCHEME_RENAMES:
            dataset.scheme = SCHEME_RENAMES[scheme]
            changes.append(f"scheme {scheme!r} -> {dataset.scheme!r}")

    if changes:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    return changes


def migrate_checkpoint(path):
    """Re-save a .pt so its internal zip entries use the current filename.

    Only rewrites when the file actually contains a stale name, so re-runs are
    no-ops. Weights are round-tripped unchanged.
    """
    with open(path, "rb") as f:
        if not STALE.search(f.read().decode("latin-1")):
            return []

    import torch  # imported lazily: only .pt inputs need it

    state = torch.load(path, map_location="cpu", weights_only=False)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return [f"re-saved to clear stale internal entry names"]


def main(argv):
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    for path in argv:
        if path.endswith(".pt"):
            changes = migrate_checkpoint(path)
        else:
            changes = migrate(path)
        if changes:
            print(f"migrated {path}")
            for c in changes:
                print(f"    {c}")
        else:
            print(f"already clean: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
