#!/usr/bin/env python3
"""Where the analysis scripts read from and write to.

Every path is derived from this file's own location, so the whole `analysis/`
tree moves with the repository and needs no editing after a clone:

    REPO      the repository root      (this file is <REPO>/analysis/scripts/paths.py)
    ANALYSIS  <REPO>/analysis          the tree these scripts own
    RESULTS   <REPO>/analysis/results  machine-readable metrics (.json, .npz)
    FIGURES   <REPO>/analysis/figures  rendered figures (.pdf, .png, .tex fragments)

`vessel_map_run(config)` resolves the published top-view-map run of one
configuration. Two layouts are accepted, in this order:

1. `difftactile/output/vessel_maps/<cfg>/<timestamp>/` — what `docker/vessel_map_all.sh`
   writes locally. The newest timestamp of each configuration is the published run
   (runs are never overwritten), which is exactly the rule `data/make_data_bundle.sh`
   uses when it stages them, so "newest" and "published" cannot drift apart.
2. `difftactile/output/manuscript_artifacts/vessel_maps/<cfg>/` — the copy the Zenodo
   bundle restores. It has no timestamp level, because the bundle ships one run per
   configuration.

So a fresh clone plus `./data/restore_data.sh` reproduces every number below without
re-running the maps, and a repository that has run them locally uses its own.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"
RESULTS = ANALYSIS / "results"
FIGURES = ANALYSIS / "figures"

OUTPUT = REPO / "difftactile" / "output"
MAPS = OUTPUT / "vessel_maps"
BUNDLE_MAPS = OUTPUT / "manuscript_artifacts" / "vessel_maps"

# Directory name of each configuration's published map run, without the timestamp.
# Ground-truth source per configuration: the simulator for Sim->Sim, the annotated
# video for the two real test sets.
MAP_DIRS = {"A-to-A": "sim-to-sim_gt-simulator",
            "A-to-B": "sim-to-silicone_gt-video",
            "A-to-C": "sim-to-meat_gt-video",
            "C-to-B": "meat-to-silicone_gt-video"}

CONFIGS = ("A-to-A", "A-to-B", "A-to-C", "C-to-B")
PRETTY = {"A-to-A": "Sim->Sim", "A-to-B": "Sim->Silicone",
          "A-to-C": "Sim->Meat", "C-to-B": "Meat->Silicone"}


def map_run_dir(cfg):
    """Path of the published run of one vessel-map directory, by its directory name.

    `cfg` is a directory of difftactile/output/vessel_maps/, e.g.
    "sim-to-silicone_gt-video" or "sim-to-sim-test-trajectories_gt-simulator".
    See the module docstring for the two layouts this accepts and why.
    """
    local = MAPS / cfg
    if local.is_dir():
        # Newest timestamp wins; `-legacy` runs are a different checkpoint and never published.
        runs = sorted(p for p in local.iterdir()
                      if p.is_dir() and not p.name.endswith("-legacy"))
        if runs:
            return runs[-1]
    bundled = BUNDLE_MAPS / cfg
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        f"no top-view map run for {cfg}: looked in {local} and {bundled}. "
        "Restore the Zenodo bundle (./data/restore_data.sh) or run ./docker/vessel_map_all.sh.")


def vessel_map_run(config):
    """Path of the published top-view-map run of one A-to-B style configuration."""
    return map_run_dir(MAP_DIRS[config])


def ensure_dirs():
    """Create the output trees; safe to call from every entry point."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
