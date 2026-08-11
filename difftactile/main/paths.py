"""Repository-root path resolution.

Everything in this project used to assume the process was launched from the
repository root: `constants.py` opened
`"difftactile/system_params/system-params.json"` as a *relative* path, and a
handful of modules carried absolute paths baked in to the original development
machine. Both break for anyone else (and inside Docker, where the working
directory is not guaranteed).

This module derives the repository root from the location of this file
(`<repo>/difftactile/main/paths.py` -> two parents up) and exposes helpers to
turn any repo-relative path into an absolute one. It can be overridden with the
`DIFFTACTILE_ROOT` environment variable, which is what the Docker image sets.

Usage:
    from difftactile.main.paths import repo_path
    with open(repo_path("difftactile/system_params/system-params.json")) as f:
        ...
"""

import os
from pathlib import Path

# <repo>/difftactile/main/paths.py -> parents[0]=main, [1]=difftactile, [2]=<repo>
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

# DIFFTACTILE_ROOT lets a container / cluster job point at a checkout elsewhere.
REPO_ROOT = Path(os.environ.get("DIFFTACTILE_ROOT", _DEFAULT_ROOT)).resolve()


def repo_path(*parts: str) -> str:
    """Resolve a repo-relative path to an absolute path, as a string.

    Absolute inputs are returned unchanged, so this is safe to wrap around
    values read from the JSON config regardless of how they are written.
    """
    if not parts:
        return str(REPO_ROOT)
    first = str(parts[0])
    if os.path.isabs(first):
        return os.path.join(first, *[str(p) for p in parts[1:]]) if len(parts) > 1 else first
    return str(REPO_ROOT.joinpath(*[str(p) for p in parts]))


def data_path(*parts: str) -> str:
    """Resolve a path inside the external data bundle.

    Large artifacts (the simulated dataset, real-sensor trials, trained
    checkpoints) are distributed separately via Zenodo rather than committed.
    `restore_data.sh` unpacks them into the repository, so by default this is
    just the repository root; `DIFFTACTILE_DATA_ROOT` allows keeping the bundle
    on another disk and pointing at it instead.
    """
    root = os.environ.get("DIFFTACTILE_DATA_ROOT")
    if root:
        return os.path.join(os.path.abspath(root), *[str(p) for p in parts])
    return repo_path(*parts)
