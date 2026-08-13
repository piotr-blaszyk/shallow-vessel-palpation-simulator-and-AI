import os

import numpy as np

from difftactile.main.constants_common import *
from difftactile.main.paths import REPO_ROOT, repo_path

# Create a global instance with the system parameters.
# Paths are resolved against the repository root (see difftactile/main/paths.py)
# rather than the current working directory, so scripts run from anywhere.
SYSTEM_PARAMS = ConstantsFromJson(repo_path("difftactile/system_params/system-params.json"))
SYSTEM_PARAMS_COMPUTED = ConstantsFromJson(
    repo_path("difftactile/system_params/system-params-computed.json")
)


# Config keys whose values are bare FILENAMES, not paths: callers join them onto
# a separate directory key (e.g. `f"{files.da_dir}{files.da_press}"`, or
# `.format()`-ed into a dataset directory). Absolutising these would produce
# "<dir>/<repo>/<name>" and break the lookup, so they are skipped.
_BARE_FILENAME_KEYS = {
    "da_press",
    "da_twist_z",
    "da_twist_x",
    "da_slide",
    "flat_sensor_default_state",
    "dataset_data_point",
}


def _absolutise_file_paths(files):
    """Rewrite every value in the `files` config section to an absolute path.

    The JSON stores these repo-relative (e.g. "difftactile/output/foo.npz"), and
    roughly 250 call sites pass them straight to open()/np.load(). Resolving
    them centrally, once, is what actually makes the project runnable from any
    working directory — wrapping each call site individually would be endless
    and easy to miss.

    Values containing `{}`/`{:04d}` format placeholders are handled fine:
    only the leading directory part is rewritten, and .format() is applied by
    the caller afterwards as before. Non-string values (e.g. `traj_id`, an int)
    are left untouched.
    """
    for key, value in vars(files).items():
        if key in _BARE_FILENAME_KEYS:
            continue
        if isinstance(value, str) and not os.path.isabs(value):
            absolute = repo_path(value)
            # Preserve a trailing separator: some directory values are joined to
            # a filename by plain string concatenation (`f"{da_dir}{da_press}"`),
            # and repo_path() normalises the slash away.
            if value.endswith("/") and not absolute.endswith("/"):
                absolute += "/"
            setattr(files, key, absolute)


_absolutise_file_paths(SYSTEM_PARAMS.files)


# Taichi allocates a field's .grad buffers at construction time, from
# `meta.enable_grad`, so whether gradients exist is fixed before any simulation
# runs and cannot be toggled later. The shipped value is 0: training-data
# collection never differentiates, and the gradient buffers roughly double the
# GPU memory the simulator needs.
#
# Domain adaptation DOES differentiate - it optimises contact parameters against
# real marker positions - so it needs them. Rather than flipping the shared JSON
# (which would make every `script_main` collection run pay for buffers it never
# reads), DIFFTACTILE_ENABLE_GRAD=1 turns them on for the one process that
# wants them. `domain_adaptation.sh` sets it; nothing else does.
_enable_grad_override = os.environ.get("DIFFTACTILE_ENABLE_GRAD")
if _enable_grad_override is not None:
    SYSTEM_PARAMS.meta.enable_grad = int(_enable_grad_override)

NP_RNG = np.random.default_rng()
