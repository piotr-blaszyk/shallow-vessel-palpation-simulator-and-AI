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
NP_RNG = np.random.default_rng()
