from difftactile.main.constants_common import *


# Create a global instance with the system parameters
SYSTEM_PARAMS = ConstantsFromJson("difftactile/system_params/system-params.json")
SYSTEM_PARAMS_COMPUTED = ConstantsFromJson("difftactile/system_params/system-params-computed.json")
NP_RNG = np.random.default_rng()
