from bayes_opt import BayesianOptimization, acquisition
import IPython
import json

from difftactile.main.constants import *


class BoGp:
    def __init__(self):
        pbounds = {
            'vitactip_youngs_modulus': (1e-6, 4.8e+05),
            'phantom_youngs_modulus': (1e-6, 4.8e+05),
            'vitactip_poissons_ratio': (0, 0.5),
            'phantom_poissons_ratio': (0, 0.5),
            'normal_stiffness': (0, 1e10),
            'tangential_stiffness': (0, 1e10),
            'normal_damping': (0, 1e10),
            'coulomb_friction_coeff': (0, 1),
        }
        acq = acquisition.UpperConfidenceBound(kappa=2.5)
        self.optimiser = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        self.json_path = SYSTEM_PARAMS.files.bo_gp_json
    
    @staticmethod
    def black_box_function(*args, **kwargs):
        return 0
    
    def foo(self):
        next_point = self.optimiser.suggest()
        target = BoGp.black_box_function(**next_point)
        self.optimiser.register(params=next_point, target=target)

    def my_suggest(self):
        dct = self.optimiser.suggest()
        with open(self.json_path, "w") as f:
            json.dump(dct, f, indent=4)
        return dct

def main():
    b = BoGp()
    IPython.embed()
