import json

import IPython
from bayes_opt import BayesianOptimization, acquisition

from difftactile.main.constants import *


class BoGp:
    def __init__(self):
        self.pbounds = {
            'vitactip_youngs_modulus': (1e4, 4.8e+05),
            # 'phantom_youngs_modulus': (1e4, 4.8e+05),
            'vitactip_poissons_ratio': (0.3, 0.5),
            # 'phantom_poissons_ratio': (0.3, 0.5),
            'normal_stiffness': (0, 5e1),
            'tangential_stiffness': (0, 5e1),
            'normal_damping': (0, 5e1),
            'coulomb_friction_coeff': (0, 1),
        }
        self.pbounds_normalised = {
            'vitactip_youngs_modulus': (0, 1),
            # 'phantom_youngs_modulus': (0, 1),
            'vitactip_poissons_ratio': (0, 1),
            # 'phantom_poissons_ratio': (0, 1),
            'normal_stiffness': (0, 1),
            'tangential_stiffness': (0, 1),
            'normal_damping': (0, 1),
            'coulomb_friction_coeff': (0, 1),
        }
        # acq = acquisition.UpperConfidenceBound(kappa=2.5)
        acq = acquisition.ExpectedImprovement(xi=0.01)
        self.optimiser = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=self.pbounds_normalised,
            verbose=2,
            random_state=1,
        )
        self.params_path = SYSTEM_PARAMS.files.bo_gp_json
        self.target_path = SYSTEM_PARAMS.files.bo_gp_target_json
        self.all_params_path = SYSTEM_PARAMS.files.bo_all_params
        self.all_targets_path = SYSTEM_PARAMS.files.bo_all_targets
        self.all_params = []
        self.all_targets = []
        self.target_min_max = (0, 700)
    
    def normalise_dict(self, input_dict):
        normalized = {}
        for key, value in input_dict.items():
            if key not in self.pbounds:
                raise KeyError(f"Parameter {key} not found in pbounds")
            min_val, max_val = self.pbounds[key]
            normalized[key] = (value - min_val) / (max_val - min_val)
        return normalized

    def unnormalise_dict(self, normalized_dict):
        unnormalized = {}
        for key, value in normalized_dict.items():
            if key not in self.pbounds:
                raise KeyError(f"Parameter {key} not found in pbounds")
            min_val, max_val = self.pbounds[key]
            unnormalized[key] = value * (max_val - min_val) + min_val
        return unnormalized
    
    def normalise_target(self, target):
        min_val, max_val = self.target_min_max
        return (target - min_val) / (max_val - min_val)

    def unnormalise_target(self, normalized_target):
        min_val, max_val = self.target_min_max
        return normalized_target * (max_val - min_val) + min_val
    
    @staticmethod
    def black_box_function(*args, **kwargs):
        return 0
    
    def foo(self):
        next_point = self.optimiser.suggest()
        target = BoGp.black_box_function(**next_point)
        self.optimiser.register(params=next_point, target=target)

    def my_suggest_optimise(self):
        params = self.optimiser.suggest()
        self.my_suggest_optimise_helper(params)
    
    def my_suggest_random(self):
        params = {k: np.random.uniform(*v) for k, v in self.pbounds_normalised.items()}
        self.my_suggest_optimise_helper(params)
    
    def my_suggest_optimise_helper(self, params):
        params['tangential_stiffness'] = NP_RNG.uniform(0, 0.3) * params['normal_stiffness']
        params = self.unnormalise_dict(params)
        print(params)
        self.params = params
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=4)
    
    def my_register(self, target):
        # with open(self.target_path, "r") as f:
        #     target_data = json.load(f)
        # self.target = target_data['target']
        self.optimiser.register(
            params=self.normalise_dict(self.params),
            target=1-self.normalise_target(target),
        )
        self.all_params.append(self.params)
        self.all_targets.append(target)
    
    def write_to_file(self):
        # print("writing to file!")
        with open(self.all_params_path, "w") as f:
            json.dump(self.all_params, f, indent=4)
        with open(self.all_targets_path, "w") as f:
            json.dump(self.all_targets, f, indent=4)

def main():
    return
    b = BoGp()
    IPython.embed()
    b.write_to_file()
