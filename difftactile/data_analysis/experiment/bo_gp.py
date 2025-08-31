from bayes_opt import BayesianOptimization, acquisition
import IPython
import json

from difftactile.main.constants import *


class BoGp:
    def __init__(self):
        self.pbounds = {
            'vitactip_youngs_modulus': (1e4, 4.8e+05),
            'phantom_youngs_modulus': (1e4, 4.8e+05),
            'vitactip_poissons_ratio': (0.3, 0.5),
            'phantom_poissons_ratio': (0.3, 0.5),
            'normal_stiffness': (0, 1e10),
            'tangential_stiffness': (0, 1e10),
            'normal_damping': (0, 1e10),
            'coulomb_friction_coeff': (0, 1),
        }
        # acq = acquisition.UpperConfidenceBound(kappa=2.5)
        acq = acquisition.ExpectedImprovement(xi=0.01)
        self.optimiser = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=self.pbounds,
            verbose=2,
            random_state=1,
        )
        self.params_path = SYSTEM_PARAMS.files.bo_gp_json
        self.target_path = SYSTEM_PARAMS.files.bo_gp_target_json
        self.all_params_path = SYSTEM_PARAMS.files.bo_all_params
        self.all_targets_path = SYSTEM_PARAMS.files.bo_all_targets
        self.all_params = []
        self.all_targets = []
    
    @staticmethod
    def black_box_function(*args, **kwargs):
        return 0
    
    def foo(self):
        next_point = self.optimiser.suggest()
        target = BoGp.black_box_function(**next_point)
        self.optimiser.register(params=next_point, target=target)

    def my_suggest_optimise(self):
        params = self.optimiser.suggest()
        self.save_params_to_file(params)
        return params
    
    def my_suggest_random(self):
        params = {k: np.random.uniform(*v) for k, v in self.pbounds.items()}
        self.save_params_to_file(params)
        return params
    
    def save_params_to_file(self, params):
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=4)
        self.params = params
    
    def my_register(self):
        with open(self.target_path, "r") as f:
            target_data = json.load(f)
        target = target_data['target']
        self.optimiser.register(
            params=self.params,
            target=target,
        )
        self.all_params.append(self.params)
        self.all_targets.append(target)
        return target
    
    def write_to_file(self):
        print("writing to file!")
        with open(self.all_params_path, "w") as f:
            json.dump(self.all_params, f, indent=4)
        with open(self.all_targets_path, "w") as f:
            json.dump(self.all_targets, f, indent=4)

def main():
    b = BoGp()
    IPython.embed()
    b.write_to_file()
