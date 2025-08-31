from bayes_opt import BayesianOptimization, acquisition


class BoGp:
    def __init__(self):
        self.pbounds = {
            'vitactip_youngs_modulus': (1e-6, 4.8e+05),
            'phantom_youngs_modulus': (1e-6, 4.8e+05),
            'vitactip_poissons_ratio': (0, 0.5),
            'phantom_poissons_ratio': (0, 0.5),
            'normal_stiffness': (0, 1e10),
            'tangential_stiffness': (0, 1e10),
            'normal_damping': (0, 1e10),
            'coulomb_friction_coeff': (0, 1),
        }

    def go(self):
        acq = acquisition.UpperConfidenceBound(kappa=2.5)
        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=self.pbounds,
            verbose=2,
            random_state=1,
        )

        for _ in range(5):
            next_point = optimizer.suggest()
            target = BoGp.black_box_function(**next_point)
            optimizer.register(params=next_point, target=target)

            print(target, next_point)
        print(optimizer.max)


def main():
    bo_gp = BoGp()
    bo_gp.go()
