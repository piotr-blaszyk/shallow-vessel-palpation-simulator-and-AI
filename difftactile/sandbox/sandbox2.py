from bayes_opt import BayesianOptimization, acquisition


def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

acq = acquisition.UpperConfidenceBound(kappa=2.5)

pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=None,
    acquisition_function=acq,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    verbose=2,
    random_state=1,
)

for _ in range(5):
    next_point = optimizer.suggest()
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)

    print(target, next_point)
print(optimizer.max)
