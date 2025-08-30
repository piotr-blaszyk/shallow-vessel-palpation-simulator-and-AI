from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

acq = acquisition.UpperConfidenceBound(kappa=2.5)

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

# optimizer = BayesianOptimization(
#     f=black_box_function,
#     pbounds=pbounds,
#     random_state=1,
# )

# optimizer.maximize(
#     init_points=20,
#     n_iter=20,
# )

# print(optimizer.max)
# print()

# for i, res in enumerate(optimizer.res):
#     print("Iteration {}: \n\t{}".format(i, res))

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
