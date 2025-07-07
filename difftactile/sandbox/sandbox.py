import numpy as np
cost = np.array([
    [4, 1, 3], 
    [2, 0, 5], 
    [3, 2, 2], 
    [1, 2, 3]
])
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost)
print(f'row_ind: {row_ind}')
print(f'col_ind: {col_ind}')
print(cost[row_ind, col_ind].sum())