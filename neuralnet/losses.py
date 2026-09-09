import numpy as np

def compute_cost(A_last, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(A_last) + (1 - Y) * np.log(1 - A_last))
    return np.squeeze(cost)

