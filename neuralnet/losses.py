import numpy as np


def compute_cost(A_last, Y):
    """Binary cross-entropy."""
    m = Y.shape[1]
    # BUG FIX: if a prediction is exactly 0 or 1 (common once the net is
    # confident), log(0) = -inf and the cost becomes nan. Clipping keeps
    # values just inside (0, 1).
    eps = 1e-12
    A_last = np.clip(A_last, eps, 1 - eps)
    cost = -(1 / m) * np.sum(Y * np.log(A_last) + (1 - Y) * np.log(1 - A_last))
    return np.squeeze(cost)


def compute_cost_multiclass(A_last, Y):
    """
    Categorical cross-entropy, for use with a softmax output layer.
    A_last, Y shape: (num_classes, num_examples), Y one-hot encoded.
    MISSING FUNC: activations.py already had softmax, but there was no
    matching loss function to actually train a multiclass network with it.
    """
    m = Y.shape[1]
    eps = 1e-12
    A_last = np.clip(A_last, eps, 1 - eps)
    cost = -(1 / m) * np.sum(Y * np.log(A_last))
    return np.squeeze(cost)