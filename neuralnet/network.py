from neuralnet.utils import init_params
from neuralnet.layers import forward_propagation, backward_propagation
from neuralnet.losses import compute_cost, compute_cost_multiclass
from neuralnet.optimizers import update_parameters
import numpy as np


def train(X, Y, layer_dims, activations, epochs, learning_rate, loss="bce"):
    # BUG FIX: init_params now takes the per-layer activations list too,
    # so each layer gets the correct (He vs Xavier) init scale.
    params = init_params(layer_dims, activations)
    costs = []

    for i in range(epochs):
        A_last, caches = forward_propagation(X, params, activations)

        if loss == "softmax_cce":
            cost = compute_cost_multiclass(A_last, Y)
        else:
            cost = compute_cost(A_last, Y)

        grads = backward_propagation(A_last, Y, caches, activations, loss=loss)
        params = update_parameters(params, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print(f"Epoch {i}: cost = {cost}")

    return params, costs


def predict(X, params, activations):
    """Binary predictions (threshold at 0.5)."""
    A_last, _ = forward_propagation(X, params, activations)
    return (A_last > 0.5).astype(int)


def predict_multiclass(X, params, activations):
    # MISSING FUNC: binary predict() doesn't make sense for a softmax
    # output layer — need argmax over classes instead.
    A_last, _ = forward_propagation(X, params, activations)
    return np.argmax(A_last, axis=0)


def accuracy(preds, Y):
    return (preds == Y).mean()