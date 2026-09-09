from neuralnet.utils import init_params
from neuralnet.layers import forward_propagation, backward_propagation
from neuralnet.losses import compute_cost
from neuralnet.optimizers import update_parameters

def train(X, Y, layer_dims, activations, epochs, learning_rate):
    params = init_params(layer_dims)
    costs = []

    for i in range(epochs):
        A_last, caches = forward_propagation(X, params, activations)
        cost = compute_cost(A_last, Y)
        grads = backward_propagation(A_last, Y, caches, activations)
        params = update_parameters(params, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print(f"Epoch {i}: cost = {cost}")

    return params, costs

def predict(X, params, activations):
    A_last, _ = forward_propagation(X, params, activations)
    return (A_last > 0.5).astype(int)

def accuracy(preds, Y):
    return (preds == Y).mean()