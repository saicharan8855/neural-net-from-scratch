import numpy as np
from neuralnet.utils import init_params
from neuralnet.layers import forward_propagation, backward_propagation
from neuralnet.losses import compute_cost, compute_cost_multiclass


def _dict_to_vector(d, keys):
    return np.concatenate([d[k].flatten() for k in keys])


def _vector_to_dict(vector, keys, shapes, sizes):
    d = {}
    idx = 0
    for k, shape, size in zip(keys, shapes, sizes):
        d[k] = vector[idx:idx + size].reshape(shape)
        idx += size
    return d


def _gradient_check(X, Y, layer_dims, activations, loss="bce", epsilon=1e-7):
    params = init_params(layer_dims, activations, seed=1)
    A_last, caches = forward_propagation(X, params, activations)
    grads = backward_propagation(A_last, Y, caches, activations, loss=loss)

    L = len(layer_dims) - 1
    param_keys = [f"W{l}" for l in range(1, L + 1)] + [f"b{l}" for l in range(1, L + 1)]
    grad_keys = [f"dW{l}" for l in range(1, L + 1)] + [f"db{l}" for l in range(1, L + 1)]

    param_vector = _dict_to_vector(params, param_keys)
    grad_vector = _dict_to_vector(grads, grad_keys)

    shapes = [params[k].shape for k in param_keys]
    sizes = [params[k].size for k in param_keys]

    num_grad = np.zeros_like(param_vector)
    cost_fn = compute_cost_multiclass if loss == "softmax_cce" else compute_cost

    for i in range(len(param_vector)):
        theta_plus = param_vector.copy()
        theta_plus[i] += epsilon
        theta_minus = param_vector.copy()
        theta_minus[i] -= epsilon

        params_plus = _vector_to_dict(theta_plus, param_keys, shapes, sizes)
        params_minus = _vector_to_dict(theta_minus, param_keys, shapes, sizes)

        A_plus, _ = forward_propagation(X, params_plus, activations)
        A_minus, _ = forward_propagation(X, params_minus, activations)

        num_grad[i] = (cost_fn(A_plus, Y) - cost_fn(A_minus, Y)) / (2 * epsilon)

    diff = np.linalg.norm(grad_vector - num_grad) / (
        np.linalg.norm(grad_vector) + np.linalg.norm(num_grad)
    )
    return diff


def test_gradient_check_binary_network():
    np.random.seed(1)
    X = np.random.randn(3, 4)
    Y = (np.random.rand(1, 4) > 0.5).astype(float)
    layer_dims = [3, 4, 1]
    activations = ["relu", "sigmoid"]

    diff = _gradient_check(X, Y, layer_dims, activations, loss="bce")
    assert diff < 1e-6, f"backprop gradients don't match numerical gradients: diff={diff}"


def test_gradient_check_softmax_network():
    np.random.seed(1)
    X = np.random.randn(2, 5)
    labels = np.random.randint(0, 3, size=5)
    Y = np.eye(3)[labels].T  # one-hot (3, 5)
    layer_dims = [2, 4, 3]
    activations = ["relu", "softmax"]

    diff = _gradient_check(X, Y, layer_dims, activations, loss="softmax_cce")
    assert diff < 1e-6, f"backprop gradients don't match numerical gradients: diff={diff}"


def test_gradient_check_deeper_network():
    np.random.seed(2)
    X = np.random.randn(4, 6)
    Y = (np.random.rand(1, 6) > 0.5).astype(float)
    layer_dims = [4, 5, 3, 1]
    activations = ["relu", "tanh", "sigmoid"]

    diff = _gradient_check(X, Y, layer_dims, activations, loss="bce")
    assert diff < 1e-6, f"backprop gradients don't match numerical gradients: diff={diff}"