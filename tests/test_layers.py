import numpy as np
from neuralnet.layers import (
    linear_forward, linear_activation_forward, forward_propagation,
    linear_backward, linear_activation_backward, backward_propagation,
)
from neuralnet.utils import init_params


def test_linear_forward_hand_computed():
    A_prev = np.array([[1.0], [2.0]])          # shape (2,1)
    W = np.array([[1.0, 1.0], [2.0, 2.0]])      # shape (2,2)
    b = np.array([[0.0], [1.0]])                # shape (2,1)
    Z, _ = linear_forward(A_prev, W, b)
    # W . A_prev = [[1*1+1*2],[2*1+2*2]] = [[3],[6]]; + b = [[3],[7]]
    assert np.array_equal(Z, np.array([[3.0], [7.0]]))


def test_linear_activation_forward_sigmoid_shape():
    A_prev = np.random.randn(3, 5)
    W = np.random.randn(2, 3)
    b = np.zeros((2, 1))
    A, cache = linear_activation_forward(A_prev, W, b, "sigmoid")
    assert A.shape == (2, 5)
    assert np.all((A >= 0) & (A <= 1))


def test_forward_propagation_output_shape():
    layer_dims = [3, 4, 1]
    activations = ["relu", "sigmoid"]
    params = init_params(layer_dims, activations)
    X = np.random.randn(3, 10)
    A_last, caches = forward_propagation(X, params, activations)
    assert A_last.shape == (1, 10)
    assert len(caches) == 2


def test_linear_backward_shapes():
    A_prev = np.random.randn(3, 5)
    W = np.random.randn(2, 3)
    b = np.random.randn(2, 1)
    dZ = np.random.randn(2, 5)
    cache = (A_prev, W, b)
    dA_prev, dW, db = linear_backward(dZ, cache)
    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape


def test_backward_propagation_grad_shapes_match_params():
    layer_dims = [3, 4, 1]
    activations = ["relu", "sigmoid"]
    params = init_params(layer_dims, activations)
    X = np.random.randn(3, 5)
    Y = (np.random.rand(1, 5) > 0.5).astype(float)

    A_last, caches = forward_propagation(X, params, activations)
    grads = backward_propagation(A_last, Y, caches, activations, loss="bce")

    for l in (1, 2):
        assert grads[f"dW{l}"].shape == params[f"W{l}"].shape
        assert grads[f"db{l}"].shape == params[f"b{l}"].shape


def test_backward_propagation_softmax_cce_grad_shapes():
    layer_dims = [2, 5, 3]
    activations = ["relu", "softmax"]
    params = init_params(layer_dims, activations)
    X = np.random.randn(2, 6)
    labels = np.random.randint(0, 3, size=6)
    Y = np.eye(3)[labels].T  # one-hot (3, 6)

    A_last, caches = forward_propagation(X, params, activations)
    grads = backward_propagation(A_last, Y, caches, activations, loss="softmax_cce")

    for l in (1, 2):
        assert grads[f"dW{l}"].shape == params[f"W{l}"].shape
        assert grads[f"db{l}"].shape == params[f"b{l}"].shape