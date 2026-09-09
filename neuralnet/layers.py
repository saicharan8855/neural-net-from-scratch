import numpy as np
from neuralnet.activations import sigmoid, relu, tanh, softmax, \
    sigmoid_backward, relu_backward, tanh_backward


def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "softmax":
        A = softmax(Z)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache


def forward_propagation(X, params, activations):
    caches = []
    A = X
    L = len(params) // 2

    for l in range(1, L + 1):
        A_prev = A
        W = params[f"W{l}"]
        b = params[f"b{l}"]
        A, cache = linear_activation_forward(A_prev, W, b, activations[l - 1])
        caches.append(cache)

    return A, caches


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "tanh":
        dZ = tanh_backward(dA, Z)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def backward_propagation(A_last, Y, caches, activations, loss="bce"):
    """
    loss: "bce" (binary cross-entropy, output activation must be "sigmoid")
          or "softmax_cce" (softmax + categorical cross-entropy)

    BUG FIX: the original always used the binary cross-entropy dA formula
    -(Y/A - (1-Y)/(1-A)), even if the last layer was softmax. That formula
    is only correct paired with a sigmoid output. For softmax + categorical
    cross-entropy the gradient simplifies directly to (A_last - Y), and
    that combined gradient must be used as dZ, not dA, for the last layer
    (softmax_backward isn't implemented separately on purpose).
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(A_last.shape)

    if loss == "softmax_cce":
        # Combined softmax + cross-entropy gradient shortcut.
        dZ_last = A_last - Y
        linear_cache_last, _ = caches[L - 1]
        dA_prev, dW, db = linear_backward(dZ_last, linear_cache_last)
        grads[f"dW{L}"] = dW
        grads[f"db{L}"] = db
        dA = dA_prev
        start = L - 2
    else:
        eps = 1e-12
        A_clipped = np.clip(A_last, eps, 1 - eps)
        dA_last = -(np.divide(Y, A_clipped) - np.divide(1 - Y, 1 - A_clipped))
        dA = dA_last
        start = L - 1

    for l in reversed(range(start + 1)):
        cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA, cache, activations[l])
        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db
        dA = dA_prev

    return grads