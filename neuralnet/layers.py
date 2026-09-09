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
        A, cache = linear_activation_forward(A_prev, W, b, activations[l-1])
        caches.append(cache)

    return A, caches

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
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

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_propagation(A_last, Y, caches, activations):
    grads = {}
    L = len(caches)
    Y = Y.reshape(A_last.shape)

    dA_last = -(np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))

    dA = dA_last
    for l in reversed(range(L)):
        cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA, cache, activations[l])
        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db
        dA = dA_prev

    return grads