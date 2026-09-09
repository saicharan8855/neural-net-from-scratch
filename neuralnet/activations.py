import numpy as np


def sigmoid(z):
    # BUG FIX: np.exp(-z) overflows for very negative z (e.g. z = -1000).
    # np.clip keeps z in a safe range so exp() never blows up.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(dA, z):
    s = sigmoid(z)
    return dA * s * (1 - s)


def relu(z):
    return np.maximum(0, z)


def relu_backward(dA, z):
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0
    return dZ


def tanh(z):
    return np.tanh(z)


def tanh_backward(dA, z):
    t = tanh(z)
    return dA * (1 - t ** 2)


def softmax(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)