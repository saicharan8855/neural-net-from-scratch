import numpy as np
from neuralnet.activations import (
    sigmoid, sigmoid_backward,
    relu, relu_backward,
    tanh, tanh_backward,
    softmax,
)


def test_sigmoid_known_values():
    assert np.isclose(sigmoid(np.array([0.0])), 0.5)
    assert sigmoid(np.array([100.0])) > 0.99
    assert sigmoid(np.array([-100.0])) < 0.01


def test_sigmoid_no_overflow():
    # regression test for the overflow bug
    out = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(out))


def test_sigmoid_backward_shape_and_range():
    z = np.array([-1.0, 0.0, 1.0])
    dA = np.ones_like(z)
    dZ = sigmoid_backward(dA, z)
    assert dZ.shape == z.shape
    assert np.all(dZ > 0)  # sigmoid derivative always positive


def test_relu_known_values():
    z = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(relu(z), np.array([0.0, 0.0, 3.0]))


def test_relu_backward():
    z = np.array([-1.0, 0.0, 2.0])
    dA = np.array([5.0, 5.0, 5.0])
    dZ = relu_backward(dA, z)
    assert np.array_equal(dZ, np.array([0.0, 0.0, 5.0]))


def test_tanh_known_values():
    assert np.isclose(tanh(np.array([0.0])), 0.0)
    assert tanh(np.array([100.0])) > 0.99


def test_tanh_backward_at_zero():
    z = np.array([0.0])
    dA = np.array([1.0])
    # derivative of tanh at 0 is 1
    assert np.isclose(tanh_backward(dA, z), 1.0)


def test_softmax_sums_to_one():
    z = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]])  # 3 classes, 2 examples
    out = softmax(z)
    col_sums = np.sum(out, axis=0)
    assert np.allclose(col_sums, 1.0)


def test_softmax_no_overflow_on_large_values():
    z = np.array([[1000.0], [1.0], [0.0]])
    out = softmax(z)
    assert np.all(np.isfinite(out))
    assert np.isclose(np.sum(out), 1.0)