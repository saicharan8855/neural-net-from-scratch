import numpy as np
from neuralnet.losses import compute_cost, compute_cost_multiclass


def test_compute_cost_perfect_prediction_near_zero():
    A = np.array([[0.999999999999]])
    Y = np.array([[1.0]])
    cost = compute_cost(A, Y)
    assert cost < 1e-6


def test_compute_cost_wrong_prediction_is_high():
    A = np.array([[0.001]])
    Y = np.array([[1.0]])
    cost = compute_cost(A, Y)
    assert cost > 5


def test_compute_cost_no_nan_at_extremes():
    # regression test for the log(0) bug
    A = np.array([[1.0, 0.0]])
    Y = np.array([[1.0, 0.0]])
    cost = compute_cost(A, Y)
    assert np.isfinite(cost)


def test_compute_cost_multiclass_perfect_prediction():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # 3 classes, 2 examples
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    cost = compute_cost_multiclass(A, Y)
    assert cost < 1e-6


def test_compute_cost_multiclass_no_nan_at_extremes():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    Y = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])  # deliberately wrong
    cost = compute_cost_multiclass(A, Y)
    assert np.isfinite(cost)
    assert cost > 0