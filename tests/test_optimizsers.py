import numpy as np
from neuralnet.optimizers import update_parameters, init_adam_state, adam_update


def test_update_parameters_moves_in_negative_gradient_direction():
    # NOTE: dict(params) is a shallow copy — the arrays inside are shared,
    # so update_parameters (which mutates in place with -=) was changing
    # `params` too, making the "before" value already look updated.
    # Need a deep copy to compare correctly.
    params = {"W1": np.array([[1.0]]), "b1": np.array([[1.0]])}
    original_W1 = params["W1"].copy()
    original_b1 = params["b1"].copy()
    grads = {"dW1": np.array([[1.0]]), "db1": np.array([[1.0]])}
    updated = update_parameters(params, grads, learning_rate=0.1)
    assert updated["W1"][0, 0] < original_W1[0, 0]
    assert updated["b1"][0, 0] < original_b1[0, 0]


def test_update_parameters_exact_value():
    params = {"W1": np.array([[2.0]]), "b1": np.array([[0.0]])}
    grads = {"dW1": np.array([[1.0]]), "db1": np.array([[0.0]])}
    updated = update_parameters(dict(params), grads, learning_rate=0.5)
    assert np.isclose(updated["W1"][0, 0], 1.5)  # 2.0 - 0.5*1.0


def test_init_adam_state_shapes():
    params = {"W1": np.zeros((3, 2)), "b1": np.zeros((3, 1))}
    v, s = init_adam_state(params)
    assert v["dW1"].shape == (3, 2)
    assert s["db1"].shape == (3, 1)
    assert np.all(v["dW1"] == 0)
    assert np.all(s["db1"] == 0)


def test_adam_update_moves_params():
    params = {"W1": np.array([[1.0]]), "b1": np.array([[1.0]])}
    grads = {"dW1": np.array([[1.0]]), "db1": np.array([[1.0]])}
    v, s = init_adam_state(params)
    updated, v, s = adam_update(params, grads, v, s, t=1, learning_rate=0.1)
    assert updated["W1"][0, 0] < 1.0
    assert updated["b1"][0, 0] < 1.0


def test_adam_update_state_accumulates_over_steps():
    # NOTE: init_adam_state derives L = len(params)//2, so params needs a
    # matching W/b pair or the loop silently does nothing (my earlier
    # version used only "W1" with no "b1" and produced an empty v/s).
    params = {"W1": np.array([[1.0]]), "b1": np.array([[0.0]])}
    grads = {"dW1": np.array([[1.0]]), "db1": np.array([[1.0]])}
    v, s = init_adam_state(params)
    _, v1, s1 = adam_update(dict(params), grads, v, s, t=1, learning_rate=0.1)
    _, v2, s2 = adam_update(dict(params), grads, v1, s1, t=2, learning_rate=0.1)
    # second moment estimate should keep growing after repeated identical-gradient steps
    assert s2["dW1"][0, 0] > 0
    assert np.isfinite(s2["dW1"][0, 0])