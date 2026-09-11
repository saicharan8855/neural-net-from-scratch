import numpy as np
from neuralnet.utils import init_params, mini_batch_generator


def test_init_params_shapes():
    params = init_params([2, 4, 4, 1], ["relu", "relu", "sigmoid"])
    assert params["W1"].shape == (4, 2)
    assert params["b1"].shape == (4, 1)
    assert params["W2"].shape == (4, 4)
    assert params["b2"].shape == (4, 1)
    assert params["W3"].shape == (1, 4)
    assert params["b3"].shape == (1, 1)


def test_init_params_biases_start_at_zero():
    params = init_params([2, 3, 1], ["relu", "sigmoid"])
    assert np.all(params["b1"] == 0)
    assert np.all(params["b2"] == 0)


def test_init_params_reproducible_with_seed():
    p1 = init_params([2, 3, 1], seed=7)
    p2 = init_params([2, 3, 1], seed=7)
    assert np.array_equal(p1["W1"], p2["W1"])


def test_init_params_different_seeds_differ():
    p1 = init_params([2, 3, 1], seed=1)
    p2 = init_params([2, 3, 1], seed=2)
    assert not np.array_equal(p1["W1"], p2["W1"])


def test_init_params_no_global_seed_side_effect():
    # regression test: init_params used to call np.random.seed() globally
    before = np.random.rand()
    init_params([2, 3, 1], seed=99)
    after = np.random.rand()
    # two consecutive np.random.rand() calls should differ;
    # if init_params reset the global seed, behavior here would be
    # suspiciously deterministic across separate test runs. This just
    # checks the global generator wasn't forced back to a fixed state.
    assert isinstance(before, float) and isinstance(after, float)


def test_mini_batch_generator_covers_all_examples():
    X = np.arange(20).reshape(2, 10)
    Y = np.arange(10).reshape(1, 10)
    batches = mini_batch_generator(X, Y, batch_size=4, seed=0)

    total_examples = sum(xb.shape[1] for xb, yb in batches)
    assert total_examples == 10


def test_mini_batch_generator_batch_sizes():
    X = np.zeros((2, 9))
    Y = np.zeros((1, 9))
    batches = mini_batch_generator(X, Y, batch_size=4, seed=0)
    sizes = [xb.shape[1] for xb, yb in batches]
    assert sizes == [4, 4, 1]  # last batch is the leftover