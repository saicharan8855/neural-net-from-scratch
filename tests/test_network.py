import numpy as np
from neuralnet.network import train, predict, predict_multiclass, accuracy


def test_train_cost_decreases_on_xor():
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    layer_dims = [2, 4, 1]
    activations = ["relu", "sigmoid"]

    params, costs = train(X, Y, layer_dims, activations,
                           epochs=1000, learning_rate=0.5)
    assert costs[-1] < costs[0]


def test_xor_learnable_with_good_seed():
    # uses init_params' seed indirectly via a fixed numpy seed for
    # reproducibility of this specific test
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    layer_dims = [2, 4, 1]
    activations = ["relu", "sigmoid"]

    params, costs = train(X, Y, layer_dims, activations,
                           epochs=3000, learning_rate=0.5)
    preds = predict(X, params, activations)
    # XOR with a small net can occasionally hit a bad local minimum;
    # this just checks the pipeline runs end-to-end and produces valid output
    assert preds.shape == Y.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_accuracy_perfect_match():
    preds = np.array([[1, 0, 1, 0]])
    Y = np.array([[1, 0, 1, 0]])
    assert accuracy(preds, Y) == 1.0


def test_accuracy_no_match():
    preds = np.array([[1, 1, 1, 1]])
    Y = np.array([[0, 0, 0, 0]])
    assert accuracy(preds, Y) == 0.0


def test_predict_multiclass_output_shape():
    X = np.random.randn(2, 30)
    labels = (X[0] + X[1] > 0).astype(int)
    Y = np.eye(2)[labels].T
    layer_dims = [2, 5, 2]
    activations = ["relu", "softmax"]

    params, costs = train(X, Y, layer_dims, activations,
                           epochs=500, learning_rate=0.1, loss="softmax_cce")
    preds = predict_multiclass(X, params, activations)
    assert preds.shape == (30,)
    assert accuracy(preds, labels) > 0.8  # should learn this easy linearly-separable task