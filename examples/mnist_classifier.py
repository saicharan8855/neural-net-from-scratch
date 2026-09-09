"""
Handwritten digit classifier using the from-scratch neural net.

Uses sklearn's `load_digits` dataset (1797 real handwritten digit images,
8x8 pixels, 10 classes 0-9). This is a smaller stand-in for the full
28x28 MNIST dataset -- same idea, but ships with scikit-learn so it
runs fully offline with no download needed.

Run:
    pip install scikit-learn
    python examples/mnist_classifier.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neuralnet.network import train, predict_multiclass, accuracy


def load_data():
    digits = load_digits()
    X = digits.data.T          # (64, num_examples) -- features as columns
    y = digits.target          # (num_examples,)

    # scale pixel values from [0, 16] to [0, 1]
    X = X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X_train.T, X_test.T  # back to (features, examples)

    # one-hot encode labels for softmax + categorical cross-entropy
    Y_train = np.eye(10)[y_train].T   # (10, num_train)

    return X_train, Y_train, y_train, X_test, y_test


def main():
    X_train, Y_train, y_train, X_test, y_test = load_data()
    print(f"Train set: {X_train.shape[1]} examples")
    print(f"Test set:  {X_test.shape[1]} examples")

    layer_dims = [64, 32, 16, 10]
    activations = ["relu", "relu", "softmax"]

    params, costs = train(
        X_train, Y_train, layer_dims, activations,
        epochs=1500, learning_rate=0.3, loss="softmax_cce"
    )

    train_preds = predict_multiclass(X_train, params, activations)
    test_preds = predict_multiclass(X_test, params, activations)

    print(f"\nTrain accuracy: {accuracy(train_preds, y_train):.4f}")
    print(f"Test accuracy:  {accuracy(test_preds, y_test):.4f}")


if __name__ == "__main__":
    main()