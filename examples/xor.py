"""
XOR sanity check for the from-scratch neural net.

XOR isn't linearly separable, so a network needs at least one hidden
layer to learn it. If this doesn't converge, something in forward/backward
propagation is broken -- fix that before trying anything bigger (like the
MNIST example).

Note: 8 hidden units are used instead of the more minimal 4, since 4
units can occasionally get stuck in a bad local minimum on this tiny
dataset depending on random init. This isn't a bug -- it's normal for
small networks on toy problems -- but 8 units makes the demo reliably
converge.

Run:
    python examples/xor.py
"""
import numpy as np
from neuralnet.network import train, predict, accuracy

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

layer_dims = [2, 8, 1]
activations = ["relu", "sigmoid"]

params, costs = train(X, Y, layer_dims, activations,
                       epochs=3000, learning_rate=0.5)

preds = predict(X, params, activations)
print("\nPredictions:", preds)
print("Actual:     ", Y)
print("Accuracy:   ", accuracy(preds, Y))