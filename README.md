# Neural Net From Scratch

A feedforward neural network built entirely from scratch using only NumPy — no PyTorch, no TensorFlow, no autograd. Every piece (forward pass, backprop, loss functions, optimizers) is implemented by hand to understand what's actually happening under the hood.

## Features

- Configurable fully-connected architecture (any number of layers/units)
- Activations: `sigmoid`, `relu`, `tanh`, `softmax` (with numerically stable implementations)
- Losses: binary cross-entropy, categorical cross-entropy (softmax)
- Optimizers: plain gradient descent, Adam
- Mini-batch training support
- Gradient checking (numerically verifies backprop is correct)
- 40 unit tests covering every function

## Project structure

```
neural-net-from-scratch/
├── neuralnet/
│   ├── __init__.py
│   ├── activations.py      # sigmoid, relu, tanh, softmax + derivatives
│   ├── layers.py            # linear/forward/backward propagation
│   ├── losses.py            # binary + categorical cross-entropy
│   ├── optimizers.py        # gradient descent, Adam
│   ├── network.py           # train / predict / accuracy
│   └── utils.py             # param init, mini-batch generator
├── tests/                    # 40 unit tests incl. gradient checking
├── examples/
│   ├── xor.py                # sanity check: does backprop even work?
│   └── mnist_classifier.py   # handwritten digit classifier (97%+ test accuracy)
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Setup

```bash
git clone https://github.com/saicharan8855/neural-net-from-scratch
cd neural-net-from-scratch
pip install -e .
pip install -r requirements.txt
```

Installing with `pip install -e .` makes the `neuralnet` package importable from anywhere in the repo (needed for the examples and tests to find it).

## Usage

**Run the XOR sanity check** (does the network learn a simple non-linear function?):
```bash
python examples/xor.py
```

**Run the digit classifier** (real handwritten digits, 10 classes):
```bash
python examples/mnist_classifier.py
```

**Run the test suite:**
```bash
pytest tests/ -v
```

**Train your own network:**
```python
from neuralnet.network import train, predict, accuracy

# X: (features, examples), Y: (output_dim, examples)
layer_dims = [input_size, 16, 8, output_size]
activations = ["relu", "relu", "sigmoid"]  # use "softmax" for multiclass output

params, costs = train(X, Y, layer_dims, activations,
                       epochs=1000, learning_rate=0.1)

preds = predict(X_test, params, activations)
print(accuracy(preds, Y_test))
```

For multiclass problems, use `activations[-1] = "softmax"`, pass `loss="softmax_cce"` to `train()`, and use `predict_multiclass()` instead of `predict()`.

## How it's built

The core pipeline, layer by layer:

1. **`init_params`** — He/Xavier-scaled random weight initialization per layer
2. **`forward_propagation`** — chains `linear_forward` → activation across all layers, caching intermediates
3. **`compute_cost`** — binary or categorical cross-entropy
4. **`backward_propagation`** — chains gradients back through every layer using cached values
5. **`update_parameters`** — gradient descent (or `adam_update` for Adam)

Correctness is checked with **gradient checking**: numerically estimating each gradient via finite differences and comparing it against the analytical (backprop) gradient. If those two don't match to about `1e-6`, something in forward/backward propagation is broken.

## Known limitations

- No convolutional layers — this is a plain fully-connected network, so it won't scale well to large images
- No GPU support (pure NumPy, CPU only)
- `mnist_classifier.py` uses scikit-learn's built-in 8x8 digit dataset rather than the full 28x28 MNIST, since the full dataset requires a download

