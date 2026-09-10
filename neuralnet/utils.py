import numpy as np


def init_params(layer_dims, activations=None, seed=42):
    """
    layer_dims: list of ints, e.g. [2, 4, 4, 1]
                (input size, hidden sizes..., output size)
    activations: list of activation names per layer (excluding input),
                 e.g. ["relu", "relu", "sigmoid"]. Used to pick He vs
                 Xavier init per layer. If None, defaults to He for all.
    Returns: dict of params {W1, b1, W2, b2, ...}
    """
    rng = np.random.default_rng(seed)  # BUG FIX: np.random.seed() sets a
    # GLOBAL seed, which silently affects random calls anywhere else in the
    # program. A local Generator avoids that side effect.

    params = {}
    L = len(layer_dims)

    if activations is None:
        activations = ["relu"] * (L - 1)

    for l in range(1, L):
        fan_in = layer_dims[l - 1]
        fan_out = layer_dims[l]

        # BUG FIX (missing feature): originally a single `activation`
        # string was used for the WHOLE network, so every layer got the
        # same init scale even though relu/sigmoid/tanh want different
        # scales. Now we look up each layer's own activation.
        layer_activation = activations[l - 1]

        if layer_activation == "relu":
            scale = np.sqrt(2.0 / fan_in)          # He init
        else:
            scale = np.sqrt(1.0 / fan_in)          # Xavier init

        params[f"W{l}"] = rng.standard_normal((fan_out, fan_in)) * scale
        params[f"b{l}"] = np.zeros((fan_out, 1))

    return params

# BUG FIX: the original file had test/demo code (init_params([2,4,4,1]) and
# print statements) sitting at module level. That code ran EVERY TIME this
# file was imported, anywhere, which is why you saw shape printouts show up
# during training. Real test code belongs in tests/, not in the module
# itself. See tests/test_utils.py.

def mini_batch_generator(X, Y, batch_size, seed=None):
    """MISSING FUNC: needed to train on batches instead of full-batch
    gradient descent every step."""
    rng = np.random.default_rng(seed)
    m = X.shape[1]
    permutation = rng.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    batches = []
    num_complete = m // batch_size
    for k in range(num_complete):
        X_batch = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
        batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete * batch_size:]
        Y_batch = Y_shuffled[:, num_complete * batch_size:]
        batches.append((X_batch, Y_batch))

    return batches