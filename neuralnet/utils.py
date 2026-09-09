import numpy as np

def init_params(layer_dims, activation="relu"):
    """
    layer_dims: list of ints, e.g. [2, 4, 4, 1]
                (input size, hidden sizes..., output size)
    Returns: dict of params {W1, b1, W2, b2, ...}
    """
    np.random.seed(42)  # reproducibility while testing
    params = {}
    L = len(layer_dims)  # number of layers including input

    for l in range(1, L):
        fan_in = layer_dims[l-1]
        fan_out = layer_dims[l]

        if activation == "relu":
            # He initialization
            scale = np.sqrt(2.0 / fan_in)
        else:
            # Xavier initialization (for sigmoid/tanh)
            scale = np.sqrt(1.0 / fan_in)

        params[f"W{l}"] = np.random.randn(fan_out, fan_in) * scale
        params[f"b{l}"] = np.zeros((fan_out, 1))

    return params

params = init_params([2,4,4,1])
for key in params:
    print(key, params[key].shape)