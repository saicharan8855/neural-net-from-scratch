import numpy as np


def update_parameters(params, grads, learning_rate):
    """Plain gradient descent."""
    L = len(params) // 2

    for l in range(1, L + 1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return params


def init_adam_state(params):
    """MISSING FUNC: Adam needs running moment estimates (v, s) per param."""
    v, s = {}, {}
    L = len(params) // 2
    for l in range(1, L + 1):
        v[f"dW{l}"] = np.zeros_like(params[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(params[f"b{l}"])
        s[f"dW{l}"] = np.zeros_like(params[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(params[f"b{l}"])
    return v, s


def adam_update(params, grads, v, s, t, learning_rate=0.001,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
    """MISSING FUNC: Adam optimizer, generally converges faster/more
    reliably than plain gradient descent."""
    L = len(params) // 2
    v_corrected, s_corrected = {}, {}

    for l in range(1, L + 1):
        for p in (f"dW{l}", f"db{l}"):
            v[p] = beta1 * v[p] + (1 - beta1) * grads[p]
            s[p] = beta2 * s[p] + (1 - beta2) * (grads[p] ** 2)

            v_corrected[p] = v[p] / (1 - beta1 ** t)
            s_corrected[p] = s[p] / (1 - beta2 ** t)

            key = p[1:]  # "dW1" -> "W1", "db1" -> "b1"
            params[key] -= learning_rate * v_corrected[p] / \
                (np.sqrt(s_corrected[p]) + epsilon)

    return params, v, s