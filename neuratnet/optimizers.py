def update_parameters(params, grads, learning_rate):
    L = len(params) // 2

    for l in range(1, L + 1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return params