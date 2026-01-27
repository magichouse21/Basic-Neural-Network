import numpy as np
from keras.datasets import mnist

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ----------------------------
# init
# ----------------------------
def initialize(layer_dims, seed=67, weight_scale=0.01):
    """
    layer_dims: list like [n_x, n_h1, n_h2, ..., n_y]
    returns dict of parameters W1, b1, ..., WL, bL
    """
    np.random.seed(seed)
    L = len(layer_dims) - 1
    params = {}

    for l in range(1, L + 1):
        params[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * weight_scale
        params[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return params

# ----------------------------
# activations
# ----------------------------
def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)  # stability
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# ----------------------------
# forward
# ----------------------------
def forward_prop(X, params):
    """
    X: (n_x, m)
    returns:
      AL: (n_y, m)
      caches: list of caches per layer
        cache = (A_prev, W, b, Z, activation_name)
    """
    A = X
    caches = []
    L = len(params) // 2

    # hidden layers: ReLU
    for l in range(1, L):
        W, b = params[f"W{l}"], params[f"b{l}"]
        Z = W @ A + b
        A_next = relu(Z)
        caches.append((A, W, b, Z, "relu"))
        A = A_next

    # output layer: Softmax
    W, b = params[f"W{L}"], params[f"b{L}"]
    ZL = W @ A + b
    AL = softmax(ZL)
    caches.append((A, W, b, ZL, "softmax"))

    return AL, caches

# ----------------------------
# loss
# ----------------------------
def cost(AL, Y, eps=1e-12):
    """
    AL: (n_y, m) softmax probabilities
    Y:  (n_y, m) one-hot labels
    """
    m = Y.shape[1]
    AL_clipped = np.clip(AL, eps, 1.0)  # avoid log(0)
    return -(1 / m) * np.sum(Y * np.log(AL_clipped))

# ----------------------------
# backward
# ----------------------------
def backprop(AL, Y, caches):
    """
    AL: (n_y, m)
    Y:  (n_y, m)
    caches: from forward_prop
    returns grads dict dWl, dbl
    """
    grads = {}
    L = len(caches)
    m = Y.shape[1]

    # --- output layer: softmax + cross-entropy ---
    # dZ = AL - Y (this is the nice simplification)
    A_prev, W, b, Z, act = caches[-1]
    dZ = AL - Y                          # (n_y, m)
    grads[f"dW{L}"] = (1 / m) * (dZ @ A_prev.T)
    grads[f"db{L}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ                   # (n_{L-1}, m)

    # --- hidden layers: ReLU ---
    for l in range(L - 1, 0, -1):
        A_prev, W, b, Z, act = caches[l - 1]  # layer l cache
        dZ = relu_backward(dA_prev, Z)
        grads[f"dW{l}"] = (1 / m) * (dZ @ A_prev.T)
        grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ

    return grads

# ----------------------------
# update
# ----------------------------
def update_params(params, grads, lr):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f"W{l}"] -= lr * grads[f"dW{l}"]
        params[f"b{l}"] -= lr * grads[f"db{l}"]
    return params

# ----------------------------
# utilities
# ----------------------------
def predict(X, params):
    AL, _ = forward_prop(X, params)
    return np.argmax(AL, axis=0)  # (m,)

def one_hot(y, num_classes):
    """
    y: (m,) integer labels
    returns (num_classes, m)
    """
    m = y.shape[0]
    Y = np.zeros((num_classes, m))
    Y[y, np.arange(m)] = 1
    return Y

# ----------------------------
# training loop
# ----------------------------
def train(X, y, layer_dims, lr=0.1, epochs=1000, print_every=100):
    """
    X: (n_x, m)
    y: (m,) integer labels
    layer_dims: [n_x, ..., n_y]
    """
    n_y = layer_dims[-1]
    Y = one_hot(y, n_y)

    params = initialize(layer_dims)

    for e in range(1, epochs + 1):
        AL, caches = forward_prop(X, params)
        J = cost(AL, Y)
        grads = backprop(AL, Y, caches)
        params = update_params(params, grads, lr)

        if print_every and (e % print_every == 0 or e == 1):
            preds = np.argmax(AL, axis=0)
            acc = np.mean(preds == y)
            print(f"epoch {e:4d} | loss {J:.4f} | acc {acc:.4f}")

    return params   

(X_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1).T

params = train(X_train, y_train, layer_dims=[784, 128, 64, 10], lr=0.1, epochs=1000, print_every=100)

y_pred = predict(x_test, params)
test_acc = np.mean(y_pred == y_test)
print("test acc:", test_acc)