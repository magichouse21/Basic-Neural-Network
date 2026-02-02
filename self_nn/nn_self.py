import numpy as np
from keras.datasets import mnist

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ----------------------------
# init
# ----------------------------
def initialize(layer_dims, seed=67, weight_scale=0.01):
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
    A = X
    caches = []
    L = len(params) // 2

    for l in range(1, L):
        W, b = params[f"W{l}"], params[f"b{l}"]
        Z = W @ A + b
        A_next = relu(Z)
        caches.append((A, W, b, Z, "relu"))
        A = A_next

    W, b = params[f"W{L}"], params[f"b{L}"]
    ZL = W @ A + b
    AL = softmax(ZL)
    caches.append((A, W, b, ZL, "softmax"))

    return AL, caches

# ----------------------------
# loss
# ----------------------------
def cost(AL, Y, eps=1e-12):
    m = Y.shape[1]
    AL_clipped = np.clip(AL, eps, 1.0)
    return -(1 / m) * np.sum(Y * np.log(AL_clipped))

# ----------------------------
# backward
# ----------------------------
def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = Y.shape[1]

    A_prev, W, b, Z, act = caches[-1]
    dZ = AL - Y
    grads[f"dW{L}"] = (1 / m) * (dZ @ A_prev.T)
    grads[f"db{L}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    for l in range(L - 1, 0, -1):
        A_prev, W, b, Z, act = caches[l - 1]
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
    return np.argmax(AL, axis=0)

def one_hot(y, num_classes):
    m = y.shape[0]
    Y = np.zeros((num_classes, m))
    Y[y, np.arange(m)] = 1
    return Y

def create_minibatches(X, y, batch_size, seed=0):
    """
    X: (n_x, m)
    y: (m,)
    yields (X_batch, y_batch)
    """
    m = X.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)

    X_shuff = X[:, perm]
    y_shuff = y[perm]

    for start in range(0, m, batch_size):
        end = start + batch_size
        yield X_shuff[:, start:end], y_shuff[start:end]

# ----------------------------
# training loop (MINIBATCH)
# ----------------------------
def train_minibatch(X, y, layer_dims, lr, epochs, batch_size=64, print_every=1, seed=67):
    """
    X: (n_x, m)
    y: (m,)
    """
    params = initialize(layer_dims, seed=seed)
    n_y = layer_dims[-1]

    for e in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        seen = 0

        # different shuffle each epoch
        for Xb, yb in create_minibatches(X, y, batch_size, seed=seed + e):
            Yb = one_hot(yb, n_y)

            ALb, caches = forward_prop(Xb, params)
            Jb = cost(ALb, Yb)
            grads = backprop(ALb, Yb, caches)
            params = update_params(params, grads, lr)

            mb = yb.shape[0]
            epoch_loss += Jb * mb
            preds_b = np.argmax(ALb, axis=0)
            epoch_correct += np.sum(preds_b == yb)
            seen += mb

        if print_every and (e % print_every == 0 or e == 1):
            print(f"epoch {e:4d} | loss {epoch_loss/seen:.4f} | acc {epoch_correct/seen:.4f}")

    return params

# ----------------------------
# load data + preprocess
# ----------------------------
(X_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize to [0,1]
X_train = X_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# flatten and transpose to (784, m)
X_train = X_train.reshape(X_train.shape[0], -1).T
x_test  = x_test.reshape(x_test.shape[0], -1).T

# ----------------------------
# run training (minibatches)
# ----------------------------
batch_size = 128
params = train_minibatch(
    X_train, y_train,
    layer_dims=[784, 128, 64, 10],
    lr=0.1,
    epochs=20,
    batch_size=batch_size,
    print_every=1
)

# ----------------------------
# evaluate on test
# ----------------------------
y_pred = predict(x_test, params)
test_acc = np.mean(y_pred == y_test)
print("test acc:", test_acc)
