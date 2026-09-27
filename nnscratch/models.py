"""Softmax regression and a one-hidden-layer MLP in plain numpy.

Every model exposes the same interface so one training loop drives all of them:
    params             dict of numpy arrays (updated in place by the optimizer)
    predict_proba(X)   class probabilities
    loss_and_grads(X, y) -> (loss, {name: gradient})   with hand-derived gradients

X may be a dense array or a scipy sparse matrix. Losses are mean cross-entropy plus an
L2 penalty lambda * sum(W^2) on weight matrices (biases are not regularized).
"""
import numpy as np


def sigmoid(z):
    # Split by sign so exp never overflows.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


EPS = 1e-12


class SoftmaxRegression:
    """Multiclass logistic regression: p(y | x) = softmax(xW + b).

    With softmax + categorical cross-entropy, dL/dZ = P - Y (Y one-hot).
    """

    def __init__(self, n_features, n_classes, l2=0.0):
        self.l2 = l2
        self.n_classes = n_classes
        self.params = {"W": np.zeros((n_features, n_classes)), "b": np.zeros(n_classes)}

    def predict_proba(self, X):
        return softmax(np.asarray(X @ self.params["W"]) + self.params["b"])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def loss_and_grads(self, X, y):
        W = self.params["W"]
        P = self.predict_proba(X)
        Y = one_hot(y, self.n_classes)
        n = len(y)
        loss = -np.sum(Y * np.log(P + EPS)) / n + self.l2 * np.sum(W ** 2)
        dZ = (P - Y) / n
        grads = {"W": np.asarray(X.T @ dZ) + 2 * self.l2 * W, "b": dZ.sum(axis=0)}
        return loss, grads


class MLP:
    """x -> sigmoid(x W1 + b1) -> softmax(h W2 + b2).

    Backprop:
        dZ2 = (P - Y) / n
        dW2 = H^T dZ2 + 2 lambda W2          db2 = sum(dZ2)
        dH  = dZ2 W2^T
        dZ1 = dH * H * (1 - H)               (sigmoid'(z) = s(z)(1 - s(z)))
        dW1 = X^T dZ1 + 2 lambda W1          db1 = sum(dZ1)
    """

    def __init__(self, n_features, n_hidden, n_classes, l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.l2 = l2
        self.n_classes = n_classes
        self.params = {
            "W1": rng.standard_normal((n_features, n_hidden)) * 0.01,
            "b1": np.zeros(n_hidden),
            "W2": rng.standard_normal((n_hidden, n_classes)) * 0.01,
            "b2": np.zeros(n_classes),
        }

    def _forward(self, X):
        p = self.params
        H = sigmoid(np.asarray(X @ p["W1"]) + p["b1"])
        return H, softmax(H @ p["W2"] + p["b2"])

    def predict_proba(self, X):
        return self._forward(X)[1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def loss_and_grads(self, X, y):
        p = self.params
        H, P = self._forward(X)
        Y = one_hot(y, self.n_classes)
        n = len(y)
        loss = -np.sum(Y * np.log(P + EPS)) / n + self.l2 * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))

        dZ2 = (P - Y) / n
        dH = dZ2 @ p["W2"].T
        dZ1 = dH * H * (1 - H)
        grads = {
            "W2": H.T @ dZ2 + 2 * self.l2 * p["W2"],
            "b2": dZ2.sum(axis=0),
            "W1": np.asarray(X.T @ dZ1) + 2 * self.l2 * p["W1"],
            "b1": dZ1.sum(axis=0),
        }
        return loss, grads
