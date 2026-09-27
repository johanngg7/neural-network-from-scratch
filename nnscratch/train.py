"""Optimizers, the mini-batch training loop, and evaluation metrics."""
from typing import Callable, Dict, List

import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class AdaGrad:
    """Per-parameter learning rates: w <- w - lr * g / sqrt(G + eps), where G is the running
    sum of squared gradients. Weights that rarely get gradient (rare words) keep larger steps."""

    def __init__(self, lr, eps=1e-9):
        self.lr = lr
        self.eps = eps
        self.G = {}

    def step(self, params, grads):
        for k in params:
            self.G[k] = self.G.get(k, 0.0) + grads[k] ** 2
            params[k] -= self.lr * grads[k] / np.sqrt(self.G[k] + self.eps)


def fit(model, optimizer, X, y, epochs: int, batch_size: int, seed: int = 0,
        on_epoch_end: Callable[[int], Dict] = None, verbose: bool = True) -> List[Dict]:
    """Shuffled mini-batch training. `on_epoch_end(epoch)` may return extra metrics to log."""
    rng = np.random.default_rng(seed)
    history = []
    n = X.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            loss, grads = model.loss_and_grads(X[idx], y[idx])
            optimizer.step(model.params, grads)
            total += loss * len(idx)
        row = {"epoch": epoch, "train_loss": total / n, **(on_epoch_end(epoch) if on_epoch_end else {})}
        history.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                  flush=True)
    return history


def classification_report(y_true, y_pred, n_classes) -> Dict:
    """Accuracy plus per-class and macro-averaged precision, recall and F1, and the confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(n_classes), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes),
                   where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
    }
