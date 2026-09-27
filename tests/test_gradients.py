"""Check every hand-derived gradient against central finite differences."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnscratch.models import MLP, SoftmaxRegression  # noqa: E402


def numerical_grad(model, X, y, name, h=1e-6):
    param = model.params[name]
    grad = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + h
        plus, _ = model.loss_and_grads(X, y)
        param[i] = old - h
        minus, _ = model.loss_and_grads(X, y)
        param[i] = old
        grad[i] = (plus - minus) / (2 * h)
    return grad


def make_models(n_features, rng):
    models = [
        (SoftmaxRegression(n_features, 3, l2=0.01), rng.integers(0, 3, 12)),
        (MLP(n_features, 5, 3, l2=0.01, seed=1), rng.integers(0, 3, 12)),
    ]
    for model, _ in models:  # move away from the all-zeros init so every term is exercised
        for p in model.params.values():
            p += rng.standard_normal(p.shape) * 0.5
    return models


@pytest.mark.parametrize("use_sparse", [False, True])
def test_gradients_match_finite_differences(use_sparse):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 7))
    if use_sparse:
        X[rng.random(X.shape) < 0.5] = 0.0
        X = sparse.csr_matrix(X)
    for model, y in make_models(7, rng):
        _, grads = model.loss_and_grads(X, y)
        for name in model.params:
            expected = numerical_grad(model, X, y, name)
            rel_err = np.abs(grads[name] - expected).max() / max(1e-8, np.abs(expected).max())
            assert rel_err < 1e-6, f"{type(model).__name__}.{name}: relative error {rel_err:.2e}"
