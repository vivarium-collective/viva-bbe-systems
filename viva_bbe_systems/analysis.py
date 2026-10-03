"""Dynamical-systems analysis of CTRNNs: equilibria, stability, nullclines,
bifurcation sweeps. Pure numpy/scipy; no plotting."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import fsolve
from .ctrnn import CTRNN, sigmoid


def equilibria(net: CTRNN, I, n_starts=50, rng=None, tol=1e-8, dedupe_tol=1e-4,
               start_range=20.0):
    I = np.asarray(I, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    found: list[np.ndarray] = []

    def f(y):
        return net.derivatives(y, I)

    starts = [np.zeros(net.size)] + [
        rng.uniform(-start_range, start_range, net.size) for _ in range(n_starts)
    ]
    for y0 in starts:
        with warnings.catch_warnings():
            # non-convergence is handled via ier / residual check below
            warnings.simplefilter("ignore", RuntimeWarning)
            sol, info, ier, _ = fsolve(f, y0, full_output=True)
        if ier != 1:
            continue  # did not converge -> discard
        if np.max(np.abs(f(sol))) > tol:
            continue
        if not any(np.linalg.norm(sol - e) < dedupe_tol for e in found):
            found.append(sol)
    return found


def jacobian(net: CTRNN, y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = sigmoid(y + net.theta)
    dsig = s * (1.0 - s)                      # sigma'(y_j + theta_j)
    J = net.weights * dsig[np.newaxis, :]     # weights[i,j] * dsig[j]
    J = J - np.eye(net.size)
    J = J / net.tau[:, np.newaxis]
    return J


def eigenvalues(net: CTRNN, y) -> np.ndarray:
    return np.linalg.eigvals(jacobian(net, y))


def is_stable(net: CTRNN, y) -> bool:
    return bool(np.all(eigenvalues(net, y).real < 0.0))
