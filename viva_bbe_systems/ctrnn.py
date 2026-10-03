"""Continuous-time recurrent neural network (Beer).

Clean-room reproduction of the CTRNN used across Beer's minimal-cognition
models. State equation (per neuron i):

    tau_i * dy_i/dt = -y_i + sum_j weights[i,j] * sigma(y_j + theta_j) + I_i

with sigma the logistic function and output o_i = sigma(y_i + theta_i).
weights[i,j] is the connection FROM neuron j TO neuron i.
"""
from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def center_crossing_biases(weights: np.ndarray) -> np.ndarray:
    """theta_i = - sum_j weights[i,j] / 2  (Beer's center-crossing condition)."""
    return -0.5 * np.asarray(weights, dtype=float).sum(axis=1)


class CTRNN:
    def __init__(self, size: int, dt: float = 0.01):
        self.size = int(size)
        self.dt = float(dt)
        self.tau = np.ones(self.size)
        self.theta = np.zeros(self.size)
        self.weights = np.zeros((self.size, self.size))
        self.y = np.zeros(self.size)

    def reset(self, y0=None) -> None:
        self.y = np.zeros(self.size) if y0 is None else np.array(y0, dtype=float)

    def outputs(self) -> np.ndarray:
        return sigmoid(self.y + self.theta)

    def derivatives(self, y: np.ndarray, external_input: np.ndarray) -> np.ndarray:
        o = sigmoid(y + self.theta)
        return (-y + self.weights @ o + external_input) / self.tau

    def step(self, external_input=None) -> np.ndarray:
        I = np.zeros(self.size) if external_input is None else np.asarray(external_input, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):  # divergence is reported via the guard below
            self.y = self.y + self.dt * self.derivatives(self.y, I)
        if not np.all(np.isfinite(self.y)):
            raise FloatingPointError("CTRNN state became non-finite (dt too large / unbounded input)")
        return self.outputs()
