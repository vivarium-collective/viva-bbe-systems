"""Helpers for the CTRNN parameter-space investigation (Beer 2006, 2022)."""
from __future__ import annotations
import numpy as np
from .ctrnn import CTRNN, center_crossing_biases
from .analysis import equilibria


def single_neuron_net(self_weight):
    net = CTRNN(1)
    net.weights[:] = np.array([[self_weight]])
    net.theta[:] = center_crossing_biases(net.weights)
    return net


def two_neuron_net(w, center_crossing=True):
    net = CTRNN(2)
    net.weights[:] = np.asarray(w, dtype=float)
    if center_crossing:
        net.theta[:] = center_crossing_biases(net.weights)
    return net


def codim2_equilibria_count(p1_values, p2_values, make_net, I=0.0):
    """Grid of equilibrium counts over two parameters; make_net(p1, p2) -> CTRNN."""
    rng = np.random.default_rng(0)
    grid = np.zeros((len(p1_values), len(p2_values)), dtype=int)
    for a, p1 in enumerate(p1_values):
        for b, p2 in enumerate(p2_values):
            net = make_net(p1, p2)
            Ivec = np.full(net.size, I)
            grid[a, b] = len(equilibria(net, Ivec, n_starts=80, rng=rng))
    return grid
