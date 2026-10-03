"""Dynamical-systems visualizations (Beer's signature plots). AI-free.

Plain functions returning matplotlib Figures so they are unit-testable and
usable from a gallery script or /viva-viz wiring. No dashboard decorator."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import equilibria, nullcline_grid, bifurcation_sweep, is_stable


def phase_portrait_2d(net, I, y_range=(-10, 10), resolution=25):
    I = np.asarray(I, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    axis = np.linspace(y_range[0], y_range[1], resolution)
    Y1, Y2 = np.meshgrid(axis, axis)
    U = np.zeros_like(Y1); V = np.zeros_like(Y2)
    for a in range(resolution):
        for b in range(resolution):
            d = net.derivatives(np.array([Y1[a, b], Y2[a, b]]), I)
            U[a, b], V[a, b] = d
    ax.quiver(Y1, Y2, U, V, color="0.6", pivot="mid")
    for neuron, color in [(0, "#1f77b4"), (1, "#d62728")]:
        G1, G2, Z = nullcline_grid(net, I, neuron, y_range, resolution=max(resolution, 60))
        ax.contour(G1, G2, Z, levels=[0.0], colors=[color], linewidths=1.5)
    for e in equilibria(net, I, rng=np.random.default_rng(0)):
        stable = is_stable(net, e)
        ax.plot(e[0], e[1], "o", mfc=("k" if stable else "none"), mec="k", ms=9)
    ax.set_xlabel("y1"); ax.set_ylabel("y2"); ax.set_title("CTRNN phase portrait")
    return fig


def bifurcation_diagram(net_factory, param_values, param_name, I=0.0):
    sweep = bifurcation_sweep(net_factory, list(param_values), I=I,
                              rng=np.random.default_rng(0))
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in sweep:
        for e, st in zip(row["equilibria"], row["stability"]):
            ax.plot(row["value"], e[0], ".",
                    color=("k" if st else "r"), ms=6)
    ax.set_xlabel(param_name); ax.set_ylabel("equilibrium y1")
    ax.set_title("Bifurcation diagram (black=stable, red=unstable)")
    return fig
