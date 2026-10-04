"""Render the CTRNN parameter-space investigation's signature figures.

Beer's dynamical-systems analyses of small CTRNNs (Beer 2006, 2022), produced
from the workspace's own `analysis`/`viz`/`param_space` code:

- a 2-neuron phase portrait with nullclines + equilibria,
- a codim-1 bifurcation diagram (single self-excitatory neuron),
- a codim-2 equilibrium-count map over (self-weight, bias).

    python -m viva_bbe_systems.gallery [OUTDIR]

writes <OUTDIR>/{phase_portrait.png, bifurcation.png, codim2.png}. OUTDIR
defaults to the ctrnn-parameter-space investigation's figures/ dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ctrnn import CTRNN
from .analysis import nullcline_grid
from .param_space import single_neuron_net, two_neuron_net, codim2_equilibria_count
from .viz import phase_portrait_2d, bifurcation_diagram


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "ctrnn-parameter-space" / "figures"


def codim2_heatmap(w_values, b_values):
    """Parameter-plane map: number of equilibria over (self-weight, bias)."""
    grid = codim2_equilibria_count(
        w_values, b_values,
        lambda w, b: _single_neuron_with_bias(w, b),
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(
        grid.T, origin="lower", aspect="auto", cmap="viridis",
        extent=[w_values[0], w_values[-1], b_values[0], b_values[-1]],
    )
    cbar = fig.colorbar(im, ax=ax, ticks=sorted(set(grid.reshape(-1).tolist())))
    cbar.set_label("number of equilibria")
    ax.set_xlabel("self-weight w")
    ax.set_ylabel("bias θ")
    ax.set_title("Codim-2 equilibrium-count map (single neuron)")
    return fig


def _single_neuron_with_bias(self_weight, bias):
    net = CTRNN(1)
    net.weights[:] = np.array([[self_weight]])
    net.theta[:] = [bias]
    return net


def animate_phase_trajectory(net, path, *, y0=(0.1, -0.1), steps=1200, stride=6,
                             y_range=(-6, 10)):
    """GIF (brain-only 'video'): a state trajectory tracing the 2-neuron
    oscillator's limit cycle, over the nullclines + vector field."""
    from .anim import save_gif
    net.reset(np.array(y0, dtype=float))
    traj = np.zeros((steps, 2))
    for t in range(steps):
        net.step()
        traj[t] = net.y
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    axis = np.linspace(*y_range, 22)
    Y1, Y2 = np.meshgrid(axis, axis)
    U = np.zeros_like(Y1); V = np.zeros_like(Y2)
    for a in range(Y1.shape[0]):
        for b in range(Y1.shape[1]):
            d = net.derivatives(np.array([Y1[a, b], Y2[a, b]]), np.zeros(2))
            U[a, b], V[a, b] = d
    ax.quiver(Y1, Y2, U, V, color="0.8", pivot="mid")
    for neuron, color in [(0, "#1f77b4"), (1, "#d62728")]:
        G1, G2, Z = nullcline_grid(net, np.zeros(2), neuron, y_range, resolution=80)
        ax.contour(G1, G2, Z, levels=[0.0], colors=[color], linewidths=1.3)
    ax.set_xlim(*y_range); ax.set_ylim(*y_range)
    ax.set_xlabel("y1"); ax.set_ylabel("y2")
    ax.set_title("CTRNN state trajectory → limit cycle")
    line, = ax.plot([], [], color="#11355e", lw=1.5)
    head, = ax.plot([], [], "o", color="#11355e", ms=7)
    frames = list(range(2, steps, stride))

    def update(i):
        line.set_data(traj[:i, 0], traj[:i, 1])
        head.set_data([traj[i-1, 0]], [traj[i-1, 1]])
        return [line, head]

    return save_gif(fig, update, frames, path, fps=24, dpi=80)


def render(outdir: Path, *, phase_res=25, bifurcation_points=40,
           codim2_res=40, animate=True) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Phase portrait of a 2-neuron center-crossing oscillator.
    net = two_neuron_net(np.array([[4.5, 1.0], [-1.0, 4.5]]))
    fig = phase_portrait_2d(net, np.zeros(2), y_range=(-10, 10), resolution=phase_res)
    p = outdir / "phase_portrait.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2. Codim-1 bifurcation: single self-excitatory neuron, 1 -> 3 equilibria.
    fig = bifurcation_diagram(
        lambda w: single_neuron_net(w),
        np.linspace(0.5, 10.0, bifurcation_points),
        "self-weight w",
    )
    p = outdir / "bifurcation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3. Codim-2 equilibrium-count map over (self-weight, bias).
    fig = codim2_heatmap(np.linspace(0.5, 12.0, codim2_res),
                         np.linspace(-8.0, 2.0, codim2_res))
    p = outdir / "codim2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 4. Animated phase-space trajectory (brain-only 'video' of the dynamics).
    if animate:
        net_osc = two_neuron_net(np.array([[4.5, 1.0], [-1.0, 4.5]]))
        written.append(animate_phase_trajectory(net_osc, outdir / "phase_trajectory.gif"))

    return written


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    written = render(outdir)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
