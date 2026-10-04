"""Animated GIFs of the CPG legged walker (Beer): a follow-camera side view and
the CTRNN's neural activity.

    python -m viva_bbe_systems.walker_anim [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .anim import save_gif
from .agents.walker_agent import WALKER_SIZE, make_walker
from .genome import decode
from .tasks.evolve_walker import load_seed, walker_spec

BODY_W, BODY_H, LEG_LEN = 8.0, 3.0, 7.0
VIEW_W = 60.0


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "legged-locomotion" / "figures"


def _labels(n=WALKER_SIZE):
    return ["prop"] + ["inter"] * (n - 4) + ["foot", "BS", "FS"]


def animate_walk(agent, out_path, *, steps=500, n_frames=160):
    r = agent.run_trial(steps=steps)
    xs, ang, foot = r["x_hist"], r["angle_hist"], r["foot_hist"]
    T = len(xs)
    frames = list(range(0, T, max(1, T // n_frames)))
    ground_y = 0.0
    hip_y = ground_y + LEG_LEN * 0.9 + 0.5

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.set_ylim(-3.5, 14)
    ax.set_facecolor("#f4f7fb")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ground = ax.axhline(ground_y, color="#444", lw=2, zorder=2)
    ticks, = ax.plot([], [], "|", color="#888", ms=10, zorder=1)
    body = Rectangle((0, hip_y), BODY_W, BODY_H, fc="#11355e", ec="k", zorder=3)
    ax.add_patch(body)
    leg, = ax.plot([], [], color="#d95f02", lw=3.5, solid_capstyle="round", zorder=4)
    foot_pt, = ax.plot([], [], "o", ms=11, mew=2, color="#d95f02", zorder=5)
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10)
    ax.set_title("Legged CPG walker (follow camera)")
    ax.set_xlabel("x")

    def update(t):
        x = xs[t]
        ax.set_xlim(x - VIEW_W / 2, x + VIEW_W / 2)
        body.set_xy((x - BODY_W / 2, hip_y))
        a = ang[t]
        hx, hy = x, hip_y
        fx, fy = hx + LEG_LEN * np.sin(a), hy - LEG_LEN * np.cos(a)
        leg.set_data([hx, fx], [hy, fy])
        down = bool(foot[t])
        foot_pt.set_data([fx], [fy])
        foot_pt.set_markerfacecolor("#d95f02" if down else "white")
        k0 = np.floor((x - VIEW_W) / 5) * 5
        tk = np.arange(k0, x + VIEW_W, 5)
        ticks.set_data(tk, np.full_like(tk, -1.2))
        txt.set_text(f"step {t}   distance {x:6.1f}   {'stance' if down else 'swing'}")
        return [body, leg, foot_pt, ticks, txt]

    fig.tight_layout()
    return save_gif(fig, update, frames, out_path, fps=20, dpi=80)


def animate_neural(agent, out_path, *, steps=500, n_frames=140):
    r = agent.run_trial(steps=steps, record=True)
    outs = r["outputs"]
    T, N = outs.shape
    W = agent.ctrnn.weights
    labels = _labels(N)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    px, py = np.cos(ang), np.sin(ang)

    fig, (axg, axr) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    wmax = np.abs(W).max() or 1.0
    for i in range(N):
        for j in range(N):
            w = W[i, j]
            if i == j or abs(w) < 0.15 * wmax:
                continue
            axg.plot([px[j], px[i]], [py[j], py[i]],
                     color=("#1f77b4" if w > 0 else "#d62728"),
                     lw=0.4 + 2.0 * abs(w) / wmax, alpha=0.25, zorder=1)
    nodes = axg.scatter(px, py, s=600, c=outs[0], cmap="viridis", vmin=0, vmax=1,
                        edgecolors="k", zorder=3)
    for i, lb in enumerate(labels):
        axg.annotate(lb, (px[i] * 1.3, py[i] * 1.3), ha="center", va="center", fontsize=9)
    axg.set_xlim(-1.6, 1.6); axg.set_ylim(-1.6, 1.6); axg.set_aspect("equal"); axg.axis("off")
    axg.set_title("CTRNN activation (node = neuron output)")
    fig.colorbar(nodes, ax=axg, fraction=0.045, label="output")
    im = axr.imshow(outs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                    extent=[0, T, N - 0.5, -0.5], interpolation="nearest")
    axr.set_yticks(range(N)); axr.set_yticklabels(labels, fontsize=9)
    axr.set_xlabel("time step"); axr.set_title("Neuron outputs over time")
    sweep = axr.axvline(0, color="w", lw=1.2)
    fig.colorbar(im, ax=axr, fraction=0.045, label="output")
    frames = list(range(0, T, max(1, T // n_frames)))

    def update(i):
        nodes.set_array(outs[i])
        sweep.set_xdata([i, i])
        return [nodes, sweep]

    fig.tight_layout()
    return save_gif(fig, update, frames, out_path, fps=20, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    agent = make_walker(decode(load_seed(), walker_spec()))
    return [animate_walk(agent, outdir / "walk.gif"),
            animate_neural(agent, outdir / "neural_activity.gif")]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
