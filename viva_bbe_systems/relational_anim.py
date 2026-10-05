"""Animated GIFs of the relational-categorization agent (Williams/Beer/Gasser 2008):
side-view catch/avoid trials, and neural activity across object 1 | ISI | object 2.

    python -m viva_bbe_systems.relational_anim [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .anim import save_gif
from .tasks.evolve_relational import load_seed, relational_spec
from .bodies.relational_genome import decode_agent
from .environments.object_stream import TwoObjectStream

OBJ_COLOR = "#e07b00"
CATCH_COLOR = "#2ca02c"
AVOID_COLOR = "#d62728"


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "relational-categorization" / "figures"


def _phase_bounds(stream):
    p1 = stream.phase1_steps
    return p1, p1 + stream.isi_steps, stream.total_steps


def animate_relational(agent, out_path, *, pairs=((3.0, 5.0), (5.0, 3.0)), dt=0.1,
                       n_frames=160, offset2=0.0):
    runs = []
    for s1, s2 in pairs:
        r = agent.run_trial(s1, s2, offset2=offset2, dt=dt, record=True)
        runs.append((s1, s2, r, TwoObjectStream(s1, s2, offset2=offset2)))
    T = runs[0][3].total_steps
    frames = sorted(set(np.linspace(0, T - 1, n_frames).astype(int).tolist()))
    xs = np.concatenate([r["x_hist"] for _, _, r, _ in runs])
    xlim = max(12.0, np.abs(xs).max() + 4, offset2 + 8)
    ymax = 20 + 7

    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 5.6))
    axes = np.atleast_1d(axes)
    arts = []
    for ax, (s1, s2, r, st) in zip(axes, runs):
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-2.5, ymax); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color="0.2", lw=2)
        ax.add_patch(Rectangle((-xlim, -2.5), 2 * xlim, 2.5, color="0.9", zorder=0))
        circ = Circle((0, 0), 1, fc=OBJ_COLOR, ec="k", zorder=3)
        ax.add_patch(circ)
        catcher = Rectangle((-1.5, -0.1), 3.0, 1.2, fc="#11355e", ec="k", zorder=4)
        ax.add_patch(catcher)
        lbl = ax.text(0, ymax - 0.8, "", ha="center", va="top", fontsize=11)
        title = ax.set_title("", fontsize=11)
        want = "should CATCH (s2 > s1)" if s2 > s1 else "should AVOID (s2 < s1)"
        ax.set_xlabel(f"s1 = {s1:g}, s2 = {s2:g}: {want}")
        arts.append((circ, catcher, lbl, title, r, st, s1, s2))

    def update(t):
        changed = []
        for circ, catcher, lbl, title, r, st, s1, s2 in arts:
            ph = st.phase(t)
            obj = st.visible(t)
            if obj is not None:
                size = s1 if ph == "obj1" else s2
                k = t if ph == "obj1" else t - st.phase1_steps - st.isi_steps
                yc = max(st.H - st.vy * k, size)     # rests on the ground once landed
                circ.set_center((obj.center[0], yc)); circ.set_radius(size)
                circ.set_visible(True)
            else:
                circ.set_visible(False)
            catcher.set_x(r["x_hist"][t] - 1.5)
            if ph == "obj1":
                lbl.set_text(f"object 1, size {s1:g}")
            elif ph == "isi":
                lbl.set_text("remembering…")
            else:
                lbl.set_text(f"object 2, size {s2:g}")
            if t >= T - 1 or t >= T - 15:
                verdict = "CATCH" if r["caught"] else "AVOID"
                ok = "correct" if r["correct"] else "INCORRECT"
                title.set_text(f"{verdict} - {ok}")
                title.set_color(CATCH_COLOR if r["correct"] else AVOID_COLOR)
            else:
                title.set_text("t = %d" % t); title.set_color("k")
            changed += [circ, catcher, lbl, title]
        return changed

    fig.suptitle("Relational categorization: catch object 2 iff it is larger than object 1")
    fig.tight_layout()
    return save_gif(fig, update, frames, out_path, fps=20, dpi=80)


def _labels(agent, n):
    inputs = {j for j in range(n) if np.abs(agent.sensor_weights[j]).max() > 1.0}
    labs = []
    for j in range(n):
        if j in [k % n for k in agent.motor_indices]:
            labs.append(f"{j} motor")
        elif j in inputs:
            labs.append(f"{j} in")
        else:
            labs.append(f"{j} inter")
    return labs


def animate_neural(agent, out_path, *, s1=3.0, s2=5.0, n_frames=140):
    r = agent.run_trial(s1, s2, record=True)
    outs = r["outputs"]
    T, N = outs.shape
    W = agent.ctrnn.weights
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    px, py = np.cos(ang), np.sin(ang)
    labels = _labels(agent, N)
    st = TwoObjectStream(s1, s2)
    b1, b2, _ = _phase_bounds(st)

    fig, (axg, axr) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    wmax = np.abs(W).max() or 1.0
    for i in range(N):
        for j in range(N):
            w = W[i, j]
            if abs(w) < 0.15 * wmax or i == j:
                continue
            axg.plot([px[j], px[i]], [py[j], py[i]],
                     color=("#1f77b4" if w > 0 else "#d62728"),
                     lw=0.4 + 2.0 * abs(w) / wmax, alpha=0.25, zorder=1)
    nodes = axg.scatter(px, py, s=430, c=outs[0], cmap="viridis", vmin=0, vmax=1,
                        edgecolors="k", zorder=3)
    for i, lb in enumerate(labels):
        axg.annotate(lb, (px[i] * 1.3, py[i] * 1.3), ha="center", va="center", fontsize=8)
    axg.set_xlim(-1.6, 1.6); axg.set_ylim(-1.6, 1.6); axg.set_aspect("equal"); axg.axis("off")
    axg.set_title("CTRNN activation (node = neuron output)")
    phase_txt = axg.text(0, 0, "", ha="center", va="center", fontsize=11)
    fig.colorbar(nodes, ax=axg, fraction=0.045, label="output")

    im = axr.imshow(outs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                    extent=[0, T, N - 0.5, -0.5], interpolation="nearest")
    axr.axvspan(b1, b2, color="#ffe9a8", alpha=0.35, zorder=2)
    for b in (b1, b2):
        axr.axvline(b, color="w", ls="--", lw=1.2)
    for x, lb in (((0 + b1) / 2, f"object 1 (s1={s1:g})"), ((b1 + b2) / 2, "ISI"),
                  ((b2 + T) / 2, f"object 2 (s2={s2:g})")):
        axr.text(x, -0.7, lb, ha="center", va="bottom", fontsize=9)
    axr.set_yticks(range(N)); axr.set_yticklabels(labels, fontsize=8)
    axr.set_xlabel("time step"); axr.set_title("Neuron outputs over time", pad=18)
    sweep = axr.axvline(0, color="r", lw=1.4)
    fig.colorbar(im, ax=axr, fraction=0.045, label="output")

    frames = sorted(set(np.linspace(0, T - 1, n_frames).astype(int).tolist()))

    def update(i):
        nodes.set_array(outs[i])
        sweep.set_xdata([i, i])
        phase_txt.set_text({"obj1": "object 1", "isi": "ISI\n(memory)",
                            "obj2": "object 2"}.get(st.phase(i), ""))
        return [nodes, sweep, phase_txt]

    fig.tight_layout()
    return save_gif(fig, update, frames, out_path, fps=20, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    agent = decode_agent(load_seed(), relational_spec())
    return [animate_relational(agent, outdir / "relational.gif"),
            animate_neural(agent, outdir / "neural_activity.gif")]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
