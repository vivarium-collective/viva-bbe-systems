"""Animated GIFs of the categorical-perception agent solving the task.

Shows the embodied brain-body-environment loop in motion: the horizontal agent,
its fan of 7 ray sensors, and the falling object (circle or diamond), over a
catch trial and an avoid trial, from the committed seed.

    python -m viva_bbe_systems.categorical_anim [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch, Polygon

from .anim import save_gif
from .bodies.categorical_genome import CatGenomeSpec, decode_agent
from .environments.falling_objects import ray_distance
from .tasks.evolve_categorical import load_seed, DEFAULT_PATH

CIRCLE = "#1f77b4"
DIAMOND = "#d62728"


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "active-categorical-perception" / "figures"


def _record(agent, offset, shape, *, H=20.0, vy=1.0, dt=0.1, steps=200, obj_size=3.0):
    r = agent.run_trial(offset, shape, H=H, vy=vy, dt=dt, steps=steps,
                        obj_size=obj_size, record_outputs=True)
    return r["trajectory"][:, 0], r["obj_centers"], obj_size


def _object_patch(center, size, shape, color):
    if shape == "circle":
        return CirclePatch(center, size, fc=color, ec="k", alpha=0.85, zorder=3)
    pts = [center + np.array(v) * size for v in ((1, 0), (0, 1), (-1, 0), (0, -1))]
    return Polygon(pts, closed=True, fc=color, ec="k", alpha=0.85, zorder=3)


def animate_trial(agent, offset, shape, path, *, steps=200, stride=2, H=20.0,
                  obj_size=3.0, max_range=20.0):
    ax_hist, oc_hist, size = _record(agent, offset, shape, steps=steps, H=H, obj_size=obj_size)
    frames = list(range(0, steps, stride))
    color = CIRCLE if shape == "circle" else DIAMOND
    verb = "catches" if shape == "circle" else "avoids"

    xs = np.concatenate([ax_hist, oc_hist[:, 0]])
    xlo, xhi = xs.min() - 4, xs.max() + 4
    width = xhi - xlo
    height = H + 3.5
    fig, ax = plt.subplots(figsize=(min(11, 2 + width * 0.28), 2 + height * 0.22))
    ax.set_xlim(xlo, xhi); ax.set_ylim(-1.5, H + 2)
    ax.set_aspect("equal", adjustable="box")  # circles round, ray angles true
    ax.axhline(0, color="0.3", lw=2)
    ax.set_title(f"Agent {verb} a {shape} (offset {offset:+.0f})", color=color)
    ax.set_xlabel("horizontal position"); ax.set_ylabel("height")

    agent_marker, = ax.plot([], [], "^", color="k", ms=14, zorder=4)
    ray_lines = [ax.plot([], [], color=color, lw=1.0, alpha=0.5)[0] for _ in range(agent.body.n_sensors)]
    patch_holder = {"p": None}
    rays = agent.body.sensor_rays()

    def update(i):
        t = min(i, steps - 1)
        ax_x = ax_hist[t]
        oc = oc_hist[t]
        agent_marker.set_data([ax_x], [0.0])
        origin = np.array([ax_x, 0.0])
        for line, d in zip(ray_lines, rays):
            dist = ray_distance(origin, d, oc, size, shape)
            L = dist if (dist is not None and dist <= max_range) else max_range
            end = origin + d * L
            line.set_data([origin[0], end[0]], [origin[1], end[1]])
        if patch_holder["p"] is not None:
            patch_holder["p"].remove()
        patch_holder["p"] = _object_patch(oc, size, shape, color)
        ax.add_patch(patch_holder["p"])
        return [agent_marker, *ray_lines]

    return save_gif(fig, update, frames, path, fps=20, dpi=80)


def _record_outputs(agent, offset, shape, *, H=20.0, vy=1.0, dt=0.1, steps=200, obj_size=3.0):
    return agent.run_trial(offset, shape, H=H, vy=vy, dt=dt, steps=steps,
                           obj_size=obj_size, record_outputs=True)["outputs"]


def animate_brain_phase(agent, path, *, offset=3.0, steps=200, stride=2):
    """GIF: the brain's dynamical decision — the two motor neurons' output traces
    a path in state space, diverging for catch (circle) vs avoid (diamond)."""
    m0, m1 = agent.motor_indices
    oc = _record_outputs(agent, offset, "circle")
    od = _record_outputs(agent, offset, "diamond")
    frames = list(range(1, steps, stride))
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"motor neuron {m0} output"); ax.set_ylabel(f"motor neuron {m1} output")
    ax.set_title("Brain decision dynamics (catch vs avoid)")
    lc, = ax.plot([], [], color=CIRCLE, lw=1.6, label="circle (catch)")
    ld, = ax.plot([], [], color=DIAMOND, lw=1.6, label="diamond (avoid)")
    hc, = ax.plot([], [], "o", color=CIRCLE, ms=8)
    hd, = ax.plot([], [], "o", color=DIAMOND, ms=8)
    ax.legend(loc="upper right")

    def update(i):
        lc.set_data(oc[:i, m0], oc[:i, m1]); hc.set_data([oc[i-1, m0]], [oc[i-1, m1]])
        ld.set_data(od[:i, m0], od[:i, m1]); hd.set_data([od[i-1, m0]], [od[i-1, m1]])
        return [lc, ld, hc, hd]

    return save_gif(fig, update, frames, path, fps=20, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    agent = decode_agent(load_seed(DEFAULT_PATH), CatGenomeSpec(), dt=0.1)
    written = []
    written.append(animate_trial(agent, 3.0, "circle", outdir / "solve_circle.gif"))
    written.append(animate_trial(agent, 3.0, "diamond", outdir / "solve_diamond.gif"))
    written.append(animate_brain_phase(agent, outdir / "brain_phase.gif"))
    return written


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
