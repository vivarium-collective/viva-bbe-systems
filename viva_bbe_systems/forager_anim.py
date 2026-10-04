"""Animated GIF of the action-switching forager (Agmon & Beer 2014) in its world.

Shows the brain-body-environment loop: the agent moving on the 100x100 plane
between two chemical-gradient resources, its chemosensor stalks, its path, and
the two internal nutrient levels as live bars — switching between resources to
keep both nutrients going.

    python -m viva_bbe_systems.forager_anim [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch

from .anim import save_gif
from .tasks.evolve_forager import load_seed, DEFAULT_PATH, forager_spec
from .tasks.forager_fitness import TRIAL_CONFIGS
from .bodies.chemotactic_forager import ChemotacticForager
from .agents.forager_agent import ForagerAgent
from .environments.chemotaxis_resources import ChemotaxisEnv, Resource
from .genome import decode

A_COLOR = "#d62728"   # resource A (red)
B_COLOR = "#2ca02c"   # resource B (green)


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "action-switching" / "figures"


def _resource_disc(ax, center, color):
    cx, cy = center
    # faint exponential-gradient halo (a few translucent rings) + the solid body
    for r, a in ((18, 0.06), (12, 0.10), (7, 0.9)):
        ax.add_patch(CirclePatch((cx, cy), r, fc=color,
                                 ec=(color if r == 7 else "none"),
                                 alpha=a, zorder=1))


def _headings(path):
    """Approximate heading each step from the direction of motion (the agent
    moves along its heading); carry the last good heading when nearly still."""
    n = len(path)
    h = np.zeros(n)
    last = 0.0
    for t in range(n):
        if t + 1 < n:
            d = path[t + 1] - path[t]
        else:
            d = path[t] - path[t - 1] if t > 0 else np.array([1.0, 0.0])
        if np.hypot(*d) > 1e-6:
            last = float(np.arctan2(d[1], d[0]))
        h[t] = last
    return h


def animate_forage(agent, config, out_path, *, max_steps=2500, stride=None,
                   n_frames=140, sensor_dist=6.0):
    env = ChemotaxisEnv(
        Resource(center=np.array(config["resource_a"], float), signal="A"),
        Resource(center=np.array(config["resource_b"], float), signal="B"))
    r = agent.run_trial(env, config["init_levels"], start_pos=config["start_pos"],
                        start_angle=config["start_angle"], max_steps=max_steps)
    path = r["path"]
    levels = r["levels_hist"]
    T = len(path)
    headings = _headings(path)
    sensors = agent.body.sensor_layout()  # [(angle_offset, signal)]
    if stride is None:
        stride = max(1, T // n_frames)
    frames = list(range(0, T, stride))

    fig, (axw, axn) = plt.subplots(
        1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [3, 1]})
    # --- world panel ---
    axw.set_xlim(0, 100); axw.set_ylim(0, 100); axw.set_aspect("equal")
    _resource_disc(axw, config["resource_a"], A_COLOR)
    _resource_disc(axw, config["resource_b"], B_COLOR)
    axw.set_title("Action-switching forager")
    axw.set_xlabel("x"); axw.set_ylabel("y")
    trail, = axw.plot([], [], color="0.4", lw=1.0, alpha=0.8, zorder=2)
    agent_dot, = axw.plot([], [], "o", color="k", ms=9, zorder=4)
    heading_ln, = axw.plot([], [], color="k", lw=1.5, zorder=4)
    stalk_pts = [axw.plot([], [], ".", ms=7,
                          color=(A_COLOR if sig == "A" else B_COLOR), zorder=5)[0]
                 for _, sig in sensors]
    # --- nutrient-bars panel ---
    axn.set_xlim(-0.6, 1.6); axn.set_ylim(0, 10)
    axn.set_xticks([0, 1]); axn.set_xticklabels(["A", "B"])
    axn.set_title("nutrient levels"); axn.set_ylabel("level")
    bars = axn.bar([0, 1], [0, 0], color=[A_COLOR, B_COLOR], width=0.6)
    step_txt = axn.text(0.5, 9.3, "", ha="center", fontsize=9)

    def update(i):
        t = min(i, T - 1)
        p = path[t]
        trail.set_data(path[:t + 1, 0], path[:t + 1, 1])
        agent_dot.set_data([p[0]], [p[1]])
        h = headings[t]
        heading_ln.set_data([p[0], p[0] + 4 * np.cos(h)], [p[1], p[1] + 4 * np.sin(h)])
        for pt, (off, _sig) in zip(stalk_pts, sensors):
            sx = p[0] + sensor_dist * np.cos(h + off)
            sy = p[1] + sensor_dist * np.sin(h + off)
            pt.set_data([sx], [sy])
        for bar, lv in zip(bars, levels[t]):
            bar.set_height(lv)
        step_txt.set_text(f"t={t}")
        return [trail, agent_dot, heading_ln, *stalk_pts, *bars, step_txt]

    fig.tight_layout()
    return save_gif(fig, update, frames, out_path, fps=20, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    g = load_seed(DEFAULT_PATH)
    agent = ForagerAgent(decode(g, forager_spec("M2")), ChemotacticForager("M2"))
    # pick a config where the switching reads clearly (config 0: horizontal pair)
    return [animate_forage(agent, TRIAL_CONFIGS[0], outdir / "forage.gif")]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
