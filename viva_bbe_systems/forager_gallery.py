"""Dynamical-analysis figures for the action-switching investigation (Agmon & Beer 2014).

From the committed seed, via the plain-Python agent loop:
- spatial trajectory + nutrient-level time series (the switching behaviour),
- the nutrient-A vs nutrient-B phase plot (the switching cycle in internal state),
- distance-to-each-resource over time with the switch points marked (the paper's
  Fig-12-style reading of when the agent engages each resource).

    python -m viva_bbe_systems.forager_gallery [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tasks.evolve_forager import load_seed, DEFAULT_PATH, forager_spec
from .tasks.forager_fitness import TRIAL_CONFIGS
from .bodies.chemotactic_forager import ChemotacticForager
from .agents.forager_agent import ForagerAgent
from .environments.chemotaxis_resources import ChemotaxisEnv, Resource
from .genome import decode

A_COLOR = "#d62728"
B_COLOR = "#2ca02c"


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "action-switching" / "figures"


def _run(agent, config, max_steps=2500):
    a = np.array(config["resource_a"], float)
    b = np.array(config["resource_b"], float)
    env = ChemotaxisEnv(Resource(center=a, signal="A"), Resource(center=b, signal="B"))
    r = agent.run_trial(env, config["init_levels"], start_pos=config["start_pos"],
                        start_angle=config["start_angle"], max_steps=max_steps)
    r["res_a"], r["res_b"] = a, b
    return r


def fig_trajectory_and_nutrients(agent, config):
    r = _run(agent, config)
    path, levels = r["path"], r["levels_hist"]
    t = np.arange(len(path))
    fig, (axp, axn) = plt.subplots(1, 2, figsize=(11, 4.6))
    # spatial path
    axp.add_patch(plt.Circle(r["res_a"], 7, fc=A_COLOR, ec="k", alpha=0.85))
    axp.add_patch(plt.Circle(r["res_b"], 7, fc=B_COLOR, ec="k", alpha=0.85))
    axp.plot(path[:, 0], path[:, 1], color="0.3", lw=1.0)
    axp.plot(*path[0], "ko", ms=6)
    axp.set_xlim(0, 100); axp.set_ylim(0, 100); axp.set_aspect("equal")
    axp.set_title("Spatial trajectory: shuttling between resources")
    axp.set_xlabel("x"); axp.set_ylabel("y")
    # nutrient time series
    axn.plot(t, levels[:, 0], color=A_COLOR, label="nutrient A")
    axn.plot(t, levels[:, 1], color=B_COLOR, label="nutrient B")
    axn.set_xlabel("time"); axn.set_ylabel("nutrient level"); axn.set_ylim(0, 10)
    axn.set_title("Internal nutrients sustained by switching"); axn.legend()
    fig.tight_layout()
    return fig


def fig_nutrient_phase(agent, config):
    r = _run(agent, config)
    lv = r["levels_hist"]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    sc = ax.scatter(lv[:, 0], lv[:, 1], c=np.arange(len(lv)), cmap="viridis", s=6)
    ax.plot(lv[:, 0], lv[:, 1], color="0.7", lw=0.6, zorder=0)
    ax.set_xlabel("nutrient A"); ax.set_ylabel("nutrient B")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title("Internal-state phase plot (the switching cycle)")
    fig.colorbar(sc, ax=ax, label="time")
    fig.tight_layout()
    return fig


def fig_action_switching(agent, config):
    """Distance to each resource over time — the paper's Fig-12-style reading of
    when the agent engages (switches to) each resource, with nutrient context."""
    r = _run(agent, config)
    path, lv = r["path"], r["levels_hist"]
    t = np.arange(len(path))
    da = np.linalg.norm(path - r["res_a"], axis=1)
    db = np.linalg.norm(path - r["res_b"], axis=1)
    # "engaged with X" when within the resource radius (7)
    eng_a = da <= 7.0
    eng_b = db <= 7.0
    fig, (axd, axn) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axd.plot(t, da, color=A_COLOR, label="distance to A")
    axd.plot(t, db, color=B_COLOR, label="distance to B")
    axd.fill_between(t, 0, da.max(), where=eng_a, color=A_COLOR, alpha=0.12, step="mid")
    axd.fill_between(t, 0, da.max(), where=eng_b, color=B_COLOR, alpha=0.12, step="mid")
    axd.axhline(7, color="0.6", ls="--", lw=0.8)
    axd.set_ylabel("distance to resource"); axd.legend(loc="upper right")
    axd.set_title("Action switching: shaded bands = engaged with that resource")
    axn.plot(t, lv[:, 0], color=A_COLOR, label="nutrient A")
    axn.plot(t, lv[:, 1], color=B_COLOR, label="nutrient B")
    axn.set_xlabel("time"); axn.set_ylabel("nutrient level"); axn.set_ylim(0, 10)
    axn.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_morphologies():
    """The three morphologies' sensor layouts (Agmon & Beer 2014, Fig. 2):
    M1 two side stalks + nutrient sensors, M2 one front stalk + nutrient sensors,
    M3 two stalks and NO nutrient sensors."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, m, sub in zip(axes, ("M1", "M2", "M3"),
                          ("two side stalks + nutrient sensors",
                           "one front stalk + nutrient sensors",
                           "two stalks, NO nutrient sensors")):
        body = ChemotacticForager(m)
        ax.add_patch(plt.Circle((0, 0), 1.2, fc="0.85", ec="k"))
        ax.annotate("", xy=(0, 3), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="0.4"))  # heading (up)
        for off, sig in body.sensor_layout():
            # heading is +y here (angle shown as 0 → up): rotate offsets accordingly
            sx, sy = body.sensor_dist * -np.sin(off), body.sensor_dist * np.cos(off)
            ax.plot([0, sx], [0, sy], color="0.6", lw=1)
            ax.plot(sx, sy, "o", ms=9, color=(A_COLOR if sig == "A" else B_COLOR))
        if body.n_nutrient:
            ax.plot([-0.6, 0.6], [-0.6, -0.6], "s", ms=9, color="#8c564b")
            ax.text(0, -1.9, "nutrient sensors", ha="center", fontsize=8, color="#8c564b")
        ax.set_xlim(-9, 9); ax.set_ylim(-4, 9); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{m}\n{sub}", fontsize=10)
    fig.suptitle("Three morphologies (red = A-chemosensor, green = B-chemosensor)", y=1.02)
    fig.tight_layout()
    return fig


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    agent = ForagerAgent(decode(load_seed(DEFAULT_PATH), forager_spec("M2")),
                         ChemotacticForager("M2"))
    config = TRIAL_CONFIGS[0]
    written = []
    mfig = fig_morphologies()
    mp = outdir / "morphologies.png"
    mfig.savefig(mp, dpi=150, bbox_inches="tight"); plt.close(mfig); written.append(mp)
    for name, fn in (("trajectory_nutrients.png", fig_trajectory_and_nutrients),
                     ("nutrient_phase.png", fig_nutrient_phase),
                     ("action_switching.png", fig_action_switching)):
        fig = fn(agent, config)
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
