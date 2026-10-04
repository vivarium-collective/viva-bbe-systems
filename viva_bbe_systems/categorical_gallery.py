"""Signature figures for the active-categorical-perception investigation (Beer 2003).

Rendered from the committed seed genome via the plain-Python agent loop:
- catch/avoid trajectories (agent vs object horizontal position over time),
- the categorization map (final distance vs object offset, circle vs diamond),
- decision dynamics (CTRNN neuron-output trajectories during a catch vs an avoid).

    python -m viva_bbe_systems.categorical_gallery [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bodies.categorical_genome import CatGenomeSpec, decode_agent
from .tasks.evolve_categorical import load_seed, DEFAULT_PATH

CIRCLE = "#1f77b4"
DIAMOND = "#d62728"
OFFSETS = (-6.0, -3.0, 0.0, 3.0, 6.0)


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "active-categorical-perception" / "figures"


def run_recorded(agent, offset, shape, *, H=20.0, vy=1.0, dt=0.1, steps=200, obj_size=3.0):
    """Run one trial, recording per-step agent_x, obj_x, and CTRNN neuron outputs."""
    r = agent.run_trial(offset, shape, H=H, vy=vy, dt=dt, steps=steps,
                        obj_size=obj_size, record_outputs=True)
    return {"t": np.arange(steps) * dt, "agent_x": r["trajectory"][:, 0],
            "obj_x": r["trajectory"][:, 1], "outputs": r["outputs"]}


def fig_trajectories(agent):
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, shape, color, title in ((axc, "circle", CIRCLE, "Circles (catch)"),
                                    (axd, "diamond", DIAMOND, "Diamonds (avoid)")):
        for off in OFFSETS:
            r = run_recorded(agent, off, shape)
            ax.plot(r["t"], r["agent_x"], color=color, alpha=0.9)
            ax.axhline(off, color="0.7", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("time")
    axc.set_ylabel("horizontal position")
    fig.suptitle("Active categorical perception: agent tracks circles, flees diamonds", y=1.02)
    fig.tight_layout()
    return fig


def fig_categorization_map(agent):
    offs = np.linspace(-8, 8, 33)
    finals = {"circle": [], "diamond": []}
    for shape in ("circle", "diamond"):
        for off in offs:
            r = run_recorded(agent, off, shape)
            finals[shape].append(abs(r["agent_x"][-1] - r["obj_x"][-1]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(offs, finals["circle"], "-o", color=CIRCLE, ms=4, label="circle (catch)")
    ax.plot(offs, finals["diamond"], "-s", color=DIAMOND, ms=4, label="diamond (avoid)")
    ax.axhline(1.5, color="0.6", ls="--", lw=1, label="catch radius")
    ax.set_xlabel("object horizontal offset")
    ax.set_ylabel("final horizontal distance to object")
    ax.set_title("Categorization map: distance separates the two categories")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_decision_dynamics(agent):
    rc = run_recorded(agent, 3.0, "circle")
    rd = run_recorded(agent, 3.0, "diamond")
    n = agent.ctrnn.size
    fig, (axo, axp) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i in range(n):
        axo.plot(rc["t"], rc["outputs"][:, i], color=CIRCLE, alpha=0.6,
                 label="circle" if i == 0 else None)
        axo.plot(rd["t"], rd["outputs"][:, i], color=DIAMOND, alpha=0.6,
                 label="diamond" if i == 0 else None)
    axo.set_xlabel("time"); axo.set_ylabel("neuron output")
    axo.set_title("Neuron outputs diverge by category"); axo.legend()
    # phase projection of the two motor neurons
    m0, m1 = agent.motor_indices
    axp.plot(rc["outputs"][:, m0], rc["outputs"][:, m1], color=CIRCLE, label="circle")
    axp.plot(rd["outputs"][:, m0], rd["outputs"][:, m1], color=DIAMOND, label="diamond")
    axp.set_xlabel(f"motor neuron {m0} output"); axp.set_ylabel(f"motor neuron {m1} output")
    axp.set_title("Motor-neuron phase trajectory (catch vs avoid)"); axp.legend()
    fig.suptitle("Decision dynamics at offset +3: the coupled system commits differently", y=1.02)
    fig.tight_layout()
    return fig


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    agent = decode_agent(load_seed(DEFAULT_PATH), CatGenomeSpec(), dt=0.1)
    written = []
    for name, fn in (("trajectories.png", fig_trajectories),
                     ("categorization_map.png", fig_categorization_map),
                     ("decision_dynamics.png", fig_decision_dynamics)):
        fig = fn(agent)
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
