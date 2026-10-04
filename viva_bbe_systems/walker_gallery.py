"""Dynamical-analysis figures for the legged-locomotion walker: the CPG limit
cycle and the gait diagram.

    python -m viva_bbe_systems.walker_gallery [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .agents.walker_agent import make_walker
from .genome import decode
from .tasks.evolve_walker import load_seed, walker_spec, gait_metrics

NAVY = "#11355e"
ORANGE = "#d95f02"


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "legged-locomotion" / "figures"


def fig_cpg_limit_cycle(agent, *, steps=500, transient=100):
    r = agent.run_trial(steps=steps, record=True)
    o = r["outputs"]
    skip = min(transient, len(o) // 4)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (i, j, nm) in ((a1, (-2, -1, ("BS", "FS"))), (a2, (-3, -1, ("foot", "FS")))):
        ax.plot(o[:skip + 1, i], o[:skip + 1, j], color="0.7", lw=1, label="transient")
        seg = o[skip:]
        sc = ax.scatter(seg[:, i], seg[:, j], c=np.arange(len(seg)), cmap="viridis", s=6, zorder=3)
        ax.plot(seg[:, i], seg[:, j], color=NAVY, lw=0.8, alpha=0.5)
        ax.set_xlabel(f"{nm[0]} output"); ax.set_ylabel(f"{nm[1]} output")
        ax.set_title(f"Phase portrait: {nm[0]} vs {nm[1]}")
        ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=[a1, a2], label="time step (after transient)", fraction=0.025)
    a1.annotate("closed loop = rhythmic CPG attractor\n(limit cycle)", xy=(0.04, 0.04),
                xycoords="axes fraction", fontsize=9, color=NAVY,
                bbox=dict(boxstyle="round", fc="white", ec=NAVY))
    fig.suptitle("The CPG's limit cycle")
    return fig


def fig_gait_diagram(agent, *, steps=500, window=(100, 300)):
    r = agent.run_trial(steps=steps)
    m = gait_metrics(r["foot_hist"], r["vx_hist"])
    a, b = window[0], min(window[1], steps)
    t = np.arange(a, b)
    foot = r["foot_hist"][a:b]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax in axes:
        ax.fill_between(t, 0, 1, where=foot, transform=ax.get_xaxis_transform(),
                        color="#cfe3f5", step="mid", lw=0)
    axes[0].fill_between(t, 0, foot.astype(float), step="mid", color=NAVY)
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["swing", "stance"])
    axes[0].set_ylabel("foot")
    axes[1].plot(t, r["angle_hist"][a:b], color=ORANGE); axes[1].set_ylabel("leg angle")
    axes[2].plot(t, r["vx_hist"][a:b], color="#2ca02c"); axes[2].set_ylabel("body velocity")
    axes[2].set_xlabel("time step")
    axes[0].set_title(f"Gait diagram  |  stride period {m['stride_period']:.1f} steps, "
                      f"duty factor {m['duty_factor']:.2f}, mean velocity {m['mean_velocity']:.2f}")
    fig.text(0.99, 0.01, "shaded = stance (foot down)", ha="right", fontsize=8, color="0.4")
    fig.tight_layout()
    return fig


def render(outdir: Path) -> list[Path]:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    agent = make_walker(decode(load_seed(), walker_spec()))
    out = []
    for name, fn in (("cpg_limit_cycle.png", fig_cpg_limit_cycle),
                     ("gait_diagram.png", fig_gait_diagram)):
        fig = fn(agent)
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); out.append(p)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
