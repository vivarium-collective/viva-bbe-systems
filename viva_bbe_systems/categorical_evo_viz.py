"""Visualize how the categorical-perception agent improves over evolution.

From the per-generation checkpoints (data/genomes/categorical_evolution.npz):
- fitness + behavioral-separation curves (circle distance falls, diamond rises),
- an animated GIF of the best agent's catch trajectory improving across generations.

    python -m viva_bbe_systems.categorical_evo_viz [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .anim import save_gif
from .bodies.categorical_genome import CatGenomeSpec, decode_agent
from .tasks.evolve_categorical import load_checkpoints, CHECKPOINT_PATH

CIRCLE = "#1f77b4"
DIAMOND = "#d62728"
OFFSETS = (-6.0, -3.0, 0.0, 3.0, 6.0)


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "active-categorical-perception" / "figures"


def _mean_fd(genome, spec, shape):
    ds = []
    for off in OFFSETS:
        a = decode_agent(genome, spec, dt=0.1)
        ds.append(a.run_trial(obj_offset=off, shape=shape, steps=200)["final_distance"])
    return float(np.mean(ds))


def _circle_trajectory(genome, spec, offset=3.0, steps=200):
    a = decode_agent(genome, spec, dt=0.1)
    return a.run_trial(obj_offset=offset, shape="circle", steps=steps)["trajectory"]


def fig_evolution_curves(ck, spec=None):
    spec = spec or CatGenomeSpec()
    gens, genomes, history = ck["gens"], ck["genomes"], ck["history"]
    circ = [_mean_fd(g, spec, "circle") for g in genomes]
    diam = [_mean_fd(g, spec, "diamond") for g in genomes]
    fig, (axf, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
    axf.plot(np.arange(len(history)), history, color="#11355e")
    axf.set_xlabel("generation"); axf.set_ylabel("best fitness")
    axf.set_title("Fitness over evolution")
    axb.plot(gens, circ, "-o", color=CIRCLE, label="circle distance (catch → ↓)")
    axb.plot(gens, diam, "-s", color=DIAMOND, label="diamond distance (avoid → ↑)")
    axb.axhline(1.5, color="0.6", ls="--", lw=1, label="catch radius")
    axb.set_xlabel("generation"); axb.set_ylabel("mean final distance")
    axb.set_title("Category separation emerges over evolution"); axb.legend()
    fig.tight_layout()
    return fig


def anim_evolution(ck, path, spec=None, *, offset=3.0, steps=200):
    """GIF: the best agent's catch-circle trajectory improving across generations."""
    spec = spec or CatGenomeSpec()
    gens, genomes = ck["gens"], ck["genomes"]
    trajs = [_circle_trajectory(g, spec, offset, steps) for g in genomes]
    t = np.arange(steps) * 0.1

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(offset, color="0.6", ls=":", lw=1.2, label="circle position")
    ax.set_xlim(0, t[-1])
    allx = np.concatenate([tr[:, 0] for tr in trajs])
    ax.set_ylim(min(allx.min(), offset) - 2, max(allx.max(), offset) + 2)
    ax.set_xlabel("time"); ax.set_ylabel("agent horizontal position")
    ax.legend(loc="upper left")
    line, = ax.plot([], [], color=CIRCLE, lw=2)
    title = ax.set_title("")

    def update(i):
        line.set_data(t, trajs[i][:, 0])
        fd = abs(trajs[i][-1, 0] - offset)
        title.set_text(f"generation {int(gens[i])}: final distance to circle = {fd:.2f}")
        return [line, title]

    # ~0.7 s per generation so each is readable (fps 1.4 over the checkpoints)
    return save_gif(fig, update, len(gens), path, fps=1.4, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoints(CHECKPOINT_PATH)
    written = []
    fig = fig_evolution_curves(ck)
    p = outdir / "evolution_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); written.append(p)
    written.append(anim_evolution(ck, outdir / "evolution_progress.gif"))
    return written


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
