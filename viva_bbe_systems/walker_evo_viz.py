"""Visualize how the CPG walker improves over evolution (from per-generation
checkpoints): fitness/distance curves and a gait-improvement GIF.

    python -m viva_bbe_systems.walker_evo_viz [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .anim import save_gif
from .agents.walker_agent import make_walker
from .genome import decode
from .tasks.evolve_walker import load_checkpoints, CHECKPOINT_PATH, walker_spec

STEPS = 500


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "legged-locomotion" / "figures"


def _trial(genome):
    return make_walker(decode(genome, walker_spec())).run_trial(steps=STEPS)


def fig_evolution_curves(ck):
    gens, genomes, history = ck["gens"], ck["genomes"], ck["history"]
    dists = [_trial(g)["distance"] for g in genomes]
    fig, (axf, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
    axf.plot(np.arange(len(history)), history, color="#11355e")
    axf.set_xlabel("generation"); axf.set_ylabel("best fitness (shaped)")
    axf.set_title("Fitness over evolution")
    axb.plot(gens, dists, "-o", color="#d95f02")
    axb.set_xlabel("generation"); axb.set_ylabel("distance walked (unshaped)")
    axb.set_title("Distance of best genome per checkpoint")
    fig.tight_layout()
    return fig


def anim_evolution(ck, out_path):
    gens, genomes = ck["gens"], ck["genomes"]
    runs = [_trial(g) for g in genomes]
    fig, (axa, axx) = plt.subplots(2, 1, figsize=(8, 5.4))
    axa.set_xlim(0, STEPS)
    amax = max(1e-3, max(np.abs(r["angle_hist"]).max() for r in runs)) * 1.1
    axa.set_ylim(-amax, amax); axa.set_ylabel("leg angle")
    axx.set_xlim(0, STEPS)
    xmax = max(1.0, max(r["x_hist"].max() for r in runs)) * 1.05
    axx.set_ylim(min(0, min(r["x_hist"].min() for r in runs)), xmax)
    axx.set_ylabel("body x"); axx.set_xlabel("time step")
    la, = axa.plot([], [], color="#d95f02", lw=1.4)
    lx, = axx.plot([], [], color="#11355e", lw=1.8)
    title = axa.set_title("")

    def update(i):
        r = runs[i]
        t = np.arange(len(r["x_hist"]))
        la.set_data(t, r["angle_hist"]); lx.set_data(t, r["x_hist"])
        title.set_text(f"generation {int(gens[i])}  (distance {r['distance']:.1f})")
        return [la, lx, title]

    fig.tight_layout()
    return save_gif(fig, update, len(gens), out_path, fps=1.4, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoints(CHECKPOINT_PATH)
    fig = fig_evolution_curves(ck)
    p = outdir / "evolution_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return [p, anim_evolution(ck, outdir / "evolution_progress.gif")]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
