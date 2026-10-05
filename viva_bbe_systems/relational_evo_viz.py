"""How the relational-categorization agent improves over evolution:
fitness + accuracy curves and a decision-map progress GIF from checkpoints.

    python -m viva_bbe_systems.relational_evo_viz [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .anim import save_gif
from .tasks.evolve_relational import (load_checkpoints, CHECKPOINT_PATH, relational_spec,
                                      accuracy_report, always_catch_accuracy,
                                      always_avoid_accuracy)
from .bodies.relational_genome import decode_agent
from .relational_gallery import decision_grid, draw_decision_map

NAVY = "#11355e"


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "relational-categorization" / "figures"


def fig_evolution_curves(ck):
    gens, genomes, history = ck["gens"], ck["genomes"], ck["history"]
    spec = relational_spec()
    accs = [accuracy_report(g, spec)["accuracy"] for g in genomes]
    ceiling = max(always_catch_accuracy(), always_avoid_accuracy())
    fig, (axf, axa) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axf.plot(np.arange(len(history)), history, color=NAVY)
    axf.set_xlabel("generation"); axf.set_ylabel("best fitness (shaped)")
    axf.set_title("Fitness over evolution")
    axa.plot(gens, accs, "-o", color="#1f77b4", label="categorization accuracy")
    axa.axhline(ceiling, color="#d62728", ls="--", label=f"memoryless ceiling ({ceiling:.3f})")
    axa.axhline(0.5, color="0.5", ls=":", label="chance (0.5)")
    axa.set_ylim(0.3, 1.0)
    axa.set_xlabel("generation"); axa.set_ylabel("accuracy (best genome)")
    axa.set_title("Relational categorization emerges")
    axf2 = axa.twinx()
    axf2.plot(gens, ck["fitness"], "-s", color=NAVY, alpha=0.4, ms=3, label="fitness")
    axf2.set_ylabel("fitness at checkpoint")
    axa.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def anim_evolution(ck, out_path, n=5):
    gens, genomes = ck["gens"], ck["genomes"]
    spec = relational_spec()
    maps, accs = [], []
    for g in genomes:
        sizes, frac = decision_grid(decode_agent(g, spec), n, offsets=(0.0,))
        maps.append(frac); accs.append(accuracy_report(g, spec)["accuracy"])
    fig, ax = plt.subplots(figsize=(5, 4.8))
    draw_decision_map(ax, sizes, maps[0])
    im = ax.images[0]
    title = ax.set_title("")

    def update(i):
        im.set_data(maps[i])
        title.set_text(f"generation {int(gens[i])}  (accuracy {accs[i]:.3f})\n"
                       "green = catch, red = avoid, dashed = ideal")
        return [im, title]

    fig.tight_layout()
    return save_gif(fig, update, len(gens), out_path, fps=2, dpi=80)


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
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
