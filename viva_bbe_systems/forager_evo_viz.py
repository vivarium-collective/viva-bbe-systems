"""Visualize how the action-switching forager improves over evolution.

From the per-generation checkpoints (data/genomes/action_switching_M2_evolution.npz):
- fitness + behavioural curves (mean survival and #configs it forages both in, vs gen),
- an animated GIF of the best agent's spatial trajectory improving across generations
  (early: drifts / camps one resource; late: shuttles between both).

    python -m viva_bbe_systems.forager_evo_viz [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .anim import save_gif
from .tasks.evolve_forager import load_checkpoints, CHECKPOINT_PATH, forager_spec
from .tasks.forager_fitness import TRIAL_CONFIGS
from .bodies.chemotactic_forager import ChemotacticForager
from .agents.forager_agent import ForagerAgent
from .environments.chemotaxis_resources import ChemotaxisEnv, Resource
from .genome import decode

A_COLOR = "#d62728"
B_COLOR = "#2ca02c"
MAX_STEPS = 2500


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "action-switching" / "figures"


def _agent(genome):
    return ForagerAgent(decode(genome, forager_spec("M2")), ChemotacticForager("M2"))


def _behaviour(genome):
    agent = _agent(genome)
    both, survs = 0, []
    for c in TRIAL_CONFIGS:
        env = ChemotaxisEnv(Resource(center=np.array(c["resource_a"], float), signal="A"),
                            Resource(center=np.array(c["resource_b"], float), signal="B"))
        r = agent.run_trial(env, c["init_levels"], start_pos=c["start_pos"],
                            start_angle=c["start_angle"], max_steps=MAX_STEPS)
        lh = r["levels_hist"]
        if float(np.diff(lh[:, 0]).max()) > 0 and float(np.diff(lh[:, 1]).max()) > 0:
            both += 1
        survs.append(r["survival"])
    return float(np.mean(survs)), both


def _trajectory(genome, config):
    agent = _agent(genome)
    env = ChemotaxisEnv(Resource(center=np.array(config["resource_a"], float), signal="A"),
                        Resource(center=np.array(config["resource_b"], float), signal="B"))
    return agent.run_trial(env, config["init_levels"], start_pos=config["start_pos"],
                           start_angle=config["start_angle"], max_steps=MAX_STEPS)["path"]


def fig_evolution_curves(ck):
    gens, genomes, history = ck["gens"], ck["genomes"], ck["history"]
    survs, both = [], []
    for g in genomes:
        s, b = _behaviour(g)
        survs.append(s); both.append(b)
    fig, (axf, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
    axf.plot(np.arange(len(history)), history, color="#11355e")
    axf.set_xlabel("generation"); axf.set_ylabel("best fitness (shaped)")
    axf.set_title("Fitness over evolution")
    axb.plot(gens, survs, "-o", color="#11355e", label="mean survival (steps)")
    axb.set_xlabel("generation"); axb.set_ylabel("mean survival")
    ax2 = axb.twinx()
    ax2.plot(gens, both, "-s", color="#1f77b4", label="#configs foraging both")
    ax2.set_ylabel("#configs foraging both (of 11)"); ax2.set_ylim(0, 11.5)
    axb.set_title("Switching behaviour emerges over evolution")
    lines = axb.get_lines() + ax2.get_lines()
    axb.legend(lines, [l.get_label() for l in lines], loc="lower right")
    fig.tight_layout()
    return fig


def anim_evolution(ck, out_path, config=None):
    config = config or TRIAL_CONFIGS[0]
    gens, genomes = ck["gens"], ck["genomes"]
    trajs = [_trajectory(g, config) for g in genomes]
    a = np.array(config["resource_a"], float); b = np.array(config["resource_b"], float)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.add_patch(plt.Circle(a, 7, fc=A_COLOR, ec="k", alpha=0.85))
    ax.add_patch(plt.Circle(b, 7, fc=B_COLOR, ec="k", alpha=0.85))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    line, = ax.plot([], [], color="0.3", lw=1.0)
    title = ax.set_title("")

    def update(i):
        tr = trajs[i]
        line.set_data(tr[:, 0], tr[:, 1])
        title.set_text(f"generation {int(gens[i])}  (survival {len(tr)} steps)")
        return [line, title]

    return save_gif(fig, update, len(gens), out_path, fps=1.4, dpi=80)


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
