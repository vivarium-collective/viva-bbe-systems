"""Smoke tests for the relational-categorization viz modules (committed seed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viva_bbe_systems.tasks.evolve_relational import (
    load_seed, load_checkpoints, relational_spec)
from viva_bbe_systems.bodies.relational_genome import decode_agent

AGENT = decode_agent(load_seed(), relational_spec())


def _png(fig, path):
    fig.savefig(path, dpi=60)
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 1000


def test_relational_gif(tmp_path):
    from viva_bbe_systems.relational_anim import animate_relational
    p = animate_relational(AGENT, tmp_path / "relational.gif", n_frames=8)
    assert p.exists() and p.stat().st_size > 1000


def test_neural_gif(tmp_path):
    from viva_bbe_systems.relational_anim import animate_neural
    p = animate_neural(AGENT, tmp_path / "neural.gif", n_frames=8)
    assert p.exists() and p.stat().st_size > 1000


def test_memory_dynamics(tmp_path):
    from viva_bbe_systems.relational_gallery import fig_memory_dynamics
    _png(fig_memory_dynamics(AGENT, n_s1=4), tmp_path / "m.png")


def test_decision_map(tmp_path):
    from viva_bbe_systems.relational_gallery import fig_decision_map
    _png(fig_decision_map(AGENT, n=3, offsets=(0.0,)), tmp_path / "d.png")


def test_evolution_curves(tmp_path):
    from viva_bbe_systems.relational_evo_viz import fig_evolution_curves
    ck = load_checkpoints()
    _png(fig_evolution_curves(ck), tmp_path / "e.png")


def test_evolution_gif(tmp_path):
    from viva_bbe_systems.relational_evo_viz import anim_evolution
    ck = load_checkpoints()
    small = {k: v[:2] if k in ("gens", "genomes", "fitness") else v for k, v in ck.items()}
    p = anim_evolution(small, tmp_path / "e.gif", n=3)
    assert p.exists() and p.stat().st_size > 1000
