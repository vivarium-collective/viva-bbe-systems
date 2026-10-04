"""Smoke tests: each walker visualization renders a non-trivial file."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from viva_bbe_systems.agents.walker_agent import make_walker
from viva_bbe_systems.genome import decode
from viva_bbe_systems.tasks.evolve_walker import (
    load_seed, walker_spec, load_checkpoints, CHECKPOINT_PATH)


@pytest.fixture(scope="module")
def agent():
    return make_walker(decode(load_seed(), walker_spec()))


@pytest.fixture(scope="module")
def ck():
    c = load_checkpoints(CHECKPOINT_PATH)
    idx = [0, len(c["gens"]) // 2, len(c["gens"]) - 1]
    return {"gens": c["gens"][idx], "genomes": c["genomes"][idx],
            "fitness": c["fitness"][idx], "history": c["history"]}


def _ok(p):
    assert p.exists() and p.stat().st_size > 1000


def test_walk_gif(agent, tmp_path):
    from viva_bbe_systems.walker_anim import animate_walk
    _ok(animate_walk(agent, tmp_path / "walk.gif", steps=120, n_frames=6))


def test_neural_gif(agent, tmp_path):
    from viva_bbe_systems.walker_anim import animate_neural
    _ok(animate_neural(agent, tmp_path / "n.gif", steps=120, n_frames=6))


def test_limit_cycle(agent, tmp_path):
    from viva_bbe_systems.walker_gallery import fig_cpg_limit_cycle
    fig = fig_cpg_limit_cycle(agent)
    p = tmp_path / "a.png"; fig.savefig(p, dpi=60); plt.close(fig); _ok(p)


def test_gait_diagram(agent, tmp_path):
    from viva_bbe_systems.walker_gallery import fig_gait_diagram
    fig = fig_gait_diagram(agent)
    p = tmp_path / "b.png"; fig.savefig(p, dpi=60); plt.close(fig); _ok(p)


def test_evolution_curves(ck, tmp_path):
    from viva_bbe_systems.walker_evo_viz import fig_evolution_curves
    fig = fig_evolution_curves(ck)
    p = tmp_path / "c.png"; fig.savefig(p, dpi=60); plt.close(fig); _ok(p)


def test_evolution_gif(ck, tmp_path):
    from viva_bbe_systems.walker_evo_viz import anim_evolution
    _ok(anim_evolution(ck, tmp_path / "e.gif"))
