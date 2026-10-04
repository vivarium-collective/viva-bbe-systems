"""Smoke tests for the action-switching viz modules (render from the committed seed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viva_bbe_systems.tasks.evolve_forager import load_seed, DEFAULT_PATH, forager_spec
from viva_bbe_systems.tasks.forager_fitness import TRIAL_CONFIGS
from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.agents.forager_agent import ForagerAgent
from viva_bbe_systems.genome import decode


def _agent():
    return ForagerAgent(decode(load_seed(DEFAULT_PATH), forager_spec("M2")),
                        ChemotacticForager("M2"))


def test_gallery_figures():
    from viva_bbe_systems import forager_gallery as fg
    ag = _agent()
    for fn in (fg.fig_trajectory_and_nutrients, fg.fig_nutrient_phase, fg.fig_action_switching):
        f = fn(ag, TRIAL_CONFIGS[0])
        assert f.axes
        plt.close(f)
    f = fg.fig_morphologies()
    assert len(f.axes) == 3
    plt.close(f)


def test_forage_animation_renders(tmp_path):
    from viva_bbe_systems.forager_anim import animate_forage
    p = animate_forage(_agent(), TRIAL_CONFIGS[0], tmp_path / "forage.gif", max_steps=200)
    assert p.exists() and p.stat().st_size > 1000


def test_multi_env_animation_renders(tmp_path):
    from viva_bbe_systems.forager_anim import animate_multi
    p = animate_multi(_agent(), TRIAL_CONFIGS[:6], tmp_path / "multi.gif",
                      max_steps=150, n_frames=10)
    assert p.exists() and p.stat().st_size > 1000


def test_neural_activity_animation_renders(tmp_path):
    from viva_bbe_systems.forager_anim import animate_neural
    p = animate_neural(_agent(), TRIAL_CONFIGS[0], tmp_path / "neural.gif",
                       max_steps=150, n_frames=10)
    assert p.exists() and p.stat().st_size > 1000


def test_evolution_viz_renders(tmp_path):
    from viva_bbe_systems.forager_evo_viz import anim_evolution
    from viva_bbe_systems.tasks.evolve_forager import load_checkpoints
    p = anim_evolution(load_checkpoints(), tmp_path / "evo.gif")
    assert p.exists() and p.stat().st_size > 1000
