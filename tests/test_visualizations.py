"""Smoke tests for the animation + evolution-viz modules."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viva_bbe_systems.anim import save_gif
from viva_bbe_systems.bodies.categorical_genome import CatGenomeSpec, decode_agent
from viva_bbe_systems.tasks.evolve_categorical import load_seed, load_checkpoints, DEFAULT_PATH


def test_save_gif_writes_file(tmp_path):
    fig, ax = plt.subplots()
    ln, = ax.plot([], [])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def update(i):
        ln.set_data([0, i / 5], [0, i / 5])
        return [ln]

    p = save_gif(fig, update, 5, tmp_path / "x.gif", fps=5)
    assert p.exists() and p.stat().st_size > 200


def test_animate_trial_renders_gif(tmp_path):
    from viva_bbe_systems.categorical_anim import animate_trial
    agent = decode_agent(load_seed(DEFAULT_PATH), CatGenomeSpec(), dt=0.1)
    p = animate_trial(agent, 3.0, "circle", tmp_path / "solve.gif", steps=20, stride=2)
    assert p.exists() and p.stat().st_size > 1000


def test_committed_evolution_checkpoints_reproduce_seed():
    """The recorded evolution's final checkpoint is the committed seed (seed=0)."""
    ck = load_checkpoints()
    assert ck["genomes"].shape[0] == ck["gens"].shape[0] >= 2
    assert ck["gens"][0] == 0
    # final checkpoint genome == the committed best genome (deterministic path)
    seed = load_seed(DEFAULT_PATH)
    np.testing.assert_allclose(ck["genomes"][-1], seed)
    # fitness rises over evolution
    assert ck["fitness"][-1] > ck["fitness"][0]


def test_evolution_curves_figure():
    from viva_bbe_systems.categorical_evo_viz import fig_evolution_curves
    # tiny synthetic checkpoints to keep the unit test fast
    spec = CatGenomeSpec()
    g = load_seed(DEFAULT_PATH)
    ck = {"gens": np.array([0, 1]), "genomes": np.stack([g, g]),
          "fitness": np.array([0.5, 0.93]), "history": np.array([0.5, 0.93])}
    fig = fig_evolution_curves(ck, spec)
    assert fig.axes and fig.axes[0].lines
    plt.close(fig)
