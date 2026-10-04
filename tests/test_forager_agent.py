import numpy as np
import pytest

from viva_bbe_systems.agents.forager_agent import ForagerAgent, MORPHOLOGY_SIZE, make_forager
from viva_bbe_systems.environments.chemotaxis_resources import ChemotaxisEnv, Resource


def _env():
    return ChemotaxisEnv(
        Resource(center=np.array([20.0, 20.0]), signal="A"),
        Resource(center=np.array([80.0, 80.0]), signal="B"),
    )


def _agent(morph="M2", seed=0):
    agent = make_forager(morph)
    rng = np.random.default_rng(seed)
    n = agent.ctrnn.size
    agent.ctrnn.weights = rng.normal(0, 3.0, (n, n))
    agent.ctrnn.theta = -0.5 * agent.ctrnn.weights.sum(axis=1)
    return agent


def test_morphology_sizes():
    assert MORPHOLOGY_SIZE == {"M1": 11, "M2": 9, "M3": 9}
    for m, n in MORPHOLOGY_SIZE.items():
        assert make_forager(m).ctrnn.size == n


def test_trial_shapes_and_bounds():
    r = _agent().run_trial(_env(), (5.0, 5.0), start_pos=(50, 50), max_steps=300)
    T = r["survival"]
    assert 0 < T <= 300
    assert r["path"].shape == (T, 2) and r["levels_hist"].shape == (T, 2)
    assert np.all(np.isfinite(r["path"]))


def test_dies_far_from_resources_with_low_nutrients():
    env = ChemotaxisEnv(
        Resource(center=np.array([5.0, 5.0]), signal="A"),
        Resource(center=np.array([95.0, 5.0]), signal="B"),
    )
    r = _agent().run_trial(env, (0.5, 0.5), start_pos=(50, 95), max_steps=5000)
    assert r["survival"] < 5000 and not r["alive_at_end"]


def test_levels_hist_rows_distinct():
    r = _agent().run_trial(_env(), (5.0, 5.0), start_pos=(50, 50), max_steps=50)
    lh = r["levels_hist"]
    assert not np.allclose(lh[0], lh[-1])
    assert len({tuple(row) for row in lh}) > 1


def test_record_outputs():
    agent = _agent()
    r = agent.run_trial(_env(), (5.0, 5.0), start_pos=(50, 50), max_steps=40, record=True)
    assert r["outputs"].shape == (r["survival"], agent.ctrnn.size)
    assert "outputs" not in agent.run_trial(_env(), (5.0, 5.0), start_pos=(50, 50), max_steps=5)


def test_determinism():
    agent = _agent()
    a = agent.run_trial(_env(), (5.0, 5.0), start_pos=(50, 50), max_steps=200)
    b = agent.run_trial(_env(), (5.0, 5.0), start_pos=(50, 50), max_steps=200)
    assert a["survival"] == b["survival"]
    assert np.array_equal(a["path"], b["path"])


def test_undersized_ctrnn_rejected():
    from viva_bbe_systems.ctrnn import CTRNN
    from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
    with pytest.raises(ValueError):
        ForagerAgent(CTRNN(7), ChemotacticForager("M1"))
