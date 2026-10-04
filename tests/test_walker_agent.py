import numpy as np

from viva_bbe_systems.agents.walker_agent import WALKER_SIZE, WalkerAgent, make_walker
from viva_bbe_systems.bodies.legged import LeggedBody
from viva_bbe_systems.ctrnn import CTRNN


def _cpg():
    """Hand-built CPG: inter 0/1 form an oscillator (excitatory self, cross
    excitation/inhibition); foot follows neuron 0 (stance), BS follows 0,
    FS follows 1, so the planted leg swings back then lifts and swings forward."""
    c = CTRNN(WALKER_SIZE, dt=0.1)
    w = np.zeros((WALKER_SIZE, WALKER_SIZE))
    w[0, 0] = w[1, 1] = 4.5
    w[0, 1] = -8.0
    w[1, 0] = 8.0
    w[2, 0] = 10.0   # foot down while neuron 0 is high
    w[3, 0] = 10.0   # backward swing (stance)
    w[4, 1] = 10.0   # forward swing
    c.weights = w
    c.theta = -0.5 * w.sum(axis=1)  # center-crossing
    c.tau = np.array([1.0, 1.0, 0.2, 0.2, 0.2])
    return c


def test_trial_shapes_and_finite():
    a = make_walker()
    r = a.run_trial(steps=50)
    assert np.isfinite(r["distance"])
    for k in ("x_hist", "angle_hist", "foot_hist", "vx_hist"):
        assert r[k].shape == (50,)
    assert r["foot_hist"].dtype == bool
    assert "outputs" not in r


def test_record_outputs():
    r = make_walker().run_trial(steps=30, record=True)
    assert r["outputs"].shape == (30, WALKER_SIZE)


def test_cpg_walks_forward():
    r = WalkerAgent(_cpg(), LeggedBody()).run_trial(steps=500, record=True)
    assert r["distance"] > 0
    assert r["foot_hist"].any() and not r["foot_hist"].all()


def test_deterministic():
    a = WalkerAgent(_cpg(), LeggedBody())
    r1, r2 = a.run_trial(steps=200), a.run_trial(steps=200)
    assert r1["distance"] == r2["distance"]
    assert np.array_equal(r1["x_hist"], r2["x_hist"])
