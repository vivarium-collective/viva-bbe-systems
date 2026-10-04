import numpy as np
from process_bigraph import Composite
from viva_bbe_systems.core import build_core
from viva_bbe_systems.processes.categorical_env_process import build_categorical_composite


def _traj(offset, shape="circle", steps=100, spec_edit=None):
    spec = build_categorical_composite(offset=offset, shape=shape)
    if spec_edit:
        spec_edit(spec)
    sim = Composite({"state": spec}, core=build_core())
    xs = []
    for _ in range(steps):
        sim.run(0.1)
        xs.append(float(sim.state["agent_x"]))
    return sim, np.array(xs)


def test_processes_discovered():
    core = build_core()
    assert "FallingObjectEnvironment" in core.link_registry
    assert "CategoricalBodyProcess" in core.link_registry


def test_agent_moves_toward_circle_through_engine():
    sim, xs = _traj(3.0, steps=60)               # 6 s of the 20 s fall
    assert abs(xs[-1] - 3.0) < 1.0, xs[-1]       # started 3.0 away at x=0
    assert abs(xs[-1] - 3.0) < abs(0.0 - 3.0)
    assert float(sim.state["obj_center"][1]) < 20.0   # object descended
    assert float(sim.state["obj_center"][0]) == 3.0   # x stays at offset


def test_agent_reaches_circle_on_either_side():
    for off in (3.0, -3.0):
        _, xs = _traj(off)
        assert np.min(np.abs(xs - off)) < 1.5, (off, xs.min(), xs.max())


def test_motor_feedback_matters():
    # body with motor disconnected must not move: guards against dead feedback
    def cut(spec):
        spec["body"]["inputs"]["motor_output"] = ["no_motor"]
        spec["no_motor"] = [0.0, 0.0]
    _, xs = _traj(3.0, steps=60, spec_edit=cut)
    assert np.all(xs == 0.0)
