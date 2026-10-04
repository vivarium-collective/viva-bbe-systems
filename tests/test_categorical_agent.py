import numpy as np
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.bodies.categorical_perception import CategoricalBody
from viva_bbe_systems.agents.categorical_agent import CategoricalAgent


def _agent(n=5, n_sensors=7):
    net = CTRNN(n, dt=0.1)
    net.weights[:] = 0.0
    body = CategoricalBody(n_sensors=n_sensors)
    sw = np.zeros((n, n_sensors))
    return CategoricalAgent(net, body, sw, motor_indices=(n - 2, n - 1), motor_gain=5.0)


def test_run_trial_shape_and_finiteness():
    agent = _agent()
    out = agent.run_trial(obj_offset=2.0, shape="circle", steps=50)
    assert out["trajectory"].shape == (50, 2)
    assert np.isfinite(out["final_distance"])


def test_zero_weights_agent_does_not_move():
    # no CTRNN weights, no sensor weights -> motor outputs constant -> but equal -> no net motion
    agent = _agent()
    out = agent.run_trial(obj_offset=3.0, shape="circle", steps=50)
    assert out["trajectory"][0, 0] == 0.0
    assert abs(out["trajectory"][-1, 0]) < 1e-6   # symmetric motors => stays put


def test_agent_passes_object_stays_finite():
    agent = _agent()
    out = agent.run_trial(obj_offset=0.0, shape="circle", steps=200)
    assert np.all(np.isfinite(out["trajectory"]))
