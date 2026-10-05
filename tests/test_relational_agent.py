import numpy as np
from viva_bbe_systems.agents.relational_agent import RelationalAgent, make_relational, RELATIONAL_SIZE


def _static_agent(**kw):
    agent = make_relational()
    agent.ctrnn.weights[:] = 0.0
    agent.sensor_weights[:] = 0.0
    for k, v in kw.items():
        setattr(agent, k, v)
    return agent


def test_trial_dict_and_lengths():
    a = make_relational()
    out = a.run_trial(3.0, 5.0)
    assert np.isfinite(out["final_distance"])
    assert out["x_hist"].shape == (360,)
    assert set(["caught", "should_catch", "correct"]) <= set(out)


def test_should_catch_and_correct():
    a = _static_agent()
    assert a.run_trial(3.0, 5.0)["should_catch"] is True
    assert a.run_trial(5.0, 3.0)["should_catch"] is False
    out = a.run_trial(5.0, 3.0)  # parked at 0 == offset2 -> caught, should avoid
    assert out["caught"] is True and out["correct"] is False


def test_determinism():
    a = make_relational()
    a.sensor_weights[:] = 0.3
    r1, r2 = a.run_trial(3.0, 5.0, offset2=2.0), a.run_trial(3.0, 5.0, offset2=2.0)
    assert np.array_equal(r1["x_hist"], r2["x_hist"]) and r1["caught"] == r2["caught"]


def test_caught_reflects_distance():
    a = _static_agent()
    assert a.run_trial(3.0, 5.0, offset2=0.0)["caught"] is True
    far = a.run_trial(3.0, 5.0, offset2=15.0)
    assert far["caught"] is False and far["final_distance"] > 10


def test_record():
    a = make_relational()
    out = a.run_trial(3.0, 5.0, record=True)
    assert out["outputs"].shape == (360, RELATIONAL_SIZE)
    assert len(out["obj_x_hist"]) == 360 and len(out["phase_hist"]) == 360
    assert "outputs" not in a.run_trial(3.0, 5.0)
