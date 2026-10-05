"""The committed relational seed reproduces relational categorization WITH MEMORY
(Williams, Beer & Gasser 2008).

The agent sees two objects in sequence and catches the second iff it is LARGER
than the first — which is impossible without holding the first object's size in
neural state across the inter-stimulus interval (object 1 is gone when object 2
appears). The battery is built so no single-object (memoryless) policy exceeds
~0.625 accuracy; the agent beats that AND, directly, its ISI neural state
encodes object 1's size (corr ~0.95). An honest partial reproduction: the memory
is clean, the decision readout imperfect (accuracy ~0.69).
"""
import numpy as np

from viva_bbe_systems.tasks.evolve_relational import (
    load_seed, relational_spec, accuracy_report,
    always_catch_accuracy, always_avoid_accuracy)
from viva_bbe_systems.bodies.relational_genome import decode_agent
from viva_bbe_systems.environments.object_stream import TwoObjectStream

SINGLE_OBJECT_CEILING = 0.625  # best accuracy any memoryless (absolute-size) policy reaches


def _isi_memory_corr(agent):
    """Best |correlation| between a neuron's mid-ISI state and s1, over a size
    sweep, measured during the REAL trial (the body moves) — object 1 is gone
    during the ISI, so any correlation IS a held memory."""
    sizes = np.linspace(2.0, 6.0, 9)
    states = []
    for s1 in sizes:
        agent.ctrnn.reset(np.zeros(agent.ctrnn.size))
        agent.body.x = 0.0
        stream = TwoObjectStream(s1, 4.0)
        mid_isi = stream.phase1_steps + stream.isi_steps // 2
        for t in range(mid_isi + 1):
            obj = stream.visible(t)
            shadow = agent.body.sense(obj) if obj is not None else np.zeros(agent.body.n_sensors)
            o = agent.ctrnn.step(external_input=agent.sensor_weights @ shadow)
            motor = np.array([o[agent.motor_indices[0]], o[agent.motor_indices[1]]])
            agent.body.act(motor, 0.1, agent.motor_gain)   # the agent behaves normally
        states.append(agent.ctrnn.y.copy())
    states = np.array(states)
    return max(abs(np.corrcoef(sizes, states[:, n])[0, 1]) for n in range(states.shape[1]))


def test_seed_categorizes_above_memoryless_ceiling():
    g = load_seed()
    rep = accuracy_report(g)
    # beats BOTH fixed-policy baselines and the memoryless single-object ceiling
    assert rep["accuracy"] > always_catch_accuracy()
    assert rep["accuracy"] > always_avoid_accuracy()
    assert rep["accuracy"] > SINGLE_OBJECT_CEILING, rep  # => must use s1 relative to s2
    # and does so on BOTH relations, not by always-catching or always-avoiding
    assert rep["catch_accuracy"] > 0.5 and rep["avoid_accuracy"] > 0.5, rep


def test_seed_holds_obj1_in_memory():
    agent = decode_agent(load_seed(), relational_spec())
    corr = _isi_memory_corr(agent)
    # object 1's size is clearly encoded in persistent neural state over the ISI
    assert corr > 0.80, f"ISI memory corr {corr:.2f} too weak to call it memory"


def test_seed_is_deterministic():
    a = accuracy_report(load_seed())["accuracy"]
    b = accuracy_report(load_seed())["accuracy"]
    assert a == b
