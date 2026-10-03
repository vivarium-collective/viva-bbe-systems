import numpy as np
from process_bigraph import Composite
from viva_bbe_systems.core import build_core


def test_ctrnn_process_runs_in_composite():
    core = build_core()
    spec = {
        "brain": {
            "_type": "process",
            "address": "local:CTRNNProcess",
            "config": {"size": 2, "dt": 0.01,
                       "weights": [[4.5, 1.0], [-1.0, 4.5]],
                       "motor_indices": [1]},
            "interval": 0.1,
            "inputs": {"sensory_input": ["sensory"]},
            "outputs": {"motor_output": ["motor"],
                        "neuron_states": ["states"]},
        },
        "sensory": [0.5, 0.0],
        "motor": [0.0],
        "states": [0.0, 0.0],
    }
    sim = Composite({"state": spec}, core=core)
    sim.run(1.0)
    states = np.asarray(sim.state["states"], dtype=float)
    assert states.shape == (2,)
    assert np.any(states != 0.0)
    assert sim.state["motor"] is not None
