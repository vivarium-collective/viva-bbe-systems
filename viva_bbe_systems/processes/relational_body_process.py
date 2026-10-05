"""RelationalBodyProcess: Williams, Beer & Gasser (2008) relational-categorization
body, with the two-object stream (obj1 -> ISI -> obj2) baked into its config.

Wraps :class:`CategoricalBody`, the evolved ``sensor_weights`` and a
:class:`TwoObjectStream` unchanged. The brain (``CTRNNProcess``) drives the
catcher through ``motor_output`` and reads ``sensor_weights @ shadow`` back
through ``sensory_input`` (length ``size``). ``phase`` is encoded obj1=0,
isi=1, obj2=2, done=3.

NOTE: the composite is a DEMONSTRATION vehicle — env->body->brain are wired
through stores (one-step delay) and the body acts before sensing, so its
trajectory differs slightly from ``RelationalAgent.run_trial``; it is not a
bit-faithful replay of the 0.688-accuracy evaluation.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process

from ..bodies.categorical_perception import CategoricalBody
from ..environments.object_stream import TwoObjectStream

PHASE_CODE = {"obj1": 0.0, "isi": 1.0, "obj2": 2.0, "done": 3.0}


class RelationalBodyProcess(Process):
    # sensor_weights is maybe[list[float]] with a null default (a list default
    # would be concatenation-doubled by the schema); resolved in __init__.
    config_schema = {
        "size": {"_type": "integer", "_default": 8},
        "n_sensors": {"_type": "integer", "_default": 7},
        "dt": {"_type": "float", "_default": 0.1},
        "s1": {"_type": "float", "_default": 3.0},
        "s2": {"_type": "float", "_default": 5.0},
        "offset1": {"_type": "float", "_default": 0.0},
        "offset2": {"_type": "float", "_default": 0.0},
        "motor_gain": {"_type": "float", "_default": 5.0},
        "sensor_weights": "maybe[list[float]]",
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        c = self.config
        self.size = int(c["size"])
        self.n_sensors = int(c["n_sensors"])
        sw = c.get("sensor_weights")
        if not sw:
            from ..bodies.relational_genome import rel_genome_length
            from ..genome import genome_length
            from ..tasks.evolve_relational import load_seed, relational_spec
            spec = relational_spec()
            sw = load_seed()[genome_length(spec.ctrnn_spec):]
        self.sensor_weights = np.asarray(sw, float).reshape(self.size, self.n_sensors)
        self.body = CategoricalBody(n_sensors=self.n_sensors)
        self.body.x = 0.0
        self.stream = TwoObjectStream(c["s1"], c["s2"], offset1=c["offset1"],
                                      offset2=c["offset2"])
        self._t = 0

    def inputs(self):
        return {"motor_output": "overwrite[array[float]]"}

    def outputs(self):
        return {"sensory_input": "overwrite[array[float]]",
                "agent_x": "overwrite[float]",
                "phase": "overwrite[float]"}

    def update(self, state, interval):
        m = np.asarray(state.get("motor_output", [0.0, 0.0]), dtype=float)
        self.body.act(m[:2], self.config["dt"], self.config["motor_gain"])
        t = min(self._t, self.stream.total_steps - 1)
        obj = self.stream.visible(t)
        shadow = (self.body.sense(obj) if obj is not None
                  else np.zeros(self.n_sensors))
        phase = PHASE_CODE[self.stream.phase(self._t)]
        self._t += 1
        return {"sensory_input": self.sensor_weights @ shadow,
                "agent_x": float(self.body.x),
                "phase": phase}


def build_relational_composite(s1=3.0, s2=5.0, seed_path=None, dt=0.1) -> dict:
    """Composite spec dict: CTRNNProcess brain <-> RelationalBodyProcess body."""
    from ..genome import genome_length
    from ..tasks.evolve_relational import DEFAULT_PATH, load_seed, relational_spec
    spec = relational_spec()
    g = load_seed(seed_path or DEFAULT_PATH)
    nctr = genome_length(spec.ctrnn_spec)
    size = spec.n_neurons
    return {
        "brain": {
            "_type": "process", "address": "local:CTRNNProcess",
            "config": {"size": size, "dt": dt, "genome": g[:nctr].tolist(),
                       "motor_indices": [size - 2, size - 1]},
            "interval": dt,
            "inputs": {"sensory_input": ["sensory_input"]},
            "outputs": {"motor_output": ["motor_output"],
                        "neuron_outputs": ["neuron_outputs"],
                        "neuron_states": ["neuron_states"]},
        },
        "body": {
            "_type": "process", "address": "local:RelationalBodyProcess",
            "config": {"size": size, "n_sensors": spec.n_sensors, "dt": dt,
                       "s1": s1, "s2": s2, "motor_gain": spec.motor_gain,
                       "sensor_weights": g[nctr:].tolist()},
            "interval": dt,
            "inputs": {"motor_output": ["motor_output"]},
            "outputs": {"sensory_input": ["sensory_input"],
                        "agent_x": ["agent_x"],
                        "phase": ["phase"]},
        },
        "sensory_input": [0.0] * size,
        "motor_output": [0.0, 0.0],
        "agent_x": 0.0,
        "phase": 0.0,
        "neuron_outputs": [0.0] * size,
        "neuron_states": [0.0] * size,
    }
