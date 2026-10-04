"""WalkerBodyProcess: the Beer & Gallagher (1992) one-leg body as a Process.

Wraps :class:`viva_bbe_systems.bodies.legged.LeggedBody` unchanged. The brain
(``CTRNNProcess``) drives it through the ``motor_output`` store ([foot, bs, fs])
and it feeds proprioception back through the ``sensory_input`` store, so the
brain reads the body's previous-step angle (one-step delay, exactly like the
categorical composite). Flat ground is baked into the body; there is no
environment process.
"""
from __future__ import annotations

from math import pi

import numpy as np
from process_bigraph import Process

from ..bodies.legged import LeggedBody


class WalkerBodyProcess(Process):
    config_schema = {
        "size": {"_type": "integer", "_default": 5},
        "angle_input_index": {"_type": "integer", "_default": 0},
        "dt": {"_type": "float", "_default": 0.1},
        "start_angle": {"_type": "float", "_default": 0.0},
        "leg_length": {"_type": "float", "_default": 15.0},
        "omega_gain": {"_type": "float", "_default": 1.0},
        "friction": {"_type": "float", "_default": 0.9},
        "foot_threshold": {"_type": "float", "_default": 0.5},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        c = self.config
        self.body = LeggedBody(
            leg_length=c["leg_length"], omega_gain=c["omega_gain"],
            friction=c["friction"], foot_threshold=c["foot_threshold"],
        )
        self.body.reset(angle=c["start_angle"])

    def inputs(self):
        return {"motor_output": "overwrite[array[float]]"}

    def outputs(self):
        return {"sensory_input": "overwrite[array[float]]",
                "body_x": "overwrite[float]",
                "foot_down": "overwrite[float]",
                "leg_angle": "overwrite[float]",
                "body_vx": "overwrite[float]"}

    def update(self, state, interval):
        m = np.asarray(state.get("motor_output", [0.0, 0.0, 0.0]), dtype=float)
        self.body.act(m[0], m[1], m[2], self.config["dt"])
        cur = np.zeros(self.config["size"])
        cur[self.config["angle_input_index"]] = self.body.sense()
        return {
            "sensory_input": cur,
            "body_x": float(self.body.x),
            "foot_down": float(self.body.foot_down),
            "leg_angle": float(self.body.angle),
            "body_vx": float(self.body.vx),
        }


def build_walker_composite(seed_path=None, dt=0.1) -> dict:
    """Composite spec dict: CTRNNProcess brain <-> WalkerBodyProcess body.

    The brain->body (``motor_output``) and body->brain (``sensory_input``) links
    go through stores, so each reads the other's previous-step value (one-step
    delay), mirroring the categorical composite.
    """
    from ..tasks.evolve_walker import DEFAULT_PATH, load_seed, walker_spec
    g = load_seed(seed_path or DEFAULT_PATH)
    size = walker_spec().size
    return {
        "brain": {
            "_type": "process", "address": "local:CTRNNProcess",
            "config": {"size": size, "dt": dt, "genome": g.tolist(),
                       "motor_indices": [2, 3, 4]},
            "interval": dt,
            "inputs": {"sensory_input": ["sensory_input"]},
            "outputs": {"motor_output": ["motor_output"],
                        "neuron_outputs": ["neuron_outputs"],
                        "neuron_states": ["neuron_states"]},
        },
        "body": {
            "_type": "process", "address": "local:WalkerBodyProcess",
            "config": {"size": size, "dt": dt},
            "interval": dt,
            "inputs": {"motor_output": ["motor_output"]},
            "outputs": {"sensory_input": ["sensory_input"],
                        "body_x": ["body_x"], "foot_down": ["foot_down"],
                        "leg_angle": ["leg_angle"], "body_vx": ["body_vx"]},
        },
        "sensory_input": [0.0] * size,
        "motor_output": [0.0, 0.0, 0.0],
        "body_x": 0.0,
        "foot_down": 0.0,
        "leg_angle": 0.0,
        "body_vx": 0.0,
        "neuron_outputs": [0.0] * size,
        "neuron_states": [0.0] * size,
    }
