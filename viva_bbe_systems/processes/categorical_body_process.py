"""CategoricalBodyProcess: ray-sensor fan + horizontal motor body as a Process.
Reuses CategoricalBody for sense/act (no duplicated geometry)."""
from __future__ import annotations
import numpy as np
from process_bigraph import Process
from ..bodies.categorical_perception import CategoricalBody
from ..environments.falling_objects import FallingObject


class CategoricalBodyProcess(Process):
    config_schema = {
        "sensor_weights": "list[list[float]]",
        "n_sensors": {"_type": "integer", "_default": 7},
        "half_angle": {"_type": "float", "_default": float(np.pi / 6)},
        "max_range": {"_type": "float", "_default": 20.0},
        "motor_gain": {"_type": "float", "_default": 5.0},
        "x0": {"_type": "float", "_default": 0.0},
        "dt": {"_type": "float", "_default": 0.1},
        "size": {"_type": "float", "_default": 3.0},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        c = self.config
        self.W = np.asarray(c["sensor_weights"], dtype=float)
        self.body = CategoricalBody(n_sensors=c["n_sensors"],
                                    half_angle=c["half_angle"],
                                    max_range=c["max_range"], x=c["x0"])

    def inputs(self):
        return {"obj_center": "overwrite[array[float]]",
                "obj_shape": "overwrite[float]",
                "motor_output": "overwrite[array[float]]"}

    def outputs(self):
        return {"sensory_input": "overwrite[array[float]]",
                "agent_x": "overwrite[float]"}

    def update(self, state, interval):
        shape = "diamond" if float(state["obj_shape"]) >= 0.5 else "circle"
        obj = FallingObject(center=np.asarray(state["obj_center"], dtype=float),
                            vy=0.0, size=self.config["size"], shape=shape)
        I = self.W @ self.body.sense(obj)
        motor = np.asarray(state.get("motor_output", [0.0, 0.0]), dtype=float)
        self.body.act(motor, self.config["dt"], self.config["motor_gain"])
        return {"sensory_input": I, "agent_x": float(self.body.x)}
