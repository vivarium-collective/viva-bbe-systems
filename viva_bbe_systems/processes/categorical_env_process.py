"""FallingObjectEnvironment: Beer-2003 falling circle/diamond as a Process, plus a
builder that wires env -> body -> CTRNN brain from the committed seed genome."""
from __future__ import annotations
import numpy as np
from process_bigraph import Process

SHAPE_CODES = {"circle": 0.0, "diamond": 1.0}


class FallingObjectEnvironment(Process):
    config_schema = {
        "offset": {"_type": "float", "_default": 0.0},
        "shape": {"_type": "string", "_default": "circle"},
        "vy": {"_type": "float", "_default": 1.0},
        "size": {"_type": "float", "_default": 3.0},
        "H": {"_type": "float", "_default": 20.0},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        if self.config["shape"] not in SHAPE_CODES:
            raise ValueError(f"unknown shape {self.config['shape']!r}")
        self.center = np.array([self.config["offset"], self.config["H"]], dtype=float)

    def inputs(self):
        return {}

    def outputs(self):
        return {"obj_center": "overwrite[array[float]]",
                "obj_size": "overwrite[float]", "obj_shape": "overwrite[float]"}

    def update(self, state, interval):
        self.center[1] -= self.config["vy"] * interval
        return {"obj_center": self.center.copy(),
                "obj_size": float(self.config["size"]),
                "obj_shape": SHAPE_CODES[self.config["shape"]]}


def build_categorical_composite(seed_path=None, offset=3.0, shape="circle",
                                interval=0.1) -> dict:
    """Composite spec dict (env -> body(sense) -> CTRNN brain -> body(act)).

    The brain->body and body->brain links go through stores, so each reads the
    other's previous-step value (one-step delay).
    """
    from ..genome import GenomeSpec, genome_length
    from ..tasks.evolve_categorical import DEFAULT_PATH, load_seed
    g = load_seed(seed_path or DEFAULT_PATH)
    spec = GenomeSpec(5)
    nctr = genome_length(spec)
    sw = g[nctr:].reshape(5, 7)
    return {
        "env": {
            "_type": "process", "address": "local:FallingObjectEnvironment",
            "config": {"offset": float(offset), "shape": shape},
            "interval": interval, "inputs": {},
            "outputs": {"obj_center": ["obj_center"], "obj_size": ["obj_size"],
                        "obj_shape": ["obj_shape"]},
        },
        "body": {
            "_type": "process", "address": "local:CategoricalBodyProcess",
            "config": {"sensor_weights": sw.tolist(), "dt": interval},
            "interval": interval,
            "inputs": {"obj_center": ["obj_center"], "obj_shape": ["obj_shape"],
                       "motor_output": ["motor_output"]},
            "outputs": {"sensory_input": ["sensory_input"], "agent_x": ["agent_x"]},
        },
        "brain": {
            "_type": "process", "address": "local:CTRNNProcess",
            "config": {"size": 5, "dt": interval, "genome": g[:nctr].tolist(),
                       "motor_indices": [3, 4]},
            "interval": interval,
            "inputs": {"sensory_input": ["sensory_input"]},
            "outputs": {"motor_output": ["motor_output"],
                        "neuron_outputs": ["neuron_outputs"],
                        "neuron_states": ["neuron_states"]},
        },
        "obj_center": [float(offset), 20.0],
        "obj_size": 3.0,
        "obj_shape": SHAPE_CODES[shape],
        "sensory_input": [0.0] * 5,
        "motor_output": [0.0, 0.0],
        "neuron_outputs": [0.0] * 5,
        "neuron_states": [0.0] * 5,
        "agent_x": 0.0,
    }
