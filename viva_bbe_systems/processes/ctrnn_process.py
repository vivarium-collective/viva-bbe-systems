"""CTRNNProcess: the shared Beer CTRNN brain as a process-bigraph Process."""
from __future__ import annotations
import numpy as np
from process_bigraph import Process
from ..ctrnn import CTRNN, center_crossing_biases
from ..genome import GenomeSpec, decode


class CTRNNProcess(Process):
    """CTRNN as a process-bigraph Process.

    `sensory_input` must be length `size` (it enters `-y + w@o + I` directly);
    BBE bodies zero-pad currents for non-sensor neurons.
    """

    config_schema = {
        "size": "integer",
        "dt": {"_type": "float", "_default": 0.01},
        "tau": "maybe[list[float]]",
        "theta": "maybe[list[float]]",
        "weights": "maybe[list[list[float]]]",
        "genome": "maybe[list[float]]",
        "center_crossing": {"_type": "boolean", "_default": False},
        "motor_indices": {"_type": "list[integer]", "_default": []},
        "initial_state": "maybe[list[float]]",
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        n = self.config["size"]
        if self.config.get("genome") is not None:
            self.net = decode(np.asarray(self.config["genome"]), GenomeSpec(size=n))
            self.net.dt = self.config["dt"]
        else:
            self.net = CTRNN(n, dt=self.config["dt"])
            if self.config.get("weights") is not None:
                self.net.weights[:] = np.asarray(self.config["weights"])
            if self.config.get("tau") is not None:
                self.net.tau[:] = np.asarray(self.config["tau"])
            if self.config.get("theta") is not None:
                self.net.theta[:] = np.asarray(self.config["theta"])
            elif self.config.get("center_crossing"):
                self.net.theta[:] = center_crossing_biases(self.net.weights)
        self.net.reset(self.config.get("initial_state"))
        self.motor_indices = list(self.config["motor_indices"])

    def inputs(self):
        return {"sensory_input": "array[float]"}

    def outputs(self):
        return {
            "motor_output": "array[float]",
            "neuron_outputs": "array[float]",
            "neuron_states": "array[float]",
        }

    def update(self, state, interval):
        I = np.asarray(state.get("sensory_input", np.zeros(self.config["size"])), dtype=float)
        steps = max(1, int(round(interval / self.net.dt)))
        o = self.net.outputs()
        for _ in range(steps):
            o = self.net.step(external_input=I)
        motor = o[self.motor_indices] if self.motor_indices else np.array([0.0])
        return {"motor_output": motor, "neuron_outputs": o, "neuron_states": self.net.y.copy()}
