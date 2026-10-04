"""ForagerBodyProcess: the Agmon & Beer (2014) action-switching forager body,
with its chemotaxis environment and two-nutrient metabolism baked in.

Wraps :class:`viva_bbe_systems.bodies.chemotactic_forager.ChemotacticForager`,
:class:`viva_bbe_systems.environments.chemotaxis_resources.ChemotaxisEnv` and
:class:`viva_bbe_systems.metabolism.Metabolism` unchanged -- the same way the
categorical composite bakes the falling object into its body process. The brain
(``CTRNNProcess``) drives the effectors through ``motor_output`` ([right, left])
and reads the chemo/nutrient sensors back through ``sensory_input`` (one-step
delay). The per-step order mirrors ``ForagerAgent.run_trial``:
act -> inside -> metabolism.step -> sense-for-next.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process

from ..bodies.chemotactic_forager import ChemotacticForager
from ..environments.chemotaxis_resources import ChemotaxisEnv, Resource
from ..metabolism import Metabolism


class ForagerBodyProcess(Process):
    # NOTE: the list-valued params are ``maybe[list[float]]`` with a null
    # (scalar) default, not ``list[float]`` with a list default. A list default
    # in a process config_schema is applied by concatenation during schema
    # realization, which silently DOUBLES the default (``[50, 50]`` ->
    # ``[50, 50, 50, 50]``). Keeping the schema default scalar (null) and
    # resolving the real default here in ``__init__`` sidesteps that quirk.
    config_schema = {
        "morphology": {"_type": "string", "_default": "M2"},
        "size": {"_type": "integer", "_default": 9},
        "n_sensors": {"_type": "integer", "_default": 4},
        "dt": {"_type": "float", "_default": 0.1},
        "resource_a": "maybe[list[float]]",
        "resource_b": "maybe[list[float]]",
        "radius_a": {"_type": "float", "_default": 7.0},
        "radius_b": {"_type": "float", "_default": 7.0},
        "init_levels": "maybe[list[float]]",
        "start_pos": "maybe[list[float]]",
        "start_angle": {"_type": "float", "_default": 0.0},
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        c = self.config
        resource_a = c.get("resource_a") or [30.0, 70.0]
        resource_b = c.get("resource_b") or [70.0, 30.0]
        init_levels = c.get("init_levels") or [5.0, 5.0]
        start_pos = c.get("start_pos") or [50.0, 50.0]
        self.body = ChemotacticForager(c["morphology"])
        self.env = ChemotaxisEnv(
            Resource(center=np.asarray(resource_a, float),
                     radius=c["radius_a"], signal="A"),
            Resource(center=np.asarray(resource_b, float),
                     radius=c["radius_b"], signal="B"),
        )
        self.metab = Metabolism(np.asarray(init_levels, float))
        self.body.pos = np.asarray(start_pos, float)
        self.body.angle = float(c["start_angle"])
        self.body.velocity = 0.0

    def inputs(self):
        return {"motor_output": "overwrite[array[float]]"}

    def outputs(self):
        return {"sensory_input": "overwrite[array[float]]",
                "agent_pos": "overwrite[array[float]]",
                "nutrient_levels": "overwrite[array[float]]",
                "alive": "overwrite[float]"}

    def update(self, state, interval):
        m = np.asarray(state.get("motor_output", [0.0, 0.0]), dtype=float)
        self.body.act(m[0], m[1], self.env)
        inside_a, inside_b = self.body.inside(self.env)
        self.metab.step(inside_a, inside_b)
        cur = np.zeros(self.config["size"])
        cur[:self.config["n_sensors"]] = self.body.sense(self.env, self.metab)
        return {
            "sensory_input": cur,
            "agent_pos": self.body.pos.copy(),
            "nutrient_levels": self.metab.levels.copy(),
            "alive": float(self.metab.alive),
        }


def build_forager_composite(morphology="M2", seed_path=None, dt=0.1,
                            **resource_kw) -> dict:
    """Composite spec dict: CTRNNProcess brain <-> ForagerBodyProcess body.

    The body bakes the chemotaxis environment + metabolism in. Brain->body
    (``motor_output`` = [right, left]) and body->brain (``sensory_input``) links
    go through stores for the one-step delay, mirroring the categorical builder.
    """
    from ..agents.forager_agent import MORPHOLOGY_SIZE, _N_SENSORS
    from ..tasks.evolve_forager import DEFAULT_PATH, load_seed
    g = load_seed(seed_path or DEFAULT_PATH)
    size = MORPHOLOGY_SIZE[morphology]
    n_sensors = _N_SENSORS[morphology]
    body_config = {"morphology": morphology, "size": size,
                   "n_sensors": n_sensors, "dt": dt}
    body_config.update(resource_kw)
    return {
        "brain": {
            "_type": "process", "address": "local:CTRNNProcess",
            "config": {"size": size, "dt": dt, "genome": g.tolist(),
                       "motor_indices": [8, 7]},
            "interval": dt,
            "inputs": {"sensory_input": ["sensory_input"]},
            "outputs": {"motor_output": ["motor_output"],
                        "neuron_outputs": ["neuron_outputs"],
                        "neuron_states": ["neuron_states"]},
        },
        "body": {
            "_type": "process", "address": "local:ForagerBodyProcess",
            "config": body_config,
            "interval": dt,
            "inputs": {"motor_output": ["motor_output"]},
            "outputs": {"sensory_input": ["sensory_input"],
                        "agent_pos": ["agent_pos"],
                        "nutrient_levels": ["nutrient_levels"],
                        "alive": ["alive"]},
        },
        "sensory_input": [0.0] * size,
        "motor_output": [0.0, 0.0],
        "agent_pos": [50.0, 50.0],
        "nutrient_levels": [5.0, 5.0],
        "alive": 1.0,
        "neuron_outputs": [0.0] * size,
        "neuron_states": [0.0] * size,
    }
