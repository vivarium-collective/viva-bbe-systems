"""Forager agent runner: CTRNN brain + chemotactic body + environment + metabolism."""
from __future__ import annotations

import numpy as np

from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.metabolism import Metabolism

N_INTER = 3
_N_SENSORS = {"M1": 6, "M2": 4, "M3": 4}
# morphology -> required ctrnn.size = n_sensors + n_inter + 2 (sensors first, motors last)
MORPHOLOGY_SIZE = {m: s + N_INTER + 2 for m, s in _N_SENSORS.items()}


def make_forager(morphology, ctrnn=None, **body_kwargs):
    """Build a ForagerAgent for a morphology (zero-weight CTRNN of the right size if none given)."""
    body = ChemotacticForager(morphology, **body_kwargs)
    return ForagerAgent(ctrnn if ctrnn is not None else CTRNN(MORPHOLOGY_SIZE[morphology]), body)


class ForagerAgent:
    def __init__(self, ctrnn, body, *, motor_right_index=-1, motor_left_index=-2):
        if ctrnn.size < body.n_sensors + 2:
            raise ValueError(
                f"ctrnn.size {ctrnn.size} < n_sensors+2 = {body.n_sensors + 2}")
        self.ctrnn = ctrnn
        self.body = body
        self.motor_right_index = motor_right_index
        self.motor_left_index = motor_left_index

    def run_trial(self, env, init_levels, *, start_pos, start_angle=0.0,
                  max_steps=5000, record=False):
        ctrnn, body = self.ctrnn, self.body
        ctrnn.reset(np.zeros(ctrnn.size))
        body.pos = np.array(start_pos, float)
        body.angle = float(start_angle)
        body.velocity = 0.0
        metab = Metabolism(init_levels)
        ns = body.n_sensors

        path, levels, outs = [], [], []
        steps = 0
        while steps < max_steps:
            full = np.zeros(ctrnn.size)
            full[:ns] = body.sense(env, metab)
            o = ctrnn.step(full)
            body.act(o[self.motor_right_index], o[self.motor_left_index], env)
            inside_a, inside_b = body.inside(env)
            metab.step(inside_a, inside_b)
            steps += 1
            path.append(body.pos.copy())
            levels.append(metab.levels.copy())  # levels is an alias of internal state
            if record:
                outs.append(np.array(o, float))
            if not metab.alive:
                break

        result = {
            "survival": steps,
            "alive_at_end": bool(metab.alive),
            "path": np.array(path).reshape(-1, 2),
            "levels_hist": np.array(levels).reshape(-1, 2),
        }
        if record:
            result["outputs"] = np.array(outs).reshape(-1, ctrnn.size)
        return result
