"""Chemotactic forager body: sensors, effectors, motion (Agmon & Beer 2014)."""

from math import pi

import numpy as np

_SIDE = [(pi / 2, "A"), (pi / 2, "B"), (-pi / 2, "A"), (-pi / 2, "B")]
_FRONT = [(0.0, "A"), (0.0, "B")]

# morphology -> (chemosensor layout, has nutrient sensors)
MORPHOLOGY_LAYOUTS = {
    "M1": (_SIDE, True),
    "M2": (_FRONT, True),
    "M3": (_SIDE, False),
}
MORPHOLOGIES = frozenset(MORPHOLOGY_LAYOUTS)


class ChemotacticForager:
    def __init__(self, morphology="M2", sensor_dist=6.0, max_angle=pi / 12,
                 max_thrust=0.025, friction=0.9):
        # max_thrust 0.025 (was 0.008): ~3x top speed so the agent can traverse
        # SPREAD resource layouts (sep up to ~70) within a nutrient lifetime and
        # genuinely search for the far resource, rather than only shuttling
        # between near ones. Paired with a higher eat_rate + longer-range
        # gradient + lower drain (see Metabolism / concentration) so a faster
        # pass still replenishes and distant resources stay sensable.
        if morphology not in MORPHOLOGY_LAYOUTS:
            raise ValueError(f"unknown morphology {morphology!r}")
        self.morphology = morphology
        self.sensor_dist = sensor_dist
        self.max_angle = max_angle
        self.max_thrust = max_thrust
        self.friction = friction
        self.pos = np.zeros(2)
        self.angle = 0.0
        self.velocity = 0.0

    def sensor_layout(self):
        return list(MORPHOLOGY_LAYOUTS[self.morphology][0])

    @property
    def n_chemo(self):
        return len(MORPHOLOGY_LAYOUTS[self.morphology][0])

    @property
    def n_nutrient(self):
        return 2 if MORPHOLOGY_LAYOUTS[self.morphology][1] else 0

    @property
    def n_sensors(self):
        return self.n_chemo + self.n_nutrient

    def sense(self, env, metabolism):
        vals = []
        for offset, signal in self.sensor_layout():
            a = self.angle + offset
            p = self.pos + self.sensor_dist * np.array([np.cos(a), np.sin(a)])
            vals.append(env.conc_at(p, signal))
        if self.n_nutrient:
            vals.extend(float(x) for x in metabolism.levels)
        return np.array(vals, float)

    def act(self, o_right, o_left, env):
        torque = (o_right - o_left) * self.max_angle
        thrust = (o_right + o_left) * self.max_thrust
        self.velocity = self.velocity * self.friction + thrust
        self.angle = self.angle + torque
        heading = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.pos = env.clamp(self.pos + self.velocity * heading)

    def inside(self, env):
        return (bool(env.inside(self.pos, "A")), bool(env.inside(self.pos, "B")))
