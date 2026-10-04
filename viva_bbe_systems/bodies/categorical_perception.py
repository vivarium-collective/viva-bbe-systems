"""Beer-2003 categorical-perception agent body: a horizontal mover with a fan of
ray distance sensors and a two-motor horizontal effector."""
from __future__ import annotations
import numpy as np


class CategoricalBody:
    def __init__(self, n_sensors=7, half_angle=np.pi / 6, max_range=20.0, x=0.0):
        self.n_sensors = int(n_sensors)
        self.half_angle = float(half_angle)
        self.max_range = float(max_range)
        self.x = float(x)

    def sensor_rays(self):
        if self.n_sensors == 1:
            angles = [0.0]
        else:
            angles = np.linspace(-self.half_angle, self.half_angle, self.n_sensors)
        # angle 0 = straight up (+y); +angle tilts toward +x
        return [np.array([np.sin(a), np.cos(a)]) for a in angles]

    def sense(self, obj) -> np.ndarray:
        origin = np.array([self.x, 0.0])
        out = np.zeros(self.n_sensors)
        for i, d in enumerate(self.sensor_rays()):
            dist = obj.distance_along(origin, d)
            if dist is not None and dist <= self.max_range:
                out[i] = (self.max_range - dist) / self.max_range
        return out

    def act(self, motor_output, dt, gain) -> None:
        m = np.asarray(motor_output, dtype=float)
        self.x = self.x + gain * (m[0] - m[1]) * dt

    def distance_to(self, obj) -> float:
        return float(abs(self.x - obj.center[0]))
