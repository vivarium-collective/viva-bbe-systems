"""Chemotaxis environment: two exponential-gradient resources (Agmon & Beer 2014)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Resource:
    center: np.ndarray
    radius: float = 7.0
    signal: str = ""


def _dist(resource, point):
    return float(np.linalg.norm(np.asarray(point, float) - np.asarray(resource.center, float)))


def concentration(resource, point):
    """10 within the radius, else 10*exp(-0.05*d)."""
    d = _dist(resource, point)
    if d <= resource.radius:
        return 10.0
    return float(10.0 * np.exp(-0.05 * d))


class ChemotaxisEnv:
    def __init__(self, resource_a, resource_b, width=100.0, height=100.0):
        self.resources = {resource_a.signal: resource_a, resource_b.signal: resource_b}
        self.width = width
        self.height = height

    def _get(self, signal):
        if signal not in self.resources:
            raise ValueError(f"unknown signal {signal!r}")
        return self.resources[signal]

    def conc_at(self, point, signal):
        return concentration(self._get(signal), point)

    def inside(self, point, signal):
        r = self._get(signal)
        return _dist(r, point) <= r.radius

    def clamp(self, point):
        p = np.asarray(point, float)
        return np.array([np.clip(p[0], 0.0, self.width), np.clip(p[1], 0.0, self.height)])
