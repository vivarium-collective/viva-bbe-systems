"""Falling circle / diamond objects + ray-intersection geometry (Beer 2003)."""
from __future__ import annotations
import numpy as np

Shape = str  # "circle" | "diamond"


def ray_distance(origin, direction_unit, obj_center, size, shape) -> float | None:
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction_unit, dtype=float)
    c = np.asarray(obj_center, dtype=float)
    if shape == "circle":
        # |o + t d - c|^2 = size^2 ; smallest t >= 0
        f = o - c
        b = 2.0 * f.dot(d)
        cc = f.dot(f) - size * size
        disc = b * b - 4.0 * cc
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        for t in sorted(((-b - sq) / 2.0, (-b + sq) / 2.0)):
            if t >= 0:
                return float(t)
        return None
    if shape == "diamond":
        # L1 ball: |x-cx| + |y-cy| <= size. Intersect the ray with the 4 edges.
        best = None
        verts = [c + np.array(v) * size for v in
                 ((1, 0), (0, 1), (-1, 0), (0, -1))]
        edges = [(verts[i], verts[(i + 1) % 4]) for i in range(4)]
        for p0, p1 in edges:
            t = _ray_segment(o, d, p0, p1)
            if t is not None and (best is None or t < best):
                best = t
        return best
    raise ValueError(f"unknown shape {shape!r}")


def _ray_segment(o, d, p0, p1):
    # solve o + t d = p0 + u (p1-p0), t>=0, 0<=u<=1
    e = p1 - p0
    denom = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(denom) < 1e-12:
        return None
    diff = p0 - o
    t = (diff[0] * (-e[1]) - diff[1] * (-e[0])) / denom
    u = (d[0] * diff[1] - d[1] * diff[0]) / denom
    if t >= 0 and 0.0 <= u <= 1.0:
        return float(t)
    return None


class FallingObject:
    def __init__(self, center, vy, size, shape):
        self.center = np.array(center, dtype=float)
        self.vy = float(vy)
        self.size = float(size)
        self.shape = shape

    def step(self, dt):
        self.center = self.center + np.array([0.0, -self.vy * dt])

    def distance_along(self, origin, direction_unit):
        return ray_distance(origin, direction_unit, self.center, self.size, self.shape)
