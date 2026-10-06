"""Two-nutrient internal metabolism (Agmon & Beer 2014)."""
import numpy as np


class Metabolism:
    def __init__(self, levels=(5.0, 5.0), eat_rate=0.07, drain_rate=0.0035, cap=10.0):
        # eat_rate 0.05 (was 0.02): a faster agent spends fewer steps inside a
        # resource, so each pass must replenish more. drain_rate 0.0035 (was
        # 0.0045): a longer nutrient lifetime gives travel headroom for the
        # SPREAD layouts. Together these keep spread foraging survivable.
        self._levels = np.asarray(levels, dtype=float).copy()
        self.eat_rate = eat_rate
        self.drain_rate = drain_rate
        self.cap = cap

    def step(self, inside_a: bool, inside_b: bool) -> None:
        self._levels += self.eat_rate * np.array([bool(inside_a), bool(inside_b)])
        self._levels -= self.drain_rate
        np.clip(self._levels, 0.0, self.cap, out=self._levels)

    @property
    def levels(self) -> np.ndarray:
        return self._levels

    @property
    def alive(self) -> bool:
        return bool(np.all(self._levels > 0))
