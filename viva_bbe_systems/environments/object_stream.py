"""Two sequentially presented falling objects (relational categorization,
Williams, Beer & Gasser 2008). Only one object is ever visible; the ISI is
empty, so the agent must remember object 1's size to judge object 2."""
from __future__ import annotations
from .falling_objects import FallingObject


class TwoObjectStream:
    def __init__(self, s1, s2, *, offset1=0.0, offset2=0.0, H=20.0, vy=1.0,
                 phase1_steps=160, isi_steps=40, phase2_steps=160, dt=1.0):
        self.s1, self.s2 = float(s1), float(s2)
        self.offset1, self.offset2 = float(offset1), float(offset2)
        self.H, self.vy, self.dt = float(H), float(vy), float(dt)
        self.phase1_steps = int(phase1_steps)
        self.isi_steps = int(isi_steps)
        self.phase2_steps = int(phase2_steps)

    @property
    def total_steps(self) -> int:
        return self.phase1_steps + self.isi_steps + self.phase2_steps

    @property
    def sizes(self):
        return (self.s1, self.s2)

    def phase(self, step: int) -> str:
        if step < 0:
            raise ValueError("step must be >= 0")
        if step < self.phase1_steps:
            return "obj1"
        if step < self.phase1_steps + self.isi_steps:
            return "isi"
        if step < self.total_steps:
            return "obj2"
        return "done"

    def visible(self, step: int):
        """Pure function of step: a fresh FallingObject, or None."""
        ph = self.phase(step)
        if ph == "obj1":
            k, size, x = step, self.s1, self.offset1
        elif ph == "obj2":
            k, size, x = step - self.phase1_steps - self.isi_steps, self.s2, self.offset2
        else:
            return None
        return FallingObject([x, self.H - self.vy * self.dt * k], self.vy, size, "circle")
