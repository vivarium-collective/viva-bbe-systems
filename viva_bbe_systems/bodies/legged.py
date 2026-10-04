"""Legged body: one leg + foot + body dynamics (after Beer & Gallagher 1992).

The leg-controller of the Beer & Gallagher (1992) walker: a single leg pivots
about the body through an angle theta, and a foot is either up or down.

  * Foot DOWN (stance): the foot is planted, so swinging the leg backward
    pushes the body forward.  Body displacement is tied to the leg's angular
    change, and the body has no inertia of its own.
  * Foot UP (swing): the leg swings freely; the body only coasts, its
    velocity decaying by `friction` each step.

This is a clean-room reimplementation; the constants below are our own
choices, not values copied from the paper.

Sign convention: positive theta is the backward-swung leg.  The CTRNN
effectors drive omega = omega_gain * (o_bs - o_fs), so o_bs > o_fs swings the
leg backward (+d_theta).  With the foot planted that moves the body FORWARD,
i.e. x increases by d_theta * leg_length.
"""

from math import pi

import numpy as np


class LeggedBody:
    def __init__(self, angle_range=(-pi / 6, pi / 6), leg_length=15.0,
                 omega_gain=1.0, friction=0.9, foot_threshold=0.5):
        self.angle_range = tuple(angle_range)   # leg angle limits (rad)
        self.leg_length = leg_length            # lever arm: angle -> distance
        self.omega_gain = omega_gain            # effector output -> rad/time
        self.friction = friction                # per-step coasting decay
        self.foot_threshold = foot_threshold    # foot-down if o_foot > this
        self.reset()

    def reset(self, angle=0.0):
        self.angle = float(angle)
        self.foot_down = False
        self.x = 0.0
        self.vx = 0.0
        self.omega = 0.0

    def sense(self):
        """Leg angle normalized to [-1, 1] over angle_range (proprioception)."""
        lo, hi = self.angle_range
        return float(np.clip(2.0 * (self.angle - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    def _clamp(self, a):
        return min(max(a, self.angle_range[0]), self.angle_range[1])

    def act(self, o_foot, o_bs, o_fs, dt):
        self.foot_down = bool(o_foot > self.foot_threshold)
        self.omega = self.omega_gain * (o_bs - o_fs)
        new = self._clamp(self.angle + self.omega * dt)
        if self.foot_down:
            # planted foot: backward leg swing (+d_theta) drives body forward.
            # Clamped, so at the limit d_theta = 0 and the body stops.
            d_theta = new - self.angle
            self.x += d_theta * self.leg_length
            self.vx = d_theta * self.leg_length / dt
        else:
            self.vx *= self.friction
            self.x += self.vx * dt
        self.angle = new
