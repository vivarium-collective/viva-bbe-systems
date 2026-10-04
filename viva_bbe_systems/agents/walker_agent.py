"""Walker agent runner: CTRNN central pattern generator + one-leg body (closed loop)."""
from __future__ import annotations

import numpy as np

from viva_bbe_systems.bodies.legged import LeggedBody
from viva_bbe_systems.ctrnn import CTRNN

N_INTER = 2
# 2 interneurons + 3 motors (foot, backward swing, forward swing); the leg angle feeds neuron 0
WALKER_SIZE = N_INTER + 3


def make_walker(ctrnn=None, **body_kwargs):
    """Build a WalkerAgent (zero-weight CTRNN of WALKER_SIZE if none given)."""
    return WalkerAgent(ctrnn if ctrnn is not None else CTRNN(WALKER_SIZE),
                       LeggedBody(**body_kwargs))


class WalkerAgent:
    def __init__(self, ctrnn, body, *, foot_index=-3, bs_index=-2, fs_index=-1,
                 angle_input_index=0):
        self.ctrnn = ctrnn
        self.body = body
        self.foot_index = foot_index
        self.bs_index = bs_index
        self.fs_index = fs_index
        self.angle_input_index = angle_input_index

    def run_trial(self, *, start_angle=0.0, dt=0.1, steps=500, record=False):
        ctrnn, body = self.ctrnn, self.body
        ctrnn.reset(np.zeros(ctrnn.size))
        body.reset(angle=start_angle)

        xs, angles, foots, vxs, outs = [], [], [], [], []
        for _ in range(steps):
            full = np.zeros(ctrnn.size)
            full[self.angle_input_index] = body.sense()
            o = ctrnn.step(full)
            body.act(o[self.foot_index], o[self.bs_index], o[self.fs_index], dt)
            xs.append(body.x)
            angles.append(body.angle)
            foots.append(body.foot_down)
            vxs.append(body.vx)
            if record:
                outs.append(np.array(o, float))

        result = {
            "distance": float(body.x),
            "x_hist": np.array(xs, float),
            "angle_hist": np.array(angles, float),
            "foot_hist": np.array(foots, bool),
            "vx_hist": np.array(vxs, float),
        }
        if record:
            result["outputs"] = np.array(outs).reshape(-1, ctrnn.size)
        return result
