"""Brain+body+environment loop for relational categorization (Williams, Beer &
Gasser 2008): two sequential objects; catch obj2 iff larger than obj1. The ISI
and phase 2 hide obj1, so its size must be held in CTRNN state."""
from __future__ import annotations
import numpy as np
from ..ctrnn import CTRNN
from ..bodies.categorical_perception import CategoricalBody
from ..environments.object_stream import TwoObjectStream

RELATIONAL_SIZE = 8  # 4 interneurons + 2 motor + headroom for the memory trace


class RelationalAgent:
    def __init__(self, ctrnn, body, sensor_weights, *, motor_indices=(-2, -1),
                 motor_gain=5.0, catch_radius=2.0, avoid_margin=6.0):
        self.ctrnn = ctrnn
        self.body = body
        self.sensor_weights = np.asarray(sensor_weights, dtype=float)
        self.motor_indices = tuple(motor_indices)
        self.motor_gain = float(motor_gain)
        self.catch_radius = float(catch_radius)
        self.avoid_margin = float(avoid_margin)

    def run_trial(self, s1, s2, *, offset1=0.0, offset2=0.0, dt=0.1, record=False):
        self.ctrnn.reset(np.zeros(self.ctrnn.size))
        self.body.x = 0.0
        stream = TwoObjectStream(s1, s2, offset1=offset1, offset2=offset2)
        T = stream.total_steps
        x_hist = np.zeros(T)
        outs, obj_x, phases = [], [], []
        for t in range(T):
            obj = stream.visible(t)
            shadow = (self.body.sense(obj) if obj is not None
                      else np.zeros(self.body.n_sensors))
            o = self.ctrnn.step(external_input=self.sensor_weights @ shadow)
            motor = np.array([o[self.motor_indices[0]], o[self.motor_indices[1]]])
            self.body.act(motor, dt, self.motor_gain)
            x_hist[t] = self.body.x
            if record:
                outs.append(np.array(o, dtype=float))
                obj_x.append(float(obj.center[0]) if obj is not None else np.nan)
                phases.append(stream.phase(t))
        final_distance = float(abs(self.body.x - stream.offset2))
        caught = bool(final_distance <= self.catch_radius)
        should_catch = bool(s2 > s1)
        result = {"caught": caught, "final_distance": final_distance,
                  "should_catch": should_catch, "correct": caught == should_catch,
                  "x_hist": x_hist}
        if record:
            result["outputs"] = np.array(outs).reshape(T, self.ctrnn.size)
            result["obj_x_hist"] = np.array(obj_x)
            result["phase_hist"] = phases
        return result


def make_relational(ctrnn=None, n_sensors=7):
    net = ctrnn if ctrnn is not None else CTRNN(RELATIONAL_SIZE, dt=0.1)
    return RelationalAgent(net, CategoricalBody(n_sensors=n_sensors), np.zeros((RELATIONAL_SIZE, n_sensors)))
