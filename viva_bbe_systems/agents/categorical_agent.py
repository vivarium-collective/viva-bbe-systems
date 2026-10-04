"""Plain-Python brain+body+environment loop for the categorical-perception agent."""
from __future__ import annotations
import numpy as np
from ..environments.falling_objects import FallingObject


class CategoricalAgent:
    def __init__(self, ctrnn, body, sensor_weights, motor_indices=(-2, -1),
                 motor_gain=5.0, catch_radius=1.5):
        self.ctrnn = ctrnn
        self.body = body
        self.sensor_weights = np.asarray(sensor_weights, dtype=float)
        self.motor_indices = tuple(motor_indices)
        self.motor_gain = float(motor_gain)
        self.catch_radius = float(catch_radius)

    def run_trial(self, obj_offset, shape, *, H=20.0, vy=1.0, dt=0.1, steps=200,
                  obj_size=3.0, start_x=0.0, record_outputs=False):
        self.ctrnn.reset(np.zeros(self.ctrnn.size))
        self.body.x = float(start_x)
        obj = FallingObject(center=np.array([float(obj_offset), H]), vy=vy,
                            size=obj_size, shape=shape)
        traj = np.zeros((steps, 2))
        outs, centers = [], []
        for t in range(steps):
            I = self.sensor_weights @ self.body.sense(obj)
            o = self.ctrnn.step(external_input=I)
            motor = np.array([o[self.motor_indices[0]], o[self.motor_indices[1]]])
            self.body.act(motor, dt=dt, gain=self.motor_gain)
            obj.step(dt)
            traj[t] = (self.body.x, obj.center[0])
            if record_outputs:
                outs.append(np.array(o, dtype=float))
                centers.append(obj.center.copy())
        final_distance = self.body.distance_to(obj)
        result = {"final_distance": final_distance, "trajectory": traj,
                  "caught": final_distance <= self.catch_radius}
        if record_outputs:
            result["outputs"] = np.array(outs).reshape(steps, self.ctrnn.size)
            result["obj_centers"] = np.array(centers).reshape(steps, 2)
        return result
