"""Catch-circles / avoid-diamonds fitness for the categorical-perception agent."""
from __future__ import annotations
import numpy as np
from ..bodies.categorical_genome import decode_agent, CatGenomeSpec


def catch_avoid_fitness(genome, spec: CatGenomeSpec, *,
                        offsets=(-6.0, -3.0, 0.0, 3.0, 6.0),
                        dt=0.1, steps=200, scale=3.0) -> float:
    """Mean reward over the catch-circle / avoid-diamond trial battery.

    Rewards are CONTINUOUS in the final horizontal distance `fd` (no plateau),
    so evolution has a gradient from any distance:
      - circle (catch):  scale / (scale + fd)   -> 1 at fd=0, decays with fd
      - diamond (avoid): fd / (scale + fd)       -> 0 at fd=0, -> 1 as fd grows
    A plateaued reward (reward 0 beyond a cutoff) traps the GA in a flee-both
    local optimum; the continuous form lets it climb back to actually catching.
    """
    scores = []
    for shape in ("circle", "diamond"):
        for off in offsets:
            try:
                # decode per trial so every trial starts from a fresh network
                agent = decode_agent(genome, spec, dt=dt)
                out = agent.run_trial(obj_offset=off, shape=shape, dt=dt, steps=steps)
            except FloatingPointError:
                scores.append(0.0)  # diverged CTRNN: worst reward for either shape
                continue
            fd = float(out["final_distance"])
            if shape == "circle":
                scores.append(scale / (scale + fd))   # closer is better
            else:
                scores.append(fd / (scale + fd))      # farther is better
    return float(np.mean(scores))


def make_fitness(spec: CatGenomeSpec, **kw):
    return lambda g: catch_avoid_fitness(g, spec, **kw)
