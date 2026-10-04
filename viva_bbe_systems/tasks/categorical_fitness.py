"""Catch-circles / avoid-diamonds fitness for the categorical-perception agent."""
from __future__ import annotations
import numpy as np
from ..bodies.categorical_genome import decode_agent, CatGenomeSpec


def catch_avoid_fitness(genome, spec: CatGenomeSpec, *,
                        offsets=(-6.0, -3.0, 0.0, 3.0, 6.0),
                        dt=0.1, steps=200, span=12.0) -> float:
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
            fd = min(out["final_distance"], span)
            if shape == "circle":
                scores.append((span - fd) / span)     # closer is better
            else:
                scores.append(fd / span)              # farther is better
    return float(np.mean(scores))


def make_fitness(spec: CatGenomeSpec, **kw):
    return lambda g: catch_avoid_fitness(g, spec, **kw)
