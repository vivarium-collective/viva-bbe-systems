"""Walk fitness for the legged-locomotion CPG: net signed forward distance."""
from __future__ import annotations

import numpy as np

from viva_bbe_systems.agents.walker_agent import WALKER_SIZE, make_walker
from viva_bbe_systems.genome import decode


def walk_fitness(genome, spec, *, steps=500, dt=0.1) -> float:
    """Net signed forward distance (body.x) over one flat-ground trial.

    Signed, not abs: backward walking scores below a stationary agent.
    Numerical blow-ups score 0.
    """
    if spec.size != WALKER_SIZE:
        raise ValueError(f"spec.size {spec.size} != {WALKER_SIZE}")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            agent = make_walker(decode(genome, spec))
            d = agent.run_trial(dt=dt, steps=steps)["distance"]
    except (FloatingPointError, OverflowError):
        return 0.0
    return float(d) if np.isfinite(d) else 0.0


def make_walk_fitness(spec, **kw):
    return lambda genome: walk_fitness(genome, spec, **kw)
