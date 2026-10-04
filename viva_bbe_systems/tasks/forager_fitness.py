"""Longevity fitness for the chemotactic forager (Agmon & Beer 2014).

Fitness = mean survival over 11 fixed trial configs, normalized by max_steps.
"""
from __future__ import annotations

import numpy as np

from viva_bbe_systems.agents.forager_agent import ForagerAgent, MORPHOLOGY_SIZE
from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.environments.chemotaxis_resources import ChemotaxisEnv, Resource
from viva_bbe_systems.genome import decode


def _cfg(a, b, pos, angle, levels):
    return {"resource_a": a, "resource_b": b, "start_pos": pos,
            "start_angle": angle, "init_levels": levels}


# Hard-coded (deterministic): varied separations, starts, angles, and hunger.
TRIAL_CONFIGS = [
    _cfg((20, 20), (80, 80), (50, 50), 0.0, (5.0, 5.0)),         # far, balanced
    _cfg((20, 80), (80, 20), (50, 50), 1.5, (3.0, 5.0)),         # far, hungry A
    _cfg((15, 50), (85, 50), (50, 20), 3.1, (5.0, 3.0)),         # far, hungry B
    _cfg((30, 30), (70, 70), (50, 50), 4.7, (4.0, 4.0)),         # mid
    _cfg((40, 40), (60, 60), (50, 80), 0.8, (2.5, 5.0)),         # close, hungry A
    _cfg((45, 55), (60, 45), (20, 20), 2.3, (5.0, 2.5)),         # close, hungry B
    _cfg((25, 75), (50, 85), (80, 30), 5.5, (4.0, 3.0)),         # mid, near edge
    _cfg((70, 25), (30, 25), (50, 75), 3.9, (3.0, 3.0)),         # mid horizontal
    _cfg((10, 10), (90, 90), (90, 10), 1.0, (5.0, 4.0)),         # extreme far
    _cfg((55, 30), (55, 70), (15, 50), 0.3, (3.5, 5.0)),         # vertical pair
    _cfg((35, 65), (65, 35), (80, 80), 2.0, (4.5, 2.0)),         # diagonal, hungry B
]


def longevity_fitness(genome, spec, morphology="M2", *, max_steps=5000) -> float:
    """Mean survival over TRIAL_CONFIGS / max_steps, in [0, 1]."""
    if spec.size != MORPHOLOGY_SIZE[morphology]:
        raise ValueError(f"spec.size {spec.size} != {MORPHOLOGY_SIZE[morphology]} for {morphology}")
    net = decode(genome, spec)
    agent = ForagerAgent(net, ChemotacticForager(morphology))
    total = 0.0
    for c in TRIAL_CONFIGS:
        env = ChemotaxisEnv(
            Resource(center=np.array(c["resource_a"], float), signal="A"),
            Resource(center=np.array(c["resource_b"], float), signal="B"))
        try:
            with np.errstate(all="raise"):
                r = agent.run_trial(env, c["init_levels"], start_pos=c["start_pos"],
                                    start_angle=c["start_angle"], max_steps=max_steps)
            surv = r["survival"]
        except (FloatingPointError, OverflowError):
            surv = 0
        total += surv
    return float(total / (len(TRIAL_CONFIGS) * max_steps))


def make_forager_fitness(spec, morphology="M2", **kw):
    return lambda genome: longevity_fitness(genome, spec, morphology, **kw)
