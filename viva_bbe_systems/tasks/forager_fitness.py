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
# 11 scenarios where foraging BOTH resources is physically reachable (the paper's
# intent). Separations are moderate (~25-35 units) so an agent can shuttle between
# the two within a nutrient's lifetime, and the start is near the midpoint; varied
# orientations and initial hunger force the agent to switch toward whichever
# nutrient is low. (Earlier configs had sep 70-113, > one nutrient-lifetime of
# travel, so switching was near-impossible and the GA got no gradient for it.)
TRIAL_CONFIGS = [
    _cfg((35, 50), (65, 50), (50, 50), 0.0, (5.0, 5.0)),         # horizontal, balanced
    _cfg((40, 40), (60, 60), (50, 50), 0.8, (3.0, 5.0)),         # diagonal, hungry A
    _cfg((40, 60), (60, 40), (50, 50), 2.3, (5.0, 3.0)),         # anti-diagonal, hungry B
    _cfg((32, 50), (62, 50), (45, 50), 0.0, (4.0, 4.0)),         # horizontal, offset start
    _cfg((50, 35), (50, 65), (50, 50), 1.5, (2.5, 5.0)),         # vertical, hungry A
    _cfg((44, 44), (70, 60), (55, 50), 3.0, (5.0, 2.5)),         # tilted, hungry B
    _cfg((38, 62), (62, 38), (50, 50), 5.5, (4.0, 3.0)),         # anti-diagonal, turned
    _cfg((60, 42), (35, 42), (48, 46), 3.9, (3.0, 3.0)),         # horizontal, facing away
    _cfg((40, 40), (64, 64), (52, 52), 1.0, (5.0, 4.0)),         # diagonal, mild hunger B
    _cfg((50, 38), (50, 68), (50, 52), 0.3, (3.5, 5.0)),         # vertical, hungry A
    _cfg((38, 58), (62, 42), (50, 50), 2.0, (4.5, 2.0)),         # diagonal, very hungry B
]


def longevity_fitness(genome, spec, morphology="M2", *, max_steps=5000,
                      balance_weight=0.0, nutrient_cap=10.0) -> float:
    """Mean survival over TRIAL_CONFIGS / max_steps, in [0, 1].

    The paper's fitness is pure longevity (``balance_weight=0.0``, the faithful
    default). Because the survival gradient is flat for agents that can't yet
    switch (they just die when the neglected nutrient runs out), evolution can
    stall before discovering switching. With ``balance_weight > 0`` each config
    also rewards the time-averaged MINIMUM nutrient level (normalized to [0,1]):
    an explicit, smooth gradient toward keeping BOTH resources topped up, which
    is exactly the switching behaviour. The shaped score stays in [0, 1+bw] and
    is reported transparently when used.
    """
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
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                r = agent.run_trial(env, c["init_levels"], start_pos=c["start_pos"],
                                    start_angle=c["start_angle"], max_steps=max_steps)
            score = r["survival"] / max_steps
            if balance_weight:
                lh = r["levels_hist"]  # (T, 2)
                min_nut = lh.min(axis=1) / nutrient_cap  # per-step min nutrient, [0,1]
                # time-average over the FULL horizon (steps after death count 0)
                score += balance_weight * float(min_nut.sum()) / max_steps
        except (FloatingPointError, OverflowError):
            score = 0.0
        total += score
    return float(total / len(TRIAL_CONFIGS))


def make_forager_fitness(spec, morphology="M2", **kw):
    return lambda genome: longevity_fitness(genome, spec, morphology, **kw)
