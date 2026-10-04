"""Longevity fitness for the chemotactic forager (Agmon & Beer 2014).

Fitness = mean survival over a battery of 16 VARIED environments (resource
positions + sizes differ per trial), normalized by max_steps.
"""
from __future__ import annotations

import numpy as np

from viva_bbe_systems.agents.forager_agent import ForagerAgent, MORPHOLOGY_SIZE
from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.environments.chemotaxis_resources import ChemotaxisEnv, Resource
from viva_bbe_systems.genome import decode


def _cfg(a, b, pos, angle, levels, ra=7.0, rb=7.0):
    return {"resource_a": tuple(map(float, a)), "resource_b": tuple(map(float, b)),
            "start_pos": tuple(map(float, pos)), "start_angle": float(angle),
            "init_levels": tuple(map(float, levels)),
            "radius_a": float(ra), "radius_b": float(rb)}


def _make_configs(n=16, seed=7):
    """A deterministic battery of varied environments: the two resources differ
    in POSITION and SIZE (radius) across trials, so an agent must evolve a
    general navigation/switching strategy rather than overfit a fixed layout.

    Separations are mostly ~22-42 units (a few land closer after the boundary
    clip); all < one nutrient-lifetime of travel
    with the start near the midpoint and varied orientation + initial hunger.
    Fixed-seed RNG → the same battery every run.
    """
    rng = np.random.default_rng(seed)
    cfgs = []
    for _ in range(n):
        ra, rb = float(rng.uniform(4.0, 12.0)), float(rng.uniform(4.0, 12.0))
        margin = max(ra, rb) + 3.0
        a = rng.uniform(margin, 100.0 - margin, 2)
        sep = rng.uniform(22.0, 42.0)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        b = np.clip(a + sep * np.array([np.cos(theta), np.sin(theta)]),
                    margin, 100.0 - margin)
        start = np.clip((a + b) / 2.0 + rng.uniform(-8.0, 8.0, 2), 2.0, 98.0)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        levels = (float(rng.uniform(2.0, 5.0)), float(rng.uniform(2.0, 5.0)))
        cfgs.append(_cfg(a, b, start, angle, levels, ra, rb))
    return cfgs


TRIAL_CONFIGS = _make_configs()


def env_from_config(c) -> ChemotaxisEnv:
    """Build the two-resource chemotaxis env for a config (honours per-resource
    radius; defaults to 7.0 for legacy configs without a radius)."""
    return ChemotaxisEnv(
        Resource(center=np.array(c["resource_a"], float),
                 radius=float(c.get("radius_a", 7.0)), signal="A"),
        Resource(center=np.array(c["resource_b"], float),
                 radius=float(c.get("radius_b", 7.0)), signal="B"))


def run_config(agent, c, *, max_steps=2500, record=False) -> dict:
    """Run `agent` on one config's environment (the single source of the env +
    run_trial call, shared by the fitness, the studies' viz, and the tests)."""
    return agent.run_trial(env_from_config(c), c["init_levels"],
                           start_pos=c["start_pos"], start_angle=c["start_angle"],
                           max_steps=max_steps, record=record)


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
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                r = run_config(agent, c, max_steps=max_steps)
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
