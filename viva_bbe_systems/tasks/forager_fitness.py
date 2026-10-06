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


def _cfg(a, b, pos, angle, levels, ra=7.0, rb=7.0, drain=None, drain_swap_step=None):
    c = {"resource_a": tuple(map(float, a)), "resource_b": tuple(map(float, b)),
         "start_pos": tuple(map(float, pos)), "start_angle": float(angle),
         "init_levels": tuple(map(float, levels)),
         "radius_a": float(ra), "radius_b": float(rb)}
    if drain is not None:
        c["drain"] = tuple(map(float, drain))   # per-nutrient (A,B) drain rates
    if drain_swap_step is not None:
        c["drain_swap_step"] = int(drain_swap_step)  # step at which fast/slow swap
    return c


# separation tiers (units): a genuine SPREAD — far wider than the old ~22-42,
# cycled near (~35-45) / mid (~48-58) / far (~58-70) so every battery has all
# three. These are reachable only because the agent was sped up (max_thrust
# 0.025, longer-range gradient, lower drain — see ChemotacticForager /
# concentration / Metabolism); on the near layouts the agent is dropped AT one
# resource and must SEARCH for the other, on mid/far it starts between them.
_SEP_TIERS = ((42.0, 50.0), (50.0, 58.0), (55.0, 62.0))


def _place_pair(rng, sep, margin):
    """Place resources A,B exactly `sep` apart with a random orientation, both
    inside the plane. Resample the orientation/anchor to preserve the separation
    (no boundary-clip collapse), falling back to a shrunk sep if nothing fits."""
    lo, hi = margin, 100.0 - margin
    for _ in range(80):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        a = rng.uniform(lo, hi, 2)
        b = a + sep * np.array([np.cos(theta), np.sin(theta)])
        if lo <= b[0] <= hi and lo <= b[1] <= hi:
            return a, b
    return _place_pair(rng, sep * 0.9, margin)


def _make_configs(n=16, seed=7):
    """A deterministic battery of varied environments with a genuine SPREAD of
    layouts: the two resources differ in POSITION, ORIENTATION, SIZE (radius),
    and SEPARATION — cycled through near (~35-45), mid (~48-58), and far
    (~58-70) tiers (realised separations span ~38-65), each at a random orientation. On the near/mid layouts the
    agent starts AT one resource and must SEARCH for the other; on the far
    layouts it starts between them (both just within reach). A single tight
    circle cannot cover every layout — the agent must navigate to each
    resource's actual location, following a weakening gradient. Fixed-seed RNG →
    the same battery every run.
    """
    rng = np.random.default_rng(seed)
    cfgs = []
    for i in range(n):
        ra, rb = float(rng.uniform(4.0, 12.0)), float(rng.uniform(4.0, 12.0))
        margin = max(ra, rb) + 3.0
        tier = i % len(_SEP_TIERS)
        sep = float(rng.uniform(*_SEP_TIERS[tier]))
        a, b = _place_pair(rng, sep, margin)
        # all tiers are FAR now: start between the resources (both reachable),
        # off-centre. Resources are far enough that a full round-trip away nearly
        # starves the fast-draining nutrient, so the agent cannot maintain both
        # by equal-time circling — it must stay near the urgent (fast) resource
        # and dash to the other only when it senses that nutrient getting low.
        start = (a + b) / 2.0 + rng.uniform(-10.0, 10.0, 2)
        start = np.clip(start, 2.0, 98.0)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        levels = (float(rng.uniform(6.0, 10.0)), float(rng.uniform(6.0, 10.0)))  # start with buffer (high-drain far battery)
        # ASYMMETRIC per-nutrient drain: one nutrient drains up to ~2.6x faster
        # than the other, and WHICH one varies across the battery. The drain is
        # internal (invisible in the environment), so the only way to keep both
        # alive is to SENSE which nutrient is low and go to its resource — a
        # blind equal-time circler starves the faster-draining one. This is the
        # selection pressure for genuine state-dependent action switching
        # (Agmon & Beer 2014), not a fixed loop. Geometric-mean drain is held at
        # the base rate so overall survivability is comparable across trials.
        # HIGH base drain (0.006, was 0.0035): combined with the far separations,
        # a full round-trip away drains the camped nutrient by ~6-9, so the agent
        # has little slack — it must time its trips by sensing nutrient levels.
        base = 0.005
        # moderate asymmetry (ratio 1.3x - 1.9x, sqrt-split): the fast nutrient
        # nearly starves during a full round trip, so the agent must PRIORITIZE
        # it (visit it more / stay nearer) — equal-time circling kills it. Which
        # nutrient is fast alternates across the battery.
        ratio = float(rng.uniform(1.3, 1.8))
        k = np.sqrt(ratio)
        fast_a = (i % 2 == 0)
        drain = ((base * k, base / k) if fast_a else (base / k, base * k))
        cfgs.append(_cfg(a, b, start, angle, levels, ra, rb, drain=drain))
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
                           max_steps=max_steps, record=record, drain_rate=c.get("drain"),
                           drain_swap_step=c.get("drain_swap_step"))


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
