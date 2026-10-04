"""Walk fitness for the legged-locomotion CPG: net signed forward distance."""
from __future__ import annotations

import numpy as np

from viva_bbe_systems.agents.walker_agent import WALKER_SIZE, make_walker
from viva_bbe_systems.genome import decode


def _productive_strides(foot_hist, x_hist, *, progress_eps=0.3):
    """Count stance phases (contiguous foot-down runs) during which the body
    actually advanced by more than ``progress_eps``. A single lunge scores 1;
    marching-in-place or wiggling the foot while coasting scores 0."""
    foot = np.asarray(foot_hist, bool)
    x = np.asarray(x_hist, float)
    n = len(foot)
    n_prod = 0
    t = 0
    while t < n:
        if foot[t]:
            start = t
            while t < n and foot[t]:
                t += 1
            if x[t - 1] - x[start] > progress_eps:  # body advanced during stance
                n_prod += 1
        else:
            t += 1
    return n_prod


def walk_fitness(genome, spec, *, steps=500, dt=0.1, rhythm_weight=0.0,
                 stride_cap=8, progress_eps=0.3) -> float:
    """Net signed forward distance (body.x) over one flat-ground trial.

    Signed, not abs: backward walking scores below a stationary agent.
    Numerical blow-ups score 0.

    ``rhythm_weight`` (opt-in; 0.0 = faithful pure-distance default) shapes
    toward a RHYTHMIC gait: the forward distance is multiplied by
    ``1 + rhythm_weight*min(n_productive_strides, stride_cap)``, so an agent
    that repeatedly plants-swings-recovers (a real walk) beats a single lunge
    that then coasts. Only strides that advance the body count, so stepping in
    place earns nothing. Applied only when distance is positive.
    """
    if spec.size != WALKER_SIZE:
        raise ValueError(f"spec.size {spec.size} != {WALKER_SIZE}")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            agent = make_walker(decode(genome, spec))
            r = agent.run_trial(dt=dt, steps=steps, record=rhythm_weight > 0.0)
            d = r["distance"]
            if rhythm_weight > 0.0 and np.isfinite(d) and d > 0.0:
                n_prod = _productive_strides(r["foot_hist"], r["x_hist"],
                                             progress_eps=progress_eps)
                d = d * (1.0 + rhythm_weight * min(n_prod, stride_cap))
    except (FloatingPointError, OverflowError):
        return 0.0
    return float(d) if np.isfinite(d) else 0.0


def make_walk_fitness(spec, **kw):
    return lambda genome: walk_fitness(genome, spec, **kw)
