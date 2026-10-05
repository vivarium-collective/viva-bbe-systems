"""Relative-size categorization fitness: catch obj2 iff larger than obj1."""
from __future__ import annotations
import numpy as np
from ..bodies.relational_genome import decode_agent, RelGenomeSpec


def _make_pairs(seed=0, offsets=(-4.0, 0.0, 4.0), sizes=(2.0, 3.0, 4.0, 5.0, 6.0),
                reps=2):
    """Deterministic role-balanced-per-size battery of (s1, s2, offset2).

    Uses a SHARED size grid and the ordered pairs at the minimum separation
    (|s2-s1| = 1.0), so every interior size appears as s2 AND as s1 in both the
    catch and avoid roles. Only the two extreme sizes are single-role. Including
    wider-gap pairs (the full ordered grid) lets an absolute-s2 threshold reach
    ~0.83, because the extremes then dominate; the adjacent-only set bounds any
    single-object threshold at 0.625, so memory of s1 is required.
    Each pair repeats `reps` times; offset2 cycles over `offsets`."""
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(offsets)))
    pairs = [(a, b) for a in sizes for b in sizes if abs(abs(b - a) - 1.0) < 1e-9]
    pairs = pairs * reps
    return [(a, b, float(offsets[(start + k) % len(offsets)]))
            for k, (a, b) in enumerate(pairs)]


SIZE_PAIRS = _make_pairs(seed=0)


def relational_fitness(genome, spec: RelGenomeSpec | None = None, *, record=False,
                       dt=0.1, correct_weight=0.5) -> float:
    """Symmetric, accuracy-aligned reward. Per trial, let the normalized
    closeness be ``c = max(0, 1 - final_distance/avoid_margin)`` (1 = on top of
    obj2, 0 = avoid_margin away). A should-catch trial wants c high, a should-avoid
    trial wants c low (= ``1 - c``). Each trial scores
    ``correct_weight*correctness + (1-correct_weight)*position``, where
    correctness is 1 iff the catch/avoid decision matched. This is SYMMETRIC:
    always-catch and always-avoid both score ~0.5, so the only way above 0.5 is
    to actually catch the larger ones AND avoid the smaller ones — which, on the
    role-balanced battery, requires memory of obj1. (The old asymmetric shaping
    paid for merely being *near* obj2, so evolution collapsed to always-avoid.)
    """
    spec = spec or RelGenomeSpec()
    D = spec.avoid_margin
    scores = []
    try:
        for s1, s2, off in SIZE_PAIRS:
            # intentional: fresh decode per trial -> clean CTRNN state
            agent = decode_agent(genome, spec, dt=dt)
            out = agent.run_trial(s1, s2, offset2=off, dt=dt)
            fd = float(out["final_distance"])
            c = max(0.0, 1.0 - fd / D)                 # closeness to obj2 in [0,1]
            position = c if out["should_catch"] else (1.0 - c)
            correct = 1.0 if out["correct"] else 0.0
            scores.append(correct_weight * correct + (1.0 - correct_weight) * position)
    except (FloatingPointError, OverflowError):
        return 0.0
    f = float(np.mean(scores))
    return f if np.isfinite(f) else 0.0


def make_relational_fitness(spec: RelGenomeSpec | None = None, **kw):
    return lambda g: relational_fitness(g, spec, **kw)
