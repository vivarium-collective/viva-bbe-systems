"""Relative-size categorization fitness: catch obj2 iff larger than obj1."""
from __future__ import annotations
import numpy as np
from ..bodies.relational_genome import decode_agent, RelGenomeSpec


def _make_pairs(seed=0, offsets=(-4.0, 0.0, 4.0)):
    """Deterministic balanced battery of (s1, s2, offset2). Sizes in [2,6],
    |s2-s1|>=1. Each size pair appears in both orders (catch/avoid balanced)."""
    rng = np.random.default_rng(seed)
    base = [(2.0, 3.0), (2.0, 4.5), (3.0, 4.0), (3.5, 5.5), (4.0, 6.0), (2.5, 6.0)]
    pairs = []
    for i, (a, b) in enumerate(base):
        for lo_first in (True, False):
            s1, s2 = (a, b) if lo_first else (b, a)
            off = float(offsets[int(rng.integers(len(offsets)))])
            pairs.append((s1, s2, off))
    return pairs


SIZE_PAIRS = _make_pairs(seed=0)


def relational_fitness(genome, spec: RelGenomeSpec | None = None, *, record=False,
                       dt=0.1, scale=3.0) -> float:
    spec = spec or RelGenomeSpec()
    scores = []
    try:
        for s1, s2, off in SIZE_PAIRS:
            agent = decode_agent(genome, spec, dt=dt)
            out = agent.run_trial(s1, s2, offset2=off, dt=dt)
            fd = float(out["final_distance"])
            if out["should_catch"]:
                scores.append(scale / (scale + fd))
            else:
                scores.append(min(fd, spec.avoid_margin) / spec.avoid_margin)
    except (FloatingPointError, OverflowError):
        return 0.0
    f = float(np.mean(scores))
    return f if np.isfinite(f) else 0.0


def make_relational_fitness(spec: RelGenomeSpec | None = None, **kw):
    return lambda g: relational_fitness(g, spec, **kw)
