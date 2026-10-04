"""Generic genetic algorithm that evolves any genome scored by a fitness fn.

Model-agnostic: all task/composite logic lives in `fitness_fn`, which maps a
flat genome to a scalar score (higher = better). A model evolves its BBE
composite by passing a fitness fn that instantiates the composite from the
genome, runs it, and returns the task score (see tasks/fitness.py)."""
from __future__ import annotations
import numpy as np
from .genome import GenomeSpec, random_genome, clip_to_bounds


def evolve(fitness_fn, spec: GenomeSpec, *, pop_size=50, generations=30,
           mutation_sd=0.5, seed=0, elitism=1) -> dict:
    if not (1 <= pop_size):
        raise ValueError("pop_size must be >= 1")
    if not (0 <= elitism <= pop_size):
        raise ValueError("elitism must be in [0, pop_size]")
    rng = np.random.default_rng(seed)
    pop = np.array([random_genome(spec, rng) for _ in range(pop_size)])
    fits = np.array([fitness_fn(g) for g in pop])
    history = [float(fits.max())]

    for _ in range(generations):
        order = np.argsort(fits)[::-1]       # best first
        pop, fits = pop[order], fits[order]
        new = [pop[i].copy() for i in range(elitism)]   # keep elites
        while len(new) < pop_size:
            # rank-based (truncation) selection from the top half
            parent = pop[rng.integers(0, max(1, pop_size // 2))]
            child = clip_to_bounds(parent + rng.normal(0.0, mutation_sd, parent.shape), spec)
            new.append(child)
        pop = np.array(new)
        fits = np.array([fitness_fn(g) for g in pop])
        history.append(float(fits.max()))

    best = int(np.argmax(fits))
    return {"best_genome": pop[best], "best_fitness": float(fits[best]), "history": history}


def evolve_flat(fitness_fn, length, lo, hi, *, pop_size=50, generations=30,
                mutation_sd=0.5, seed=0, elitism=1, record_every=None) -> dict:
    """Same GA as `evolve`, but with an explicit flat genome length and bounds
    (`lo`/`hi` may be scalars or length-`length` arrays).

    If `record_every` is a positive int, the result also carries a
    `checkpoints` list of `{"gen", "genome", "fitness"}` for the best genome at
    generation 0 and every `record_every` generations (and the final one) — used
    to visualize how the best agent improves over evolution. Default None keeps
    the original behavior (no checkpoints, same result shape)."""
    if not (1 <= pop_size):
        raise ValueError("pop_size must be >= 1")
    if not (0 <= elitism <= pop_size):
        raise ValueError("elitism must be in [0, pop_size]")
    lo = np.broadcast_to(np.asarray(lo, dtype=float), (length,))
    hi = np.broadcast_to(np.asarray(hi, dtype=float), (length,))
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lo, hi, size=(pop_size, length))
    fits = np.array([fitness_fn(g) for g in pop])
    history = [float(fits.max())]
    checkpoints = []

    def _record(gen):
        if record_every:
            b = int(np.argmax(fits))
            checkpoints.append({"gen": gen, "genome": pop[b].copy(),
                                "fitness": float(fits[b])})

    _record(0)
    for gen in range(1, generations + 1):
        order = np.argsort(fits)[::-1]
        pop, fits = pop[order], fits[order]
        new = [pop[i].copy() for i in range(elitism)]
        while len(new) < pop_size:
            parent = pop[rng.integers(0, max(1, pop_size // 2))]
            child = np.clip(parent + rng.normal(0.0, mutation_sd, parent.shape), lo, hi)
            new.append(child)
        pop = np.array(new)
        fits = np.array([fitness_fn(cand) for cand in pop])
        history.append(float(fits.max()))
        if record_every and (gen % record_every == 0 or gen == generations):
            _record(gen)

    best = int(np.argmax(fits))
    result = {"best_genome": pop[best], "best_fitness": float(fits[best]), "history": history}
    if record_every:
        result["checkpoints"] = checkpoints
    return result
