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
