import numpy as np
from viva_bbe_systems.genome import GenomeSpec
from viva_bbe_systems.evolution import evolve


def test_ga_improves_on_trivial_fitness():
    # fitness maximized by driving the genome toward zero
    spec = GenomeSpec(size=2)

    def fitness(genome):
        return -float(np.sum(genome ** 2))

    out = evolve(fitness, spec, pop_size=30, generations=40, mutation_sd=0.5, seed=1)
    assert out["best_fitness"] > out["history"][0]
    assert out["best_fitness"] > -50.0  # got meaningfully close to zero


def test_ga_is_deterministic_under_seed():
    spec = GenomeSpec(size=2)
    f = lambda g: -float(np.sum(g ** 2))
    a = evolve(f, spec, seed=7, generations=10, pop_size=20)
    b = evolve(f, spec, seed=7, generations=10, pop_size=20)
    assert a["best_fitness"] == b["best_fitness"]


def test_evolve_rejects_elitism_above_pop_size():
    import pytest
    with pytest.raises(ValueError):
        evolve(lambda g: 0.0, GenomeSpec(size=2), pop_size=4, elitism=5, generations=1)
