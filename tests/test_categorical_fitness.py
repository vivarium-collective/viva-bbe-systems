import numpy as np
from viva_bbe_systems.bodies.categorical_genome import (
    CatGenomeSpec, cat_genome_length, random_cat_genome)
from viva_bbe_systems.tasks.categorical_fitness import catch_avoid_fitness


def test_fitness_is_finite_and_deterministic():
    spec = CatGenomeSpec()
    g = random_cat_genome(spec, np.random.default_rng(1))
    a = catch_avoid_fitness(g, spec)
    b = catch_avoid_fitness(g, spec)
    assert np.isfinite(a) and a == b


def test_nonmover_scores_below_perfect_tracker():
    spec = CatGenomeSpec()
    nonmover = np.zeros(cat_genome_length(spec))  # zero weights/biases => no motion
    nonmover[:spec.n_neurons] = 1.0  # valid tau (tau=0 is degenerate and diverges)
    score = catch_avoid_fitness(nonmover, spec)
    # a non-mover catches nothing and avoids nothing perfectly; bounded, finite, < max
    assert np.isfinite(score)
    assert score < 1.0


def test_diverging_genome_scores_low_instead_of_raising():
    spec = CatGenomeSpec()
    g = random_cat_genome(spec, np.random.default_rng(2))
    g[:spec.n_neurons] = 1e-9  # near-zero tau => CTRNN diverges
    score = catch_avoid_fitness(g, spec)
    assert np.isfinite(score)
    assert score <= 0.1
    assert score == catch_avoid_fitness(g, spec)
