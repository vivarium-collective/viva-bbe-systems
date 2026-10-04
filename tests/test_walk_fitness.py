import numpy as np

from viva_bbe_systems.agents.walker_agent import WALKER_SIZE
from viva_bbe_systems.genome import GenomeSpec, encode, genome_length
from viva_bbe_systems.tasks.walk_fitness import make_walk_fitness, walk_fitness
from viva_bbe_systems.ctrnn import CTRNN

SPEC = GenomeSpec(WALKER_SIZE)


def _cpg():
    """Same hand-built CPG as tests/test_walker_agent.py."""
    c = CTRNN(WALKER_SIZE, dt=0.1)
    w = np.zeros((WALKER_SIZE, WALKER_SIZE))
    w[0, 0] = w[1, 1] = 4.5
    w[0, 1] = -8.0
    w[1, 0] = 8.0
    w[2, 0] = 10.0
    w[3, 0] = 10.0
    w[4, 1] = 10.0
    c.weights = w
    c.theta = -0.5 * w.sum(axis=1)
    c.tau = np.array([1.0, 1.0, 0.2, 0.2, 0.2])
    return c


def _stationary():
    c = _cpg()
    c.weights[:] = 0.0
    c.theta[:] = -20.0  # all outputs ~0: no motor drive
    return encode(c, SPEC)


def _backward():
    c = _cpg()
    w = c.weights
    w[3, 0] = 0.0; w[4, 1] = 0.0
    w[3, 1] = 10.0   # swap swing roles: swing back while foot is up
    w[4, 0] = 10.0
    c.theta = -0.5 * w.sum(axis=1)
    return encode(c, SPEC)


def test_finite_and_deterministic():
    g = np.random.default_rng(0).uniform(-5, 5, genome_length(SPEC))
    g[:WALKER_SIZE] = np.abs(g[:WALKER_SIZE]) + 0.5
    a, b = walk_fitness(g, SPEC), walk_fitness(g, SPEC)
    assert np.isfinite(a) and a == b


def test_divergent_genome_guarded():
    g = _stationary()
    g[:WALKER_SIZE] = 1e-300
    g[2 * WALKER_SIZE:] = 1e300
    assert walk_fitness(g, SPEC) == 0.0


def test_stationary_scores_zero():
    assert abs(walk_fitness(_stationary(), SPEC)) < 1e-6


def test_forward_cpg_positive():
    assert walk_fitness(encode(_cpg(), SPEC), SPEC) > 0


def test_backward_not_better_than_stationary():
    assert walk_fitness(_backward(), SPEC) <= walk_fitness(_stationary(), SPEC)


def test_make_walk_fitness_closure():
    g = encode(_cpg(), SPEC)
    assert make_walk_fitness(SPEC, steps=200)(g) == walk_fitness(g, SPEC, steps=200)
