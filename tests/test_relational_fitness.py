import numpy as np
from viva_bbe_systems.bodies.relational_genome import (
    RelGenomeSpec, rel_genome_length, decode_agent, random_rel_genome, rel_bounds)
from viva_bbe_systems.agents.relational_agent import RelationalAgent
from viva_bbe_systems.tasks.relational_fitness import (
    SIZE_PAIRS, relational_fitness, make_relational_fitness)

SPEC = RelGenomeSpec()


def test_decode_roundtrip():
    g = random_rel_genome(SPEC, np.random.default_rng(0))
    assert g.size == rel_genome_length(SPEC)
    lo, hi = rel_bounds(SPEC)
    assert lo.size == hi.size == g.size
    a = decode_agent(g, SPEC)
    assert isinstance(a, RelationalAgent)
    assert a.sensor_weights.shape == (6, 7)
    assert a.ctrnn.size == 6


def test_finite_and_deterministic():
    g = random_rel_genome(SPEC, np.random.default_rng(1))
    f1, f2 = relational_fitness(g), make_relational_fitness()(g)
    assert np.isfinite(f1) and f1 == f2 and 0.0 <= f1 <= 1.0


def test_divergent_genome_scores_zero_no_raise():
    g = np.zeros(rel_genome_length(SPEC))
    g[:6] = 1e-9                      # tau -> 0
    g[12:48] = 1e6                    # huge recurrent weights
    g[48:] = 5.0
    assert relational_fitness(g) < 0.05


def test_battery_balanced_and_separated():
    sc = sum(s2 > s1 for s1, s2, _ in SIZE_PAIRS)
    sa = sum(s2 < s1 for s1, s2, _ in SIZE_PAIRS)
    assert sc == sa and 12 <= len(SIZE_PAIRS) <= 20
    assert all(abs(s2 - s1) >= 1.0 and 2 <= min(s1, s2) and max(s1, s2) <= 6
               for s1, s2, _ in SIZE_PAIRS)


def test_never_move_scores_chance():
    g = np.zeros(rel_genome_length(SPEC))
    g[:6] = 1.0
    f = relational_fitness(g)
    assert f < 0.75, f
