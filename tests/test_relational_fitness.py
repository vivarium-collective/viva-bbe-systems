import numpy as np
import pytest
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
    assert a.sensor_weights.shape == (8, 7)
    assert a.ctrnn.size == 8


def test_finite_and_deterministic():
    g = random_rel_genome(SPEC, np.random.default_rng(1))
    f1, f2 = relational_fitness(g), make_relational_fitness()(g)
    assert np.isfinite(f1) and f1 == f2 and 0.0 <= f1 <= 1.0


def _divergent():
    g = np.zeros(rel_genome_length(SPEC))
    g[:6] = 1e-9                      # tau -> 0 (out of bounds)
    g[12:48] = 1e6                    # huge recurrent weights
    g[48:] = 5.0
    return g


def test_divergent_genome_guard_fires():
    # in-bounds (tau>=0.5) genomes can't diverge, so the guard is defensive
    g = _divergent()
    with pytest.raises(FloatingPointError):
        decode_agent(g, SPEC).run_trial(3.0, 4.0, offset2=0.0)
    assert relational_fitness(g) == 0.0


def _best_threshold_acc(xs):
    labels = [s2 > s1 for s1, s2, _ in SIZE_PAIRS]
    best = 0.0
    for t in sorted(set(xs)) + [max(xs) + 1]:
        pred = [x > t for x in xs]
        acc = np.mean([p == l for p, l in zip(pred, labels)])
        best = max(best, acc, 1 - acc)
    return best


def test_no_single_object_cue_separates_battery():
    assert _best_threshold_acc([s2 for _, s2, _ in SIZE_PAIRS]) <= 0.65
    assert _best_threshold_acc([s1 for s1, _, _ in SIZE_PAIRS]) <= 0.65


def test_battery_balanced_and_separated():
    sc = sum(s2 > s1 for s1, s2, _ in SIZE_PAIRS)
    sa = sum(s2 < s1 for s1, s2, _ in SIZE_PAIRS)
    assert sc == sa and 12 <= len(SIZE_PAIRS) <= 24
    assert all(abs(s2 - s1) >= 1.0 and 2 <= min(s1, s2) and max(s1, s2) <= 6
               for s1, s2, _ in SIZE_PAIRS)


def test_never_move_scores_chance():
    g = np.zeros(rel_genome_length(SPEC))
    g[:SPEC.n_neurons] = 1.0  # tau=1 for ALL neurons
    f = relational_fitness(g)
    # symmetric accuracy-aligned shaping -> a fixed (never-move) policy scores ~chance
    assert f == pytest.approx(0.552, abs=0.02), f  # fixed-policy chance floor
    assert f < 0.6, "a fixed policy must not beat chance"
