import numpy as np

from viva_bbe_systems.agents.forager_agent import MORPHOLOGY_SIZE
from viva_bbe_systems.genome import GenomeSpec, genome_length, random_genome
from viva_bbe_systems.tasks.forager_fitness import (
    TRIAL_CONFIGS, longevity_fitness, make_forager_fitness)

SPEC = GenomeSpec(size=MORPHOLOGY_SIZE["M2"])


def _genome(tau=1.0):
    n = SPEC.size
    g = np.zeros(genome_length(SPEC))
    g[:n] = tau
    return g


def test_eleven_configs():
    assert len(TRIAL_CONFIGS) == 11
    for c in TRIAL_CONFIGS:
        assert set(c) == {"resource_a", "resource_b", "start_pos", "start_angle", "init_levels"}


def test_finite_bounded_deterministic():
    g = random_genome(SPEC, np.random.default_rng(0))
    f1 = longevity_fitness(g, SPEC, max_steps=300)
    f2 = make_forager_fitness(SPEC, max_steps=300)(g)
    assert np.isfinite(f1) and 0.0 <= f1 <= 1.0
    assert f1 == f2


def test_divergent_genome_scores_low_no_raise():
    g = _genome(tau=1e-12)
    g[2 * SPEC.size:] = 16.0
    f = longevity_fitness(g, SPEC, max_steps=300)
    assert np.isfinite(f) and 0.0 <= f < 1.0


def test_non_mover_bounded_by_starvation():
    f = longevity_fitness(_genome(), SPEC, max_steps=5000)
    bound = np.mean([min(c["init_levels"]) / 0.0045 for c in TRIAL_CONFIGS]) / 5000
    assert 0.0 < f < 1.0
    assert f <= bound * 1.5
