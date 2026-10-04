import numpy as np
import pytest
from viva_bbe_systems.bodies.categorical_genome import (
    CatGenomeSpec, cat_genome_length, decode_agent, random_cat_genome)
from viva_bbe_systems.agents.categorical_agent import CategoricalAgent


def test_length():
    spec = CatGenomeSpec(n_neurons=5, n_sensors=7)
    assert cat_genome_length(spec) == (5 + 5 + 25) + 5 * 7


def test_decode_builds_agent():
    spec = CatGenomeSpec()
    g = random_cat_genome(spec, np.random.default_rng(0))
    agent = decode_agent(g, spec)
    assert isinstance(agent, CategoricalAgent)
    assert agent.sensor_weights.shape == (spec.n_neurons, spec.n_sensors)


def test_decode_rejects_wrong_length():
    spec = CatGenomeSpec()
    with pytest.raises(ValueError, match="length"):
        decode_agent(np.zeros(3), spec)
