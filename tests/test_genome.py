import numpy as np
import pytest
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.genome import (
    GenomeSpec, genome_length, encode, decode, random_genome, clip_to_bounds,
)


def test_roundtrip_preserves_params():
    spec = GenomeSpec(size=3)
    net = CTRNN(3)
    rng = np.random.default_rng(0)
    net.tau[:] = rng.uniform(0.5, 10.0, 3)
    net.theta[:] = rng.uniform(-5, 5, 3)
    net.weights[:] = rng.uniform(-5, 5, (3, 3))
    g = encode(net, spec)
    net2 = decode(g, spec)
    assert net2.tau == pytest.approx(net.tau)
    assert net2.theta == pytest.approx(net.theta)
    assert net2.weights == pytest.approx(net.weights)


def test_length():
    assert genome_length(GenomeSpec(size=4)) == 4 + 4 + 16


def test_decode_rejects_wrong_length():
    spec = GenomeSpec(size=3)
    with pytest.raises(ValueError, match="length"):
        decode(np.zeros(5), spec)


def test_clip_enforces_bounds():
    spec = GenomeSpec(size=2, tau_range=(1.0, 2.0))
    g = random_genome(spec, np.random.default_rng(1))
    g[0] = 999.0  # first tau out of range
    clipped = clip_to_bounds(g, spec)
    assert 1.0 <= clipped[0] <= 2.0
