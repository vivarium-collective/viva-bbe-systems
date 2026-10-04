import numpy as np
import pytest

from viva_bbe_systems.environments.chemotaxis_resources import (
    ChemotaxisEnv,
    Resource,
    concentration,
)


@pytest.fixture
def env():
    a = Resource(center=np.array([20.0, 50.0]), radius=7.0, signal="A")
    b = Resource(center=np.array([80.0, 50.0]), radius=7.0, signal="B")
    return ChemotaxisEnv(a, b)


def test_concentration_flat_inside_radius():
    r = Resource(center=np.array([0.0, 0.0]), radius=7.0, signal="A")
    assert concentration(r, np.array([0.0, 0.0])) == 10.0
    assert concentration(r, np.array([3.0, 4.0])) == 10.0
    assert concentration(r, np.array([7.0, 0.0])) == 10.0


def test_concentration_decays_outside():
    r = Resource(center=np.array([0.0, 0.0]), radius=7.0, signal="A")
    assert concentration(r, np.array([20.0, 0.0])) == pytest.approx(10 * np.exp(-1), rel=1e-9)
    assert concentration(r, np.array([20.0, 0.0])) == pytest.approx(3.6788, abs=1e-3)


def test_conc_at_selects_signal(env):
    p = np.array([22.0, 50.0])
    assert env.conc_at(p, "A") == 10.0
    assert env.conc_at(p, "B") == pytest.approx(10 * np.exp(-0.05 * 58.0))
    assert env.conc_at(p, "A") != env.conc_at(p, "B")
    with pytest.raises(ValueError):
        env.conc_at(p, "C")


def test_inside(env):
    assert env.inside(np.array([25.0, 50.0]), "A")
    assert not env.inside(np.array([25.0, 50.0]), "B")
    assert not env.inside(np.array([40.0, 50.0]), "A")


def test_clamp(env):
    out = env.clamp(np.array([-5.0, 120.0]))
    assert np.array_equal(out, np.array([0.0, 100.0]))
    assert np.array_equal(env.clamp(np.array([50.0, 50.0])), np.array([50.0, 50.0]))
