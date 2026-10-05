import numpy as np
import pytest

from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.environments.chemotaxis_resources import ChemotaxisEnv, Resource
from viva_bbe_systems.metabolism import Metabolism

PI = np.pi


def make_env(a=(50.0, 70.0), b=(10.0, 10.0)):
    return ChemotaxisEnv(
        Resource(np.array(a), 7.0, "A"), Resource(np.array(b), 7.0, "B"), 100.0, 100.0
    )


@pytest.mark.parametrize("m,chemo,nut", [("M1", 4, 2), ("M2", 2, 2), ("M3", 4, 0)])
def test_sensor_counts(m, chemo, nut):
    f = ChemotacticForager(m)
    assert (f.n_chemo, f.n_nutrient, f.n_sensors) == (chemo, nut, chemo + nut)
    assert len(f.sensor_layout()) == chemo
    out = f.sense(make_env(), Metabolism())
    assert out.shape == (chemo + nut,)


def test_layouts():
    assert sorted(ChemotacticForager("M2").sensor_layout()) == [(0.0, "A"), (0.0, "B")]
    lay = ChemotacticForager("M1").sensor_layout()
    assert sorted(lay) == sorted(
        [(PI / 2, "A"), (PI / 2, "B"), (-PI / 2, "A"), (-PI / 2, "B")]
    )
    assert ChemotacticForager("M3").sensor_layout() == lay


def test_nutrients_appended_after_chemo():
    f = ChemotacticForager("M2")
    out = f.sense(make_env(), Metabolism(levels=(3.0, 4.0)))
    np.testing.assert_allclose(out[-2:], [3.0, 4.0])
    out3 = ChemotacticForager("M3").sense(make_env(), Metabolism(levels=(3.0, 4.0)))
    assert len(out3) == 4  # chemo only, no nutrient sensors


def test_stalk_toward_resource_reads_higher():
    env = make_env(a=(50.0, 70.0))  # directly +y of the body at (50, 50)
    f = ChemotacticForager("M1", )
    f.pos = np.array([50.0, 50.0])
    f.angle = 0.0  # +x heading; +pi/2 stalk points +y toward A
    out = f.sense(env, Metabolism())
    lay = f.sensor_layout()
    a_left = out[lay.index((PI / 2, "A"))]
    a_right = out[lay.index((-PI / 2, "A"))]
    assert a_left > a_right
    np.testing.assert_allclose(a_left, env.conc_at([50.0, 56.0], "A"))


def test_turn_direction_and_straight():
    env = make_env()
    f = ChemotacticForager("M2")
    f.pos = np.array([50.0, 50.0])
    f.act(1.0, 0.0, env)
    assert f.angle == pytest.approx(PI / 12)
    assert f.velocity == pytest.approx(0.025)
    g = ChemotacticForager("M2")
    g.pos = np.array([50.0, 50.0])
    g.act(0.5, 0.5, env)
    assert g.angle == 0.0
    assert g.velocity == pytest.approx(0.025)
    assert g.pos[0] > 50.0 and g.pos[1] == pytest.approx(50.0)


def test_friction_decay():
    env = make_env()
    f = ChemotacticForager("M2")
    f.pos = np.array([50.0, 50.0])
    f.velocity = 1.0
    f.act(0.0, 0.0, env)
    assert f.velocity == pytest.approx(0.9)
    f.act(0.0, 0.0, env)
    assert f.velocity == pytest.approx(0.81)


def test_velocity_integration():
    f = ChemotacticForager("M2")
    f.pos = np.array([50.0, 50.0])
    f.velocity = 1.0
    f.act(1.0, 1.0, make_env())
    assert f.velocity == pytest.approx(0.9 + 2 * 0.025)


def test_clamps_at_wall():
    env = make_env()
    f = ChemotacticForager("M2")
    f.pos = np.array([99.9, 50.0])
    f.velocity = 5.0
    for _ in range(5):
        f.act(1.0, 1.0, env)
    assert 0.0 <= f.pos[0] <= 100.0 and 0.0 <= f.pos[1] <= 100.0


def test_inside():
    env = make_env(a=(50.0, 50.0), b=(10.0, 10.0))
    f = ChemotacticForager("M2")
    f.pos = np.array([52.0, 50.0])
    assert f.inside(env) == (True, False)
