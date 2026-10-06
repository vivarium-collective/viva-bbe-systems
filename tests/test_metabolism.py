import numpy as np
import pytest

from viva_bbe_systems.metabolism import Metabolism

EAT, DRAIN = 0.07, 0.0035  # sped-up spread-forager physics


def test_inside_a():
    m = Metabolism(levels=(5.0, 5.0))
    m.step(True, False)
    assert m.levels == pytest.approx([5.0 + EAT - DRAIN, 5.0 - DRAIN])
    assert m.levels.shape == (2,)


def test_inside_b():
    m = Metabolism(levels=(5.0, 5.0))
    m.step(False, True)
    assert m.levels == pytest.approx([5.0 - DRAIN, 5.0 + EAT - DRAIN])


def test_outside_both_drains():
    m = Metabolism(levels=(5.0, 3.0))
    m.step(False, False)
    assert m.levels == pytest.approx([5.0 - DRAIN, 3.0 - DRAIN])


def test_death_at_zero():
    m = Metabolism(levels=(0.003, 5.0))  # < DRAIN -> dies in one step
    assert m.alive
    m.step(False, False)
    assert m.levels[0] == 0.0
    assert not m.alive


def test_death_exactly_zero():
    m = Metabolism(levels=(DRAIN, 5.0))
    m.step(False, False)
    assert not m.alive


def test_cap():
    m = Metabolism(levels=(9.99, 9.99))
    m.step(True, True)
    assert np.all(m.levels == 10.0)
