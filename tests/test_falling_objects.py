import numpy as np
import pytest
from viva_bbe_systems.environments.falling_objects import ray_distance, FallingObject


def test_ray_straight_up_hits_circle_center():
    # object centered directly above origin at height 10, radius 1 -> near face at 9
    d = ray_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                     np.array([0.0, 10.0]), 1.0, "circle")
    assert d == pytest.approx(9.0, abs=1e-6)


def test_ray_misses_returns_none():
    d = ray_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                     np.array([5.0, 10.0]), 1.0, "circle")
    assert d is None


def test_ray_hits_diamond_point():
    # diamond (L1 ball) size 1 centered above at 10 -> bottom vertex at y=9
    d = ray_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                     np.array([0.0, 10.0]), 1.0, "diamond")
    assert d == pytest.approx(9.0, abs=1e-6)


def test_falling_object_descends():
    obj = FallingObject(center=np.array([0.0, 10.0]), vy=2.0, size=1.0, shape="circle")
    obj.step(0.5)
    assert obj.center[1] == pytest.approx(9.0)
