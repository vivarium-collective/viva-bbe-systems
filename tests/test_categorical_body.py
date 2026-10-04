import numpy as np
import pytest
from viva_bbe_systems.bodies.categorical_perception import CategoricalBody
from viva_bbe_systems.environments.falling_objects import FallingObject


def test_centered_object_is_symmetric():
    body = CategoricalBody(n_sensors=7)
    obj = FallingObject(center=np.array([0.0, 10.0]), vy=1.0, size=1.0, shape="circle")
    s = body.sense(obj)
    assert s.shape == (7,)
    assert s[0] == pytest.approx(s[-1], abs=1e-6)        # symmetric
    assert s.argmax() == 3                                # center sensor strongest


def test_offset_object_is_asymmetric():
    body = CategoricalBody(n_sensors=7)
    obj = FallingObject(center=np.array([3.0, 10.0]), vy=1.0, size=1.0, shape="circle")
    s = body.sense(obj)
    assert s.argmax() > 3                                 # peak shifts toward the object


def test_far_object_all_zero():
    body = CategoricalBody(n_sensors=7)
    obj = FallingObject(center=np.array([100.0, 10.0]), vy=1.0, size=1.0, shape="circle")
    assert np.all(body.sense(obj) == 0.0)


def test_effector_moves_toward_more_active_motor():
    body = CategoricalBody()
    body.act(np.array([1.0, 0.0]), dt=1.0, gain=1.0)
    assert body.x > 0
