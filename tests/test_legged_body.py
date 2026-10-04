from math import pi

import pytest

from viva_bbe_systems.bodies.legged import LeggedBody

LIM = pi / 6


def test_sense_normalizes_angle():
    b = LeggedBody()
    b.reset(angle=LIM)
    assert b.sense() == pytest.approx(1.0)
    b.reset(angle=-LIM)
    assert b.sense() == pytest.approx(-1.0)
    b.reset()
    assert b.sense() == pytest.approx(0.0)


def test_foot_down_backward_swing_moves_body_forward():
    # o_bs > o_fs -> +omega -> +d_theta; a planted foot must drive x forward.
    b = LeggedBody()
    b.act(1.0, 1.0, 0.0, 0.1)
    assert b.foot_down
    assert b.angle == pytest.approx(0.1)
    assert b.x > 0
    assert b.vx > 0
    assert b.x == pytest.approx(0.1 * b.leg_length)
    assert b.vx == pytest.approx(0.1 * b.leg_length / 0.1)


def test_foot_down_forward_swing_moves_body_backward():
    b = LeggedBody()
    b.act(1.0, 0.0, 1.0, 0.1)
    assert b.angle == pytest.approx(-0.1)
    assert b.x == pytest.approx(-0.1 * b.leg_length)


def test_foot_up_swings_leg_without_drive_and_coasts():
    b = LeggedBody()
    b.vx = 1.0
    b.act(0.0, 1.0, 0.0, 0.1)
    assert not b.foot_down
    assert b.angle == pytest.approx(0.1)
    assert b.vx == pytest.approx(0.9)
    assert b.x == pytest.approx(0.9 * 0.1)


def test_foot_up_from_rest_does_not_move_body():
    b = LeggedBody()
    for _ in range(10):
        b.act(0.0, 1.0, 0.0, 0.1)
    assert b.x == 0.0
    assert b.angle != 0.0


def test_angle_clamps_at_limit_foot_up():
    b = LeggedBody()
    for _ in range(100):
        b.act(0.0, 1.0, 0.0, 0.1)
    assert b.angle == pytest.approx(LIM)


def test_planted_at_limit_body_stops_advancing():
    b = LeggedBody()
    for _ in range(100):
        b.act(1.0, 1.0, 0.0, 0.1)
    assert b.angle == pytest.approx(LIM)
    x = b.x
    b.act(1.0, 1.0, 0.0, 0.1)
    assert b.x == pytest.approx(x)
    assert b.vx == pytest.approx(0.0)


def test_reset():
    b = LeggedBody()
    b.act(1.0, 1.0, 0.0, 0.1)
    b.reset()
    assert (b.angle, b.x, b.vx, b.omega) == (0.0, 0.0, 0.0, 0.0)
    assert not b.foot_down
