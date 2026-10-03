import numpy as np
import pytest
from viva_bbe_systems.ctrnn import CTRNN, sigmoid, center_crossing_biases


def test_single_neuron_relaxes_to_input():
    # tau ẏ = -y + I  (no self-weight) => y* = I
    net = CTRNN(1, dt=0.01)
    net.weights[:] = 0.0
    net.theta[:] = 0.0
    net.reset(np.zeros(1))
    for _ in range(5000):
        net.step(external_input=np.array([5.0]))
    assert net.y[0] == pytest.approx(5.0, abs=1e-3)


def test_center_crossing_biases():
    w = np.array([[4.5, 1.0], [-1.0, 4.5]])
    theta = center_crossing_biases(w)
    assert theta == pytest.approx([-2.75, -1.75])


def test_two_neuron_oscillator_does_not_settle():
    net = CTRNN(2, dt=0.01)
    net.weights[:] = np.array([[4.5, 1.0], [-1.0, 4.5]])
    net.theta[:] = center_crossing_biases(net.weights)
    net.reset(np.array([0.1, -0.1]))
    traj = []
    for _ in range(20000):
        traj.append(net.step()[0])
    tail = np.array(traj[-5000:])
    assert tail.max() - tail.min() > 0.05  # sustained oscillation, not a fixed point


def test_non_finite_state_raises():
    net = CTRNN(1, dt=10.0)
    net.weights[:] = 0.0
    net.reset(np.array([0.0]))
    with pytest.raises(FloatingPointError):
        for _ in range(10000):
            net.step(external_input=np.array([1e6]))
