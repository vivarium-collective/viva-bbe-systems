import numpy as np
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.analysis import equilibria, jacobian, eigenvalues, is_stable


def _net1(I_self=0.0):
    net = CTRNN(1)
    net.weights[:] = 0.0
    net.theta[:] = 0.0
    return net


def test_single_neuron_unique_equilibrium():
    net = _net1()
    eqs = equilibria(net, np.array([3.0]), rng=np.random.default_rng(0))
    assert len(eqs) == 1
    np.testing.assert_allclose(eqs[0][0], 3.0, atol=1e-4)


def test_jacobian_matches_finite_difference():
    net = CTRNN(2)
    rng = np.random.default_rng(2)
    net.weights[:] = rng.uniform(-3, 3, (2, 2))
    net.theta[:] = rng.uniform(-1, 1, 2)
    y = rng.uniform(-1, 1, 2)
    J = jacobian(net, y)
    eps = 1e-6
    Jfd = np.zeros((2, 2))
    for j in range(2):
        dy = np.zeros(2); dy[j] = eps
        Jfd[:, j] = (net.derivatives(y + dy, np.zeros(2)) - net.derivatives(y - dy, np.zeros(2))) / (2 * eps)
    np.testing.assert_allclose(J, Jfd, atol=1e-5)


def test_stable_fixed_point_detected():
    net = _net1()
    assert is_stable(net, np.array([3.0])) is True


def test_bistable_has_three_equilibria():
    # strong self-excitation -> bistable (two stable, one unstable)
    net = CTRNN(1)
    net.weights[:] = np.array([[8.0]])
    net.theta[:] = np.array([-4.0])  # center-crossing for a single self-excitatory neuron
    eqs = equilibria(net, np.array([0.0]), n_starts=200, rng=np.random.default_rng(3))
    assert len(eqs) == 3
