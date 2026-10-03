import numpy as np
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.analysis import nullcline_grid, bifurcation_sweep


def test_nullcline_grid_shapes():
    net = CTRNN(2)
    net.weights[:] = np.array([[4.5, 1.0], [-1.0, 4.5]])
    Y1, Y2, Z = nullcline_grid(net, np.zeros(2), neuron=0, y_range=(-5, 5), resolution=30)
    assert Y1.shape == (30, 30) and Z.shape == (30, 30)


def test_bifurcation_sweep_tracks_pitchfork():
    # single self-excitatory neuron: 1 equilibrium at low self-weight, 3 at high
    def factory(w_self):
        net = CTRNN(1)
        net.weights[:] = np.array([[w_self]])
        net.theta[:] = np.array([-w_self / 2.0])  # center-crossing
        return net

    values = [1.0, 8.0]  # given in order; low then high
    out = bifurcation_sweep(factory, values, I=0.0, rng=np.random.default_rng(0))
    assert [r["value"] for r in out] == values
    assert len(out[0]["equilibria"]) == 1
    assert len(out[1]["equilibria"]) == 3
