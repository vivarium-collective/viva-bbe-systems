import numpy as np
from viva_bbe_systems.analysis import equilibria, is_stable
from viva_bbe_systems.param_space import single_neuron_net, two_neuron_net, codim2_equilibria_count


def test_codim2_count_grid_shape_and_bistable_region():
    # vary self-weight and bias of a single neuron; high self-weight near
    # center-crossing -> 3 equilibria (bistable) region exists
    w_vals = np.linspace(0.5, 10.0, 8)
    b_vals = np.linspace(-6.0, 6.0, 8)

    def make(w, b):
        net = single_neuron_net(w)
        net.theta[:] = [b]
        return net

    grid = codim2_equilibria_count(w_vals, b_vals, make)
    assert grid.shape == (8, 8)
    assert grid.max() == 3  # a bistable region is found


def test_two_neuron_oscillator_has_single_unstable_equilibrium():
    net = two_neuron_net(np.array([[4.5, 1.0], [-1.0, 4.5]]))
    eqs = equilibria(net, np.zeros(2), rng=np.random.default_rng(0))
    assert len(eqs) == 1
    assert not is_stable(net, eqs[0])
