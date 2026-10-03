import numpy as np
from viva_bbe_systems.param_space import single_neuron_net, codim2_equilibria_count


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
