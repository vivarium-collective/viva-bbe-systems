import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from viva_bbe_systems.ctrnn import CTRNN, center_crossing_biases
from viva_bbe_systems.viz import phase_portrait_2d, bifurcation_diagram


def test_phase_portrait_returns_figure():
    net = CTRNN(2)
    net.weights[:] = np.array([[4.5, 1.0], [-1.0, 4.5]])
    net.theta[:] = center_crossing_biases(net.weights)
    fig = phase_portrait_2d(net, np.zeros(2), resolution=15)
    ax = fig.axes[0]
    # quiver (1) + two nullcline contour sets, each a collection
    assert len(ax.collections) >= 3
    assert any(type(c).__name__ == "Quiver" for c in ax.collections)
    # at least one equilibrium marker drawn
    assert any(len(l.get_xdata()) == 1 for l in ax.lines)
    plt.close(fig)


def test_bifurcation_diagram_returns_figure():
    def factory(w):
        net = CTRNN(1); net.weights[:] = [[w]]; net.theta[:] = [-w / 2]
        return net
    fig = bifurcation_diagram(factory, np.linspace(0.5, 8.0, 12), "self-weight")
    ax = fig.axes[0]
    assert len(ax.lines) >= 12  # at least one equilibrium per parameter value
    assert ax.get_xlabel() == "self-weight"
    plt.close(fig)
