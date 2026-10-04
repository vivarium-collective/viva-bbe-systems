"""Engine-run smoke tests for the three BBE composites (walker, forager, ctrnn).

Mirrors tests/test_categorical_composite.py: each composite must be discovered
by build_core(), its @composite_generator must register under the clean dotted
id the study baselines point at, and it must BUILD and RUN through the
process-bigraph engine (Composite({"state": spec}, core=build_core()); sim.run)
with a sensible readout.
"""
import numpy as np
import pytest
from process_bigraph import Composite

from viva_bbe_systems.core import build_core
from viva_bbe_systems.processes.ctrnn_process import build_ctrnn_composite
from viva_bbe_systems.processes.forager_body_process import build_forager_composite
from viva_bbe_systems.processes.walker_body_process import build_walker_composite


def test_processes_discovered():
    core = build_core()
    assert "WalkerBodyProcess" in core.link_registry
    assert "ForagerBodyProcess" in core.link_registry
    assert "CTRNNProcess" in core.link_registry


def test_generators_registered():
    import viva_bbe_systems.composites  # noqa: F401 -- fires the decorators
    from process_bigraph.composite_generator import _REGISTRY

    for name in ("walker", "forager", "ctrnn_parameter_space"):
        assert f"viva_bbe_systems.composites.{name}" in _REGISTRY


def test_generators_resolve_to_runnable_specs():
    """The repointed study baselines address these dotted ids; each must resolve
    to a spec that builds against the core (guards the 'composite not found in
    registry' regression)."""
    from process_bigraph.composite_generator import _REGISTRY, build_generator

    for name in ("walker", "forager", "ctrnn_parameter_space"):
        gid = f"viva_bbe_systems.composites.{name}"
        assert gid in _REGISTRY
        spec = build_generator(_REGISTRY[gid])   # resolve by registered entry
        assert isinstance(spec, dict) and "brain" in spec
        Composite({"state": spec}, core=build_core())  # builds, no error


def test_walker_walks_through_engine():
    sim = Composite({"state": build_walker_composite()}, core=build_core())
    xs = []
    for _ in range(80):
        sim.run(0.1)
        xs.append(float(sim.state["body_x"]))
    assert np.all(np.isfinite(xs))
    assert xs[-1] > xs[0]            # moved forward
    assert xs[-1] > 50.0, xs[-1]     # the committed seed walks


def test_forager_runs_and_moves_through_engine():
    sim = Composite({"state": build_forager_composite()}, core=build_core())
    start = np.asarray(sim.state["agent_pos"], float).copy()
    for _ in range(60):              # 60 steps, no error
        sim.run(0.1)
    nutrients = np.asarray(sim.state["nutrient_levels"], float)
    final = np.asarray(sim.state["agent_pos"], float)
    assert nutrients.shape == (2,)   # length-2 nutrient vector
    assert np.all(np.isfinite(nutrients))
    assert not np.allclose(start, final), (start, final)   # the agent moved


@pytest.mark.parametrize("size", [3, 5])
def test_ctrnn_outputs_finite_through_engine(size):
    sim = Composite({"state": build_ctrnn_composite(size=size)}, core=build_core())
    for _ in range(30):
        sim.run(0.1)
    outs = np.asarray(sim.state["neuron_outputs"], float)
    assert outs.shape == (size,)
    assert np.all(np.isfinite(outs))
