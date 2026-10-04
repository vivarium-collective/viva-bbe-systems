"""The committed seeded genome reproduces Beer-2003 catch/avoid behaviour.

This pins the active-categorical-perception investigation's acceptance band to
the committed seed: the agent catches circles (small final distance at every
offset) and avoids diamonds (large final distance), a clear category separation.
"""
import numpy as np
import pytest

from viva_bbe_systems.bodies.categorical_genome import CatGenomeSpec, decode_agent
from viva_bbe_systems.tasks.evolve_categorical import load_seed, DEFAULT_PATH

OFFSETS = (-6.0, -3.0, 0.0, 3.0, 6.0)


def _mean_final_distance(genome, spec, shape):
    ds = []
    for off in OFFSETS:
        agent = decode_agent(genome, spec, dt=0.1)
        ds.append(agent.run_trial(obj_offset=off, shape=shape, steps=200)["final_distance"])
    return float(np.mean(ds))


def test_seed_catches_circles_and_avoids_diamonds():
    assert DEFAULT_PATH.exists(), "committed seed genome missing"
    g = load_seed(DEFAULT_PATH)
    spec = CatGenomeSpec()
    circle = _mean_final_distance(g, spec, "circle")
    diamond = _mean_final_distance(g, spec, "diamond")
    # catches circles: within the catch radius on average
    assert circle < 1.5, f"circles not caught (mean fd {circle:.2f})"
    # avoids diamonds: far away, and a wide category separation
    assert diamond > 10.0, f"diamonds not avoided (mean fd {diamond:.2f})"
    assert diamond - circle > 8.0, f"weak category separation ({diamond - circle:.2f})"


def test_seed_catches_every_circle_offset():
    g = load_seed(DEFAULT_PATH)
    spec = CatGenomeSpec()
    for off in OFFSETS:
        agent = decode_agent(g, spec, dt=0.1)
        out = agent.run_trial(obj_offset=off, shape="circle", steps=200)
        assert out["caught"], f"circle at offset {off} not caught (fd {out['final_distance']:.2f})"
