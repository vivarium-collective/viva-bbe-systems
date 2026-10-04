"""The committed walker seed reproduces a rhythmic CPG gait (Beer & Gallagher 1992).

Pins the legged-locomotion acceptance band: the evolved CTRNN central pattern
generator drives the leg in a REGULAR stance/swing rhythm that carries the body
steadily forward — not a one-off lunge. The seed was evolved with the opt-in
rhythm-shaped fitness (rewarding productive strides); pure net distance is the
faithful default.
"""
import numpy as np

from viva_bbe_systems.tasks.evolve_walker import (
    DEFAULT_PATH, CHECKPOINT_PATH, load_seed, load_checkpoints,
    walker_spec, per_trial_report, gait_metrics)
from viva_bbe_systems.agents.walker_agent import make_walker
from viva_bbe_systems.genome import decode


def _trial(record=False):
    g = load_seed()
    agent = make_walker(decode(g, walker_spec()))
    return g, agent.run_trial(steps=500, record=record)


def test_seed_walks_forward():
    assert DEFAULT_PATH.exists(), "committed walker seed missing"
    _, r = _trial()
    # covers substantial ground (achieved ~450; a lunge-and-coast manages ~20)
    assert r["distance"] > 200.0, f"only walked {r['distance']:.1f}"
    # and never reverses: the body advances monotonically (min x near the start)
    assert r["x_hist"].min() > -1.0


def test_seed_has_rhythmic_gait():
    _, r = _trial()
    foot = r["foot_hist"]
    onsets = np.flatnonzero(~foot[:-1] & foot[1:])   # up -> down plant events
    lifts = np.flatnonzero(foot[:-1] & ~foot[1:])     # down -> up lift events
    # a real gait cycles the foot many times, not a single plant
    assert len(onsets) >= 10 and len(lifts) >= 10, f"only {len(onsets)} plants"
    m = gait_metrics(foot, r["vx_hist"])
    # balanced stance/swing and a well-defined stride period (regular rhythm)
    assert 0.2 < m["duty_factor"] < 0.7, m["duty_factor"]
    assert np.isfinite(m["stride_period"]) and m["stride_period"] > 1.0
    # the rhythm is regular: stride periods have low spread
    periods = np.diff(onsets)
    assert periods.std() < 0.5 * periods.mean(), f"irregular: {periods}"


def test_seed_is_deterministic():
    g = load_seed()
    assert per_trial_report(g)["distance"] == per_trial_report(g)["distance"]


def test_seed_matches_final_checkpoint():
    # the evolution-progress viz and the committed seed must stay in sync:
    # the last checkpoint genome IS the committed best genome.
    assert np.array_equal(load_checkpoints(CHECKPOINT_PATH)["genomes"][-1], load_seed())
