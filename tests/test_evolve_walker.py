import numpy as np

from viva_bbe_systems.genome import genome_length
from viva_bbe_systems.agents.walker_agent import WALKER_SIZE
from viva_bbe_systems.tasks import evolve_walker as ew


def test_bounds():
    lo, hi = ew.walker_bounds()
    n = genome_length(ew.walker_spec())
    assert len(lo) == len(hi) == n
    assert np.all(lo[:WALKER_SIZE] == 0.5)
    assert np.all(hi >= lo)


def test_evolve_smoke_and_roundtrip(tmp_path):
    res = ew.evolve_walker(pop_size=8, generations=3, steps=60, seed=0, record_every=1)
    assert np.isfinite(res["best_fitness"])
    assert res["best_fitness"] >= res["history"][0]
    p = tmp_path / "seed.npz"
    ew.save_seed(res, p)
    assert np.array_equal(ew.load_seed(p), res["best_genome"])
    c = tmp_path / "ck.npz"
    ew.save_checkpoints(res, c)
    ck = ew.load_checkpoints(c)
    assert np.array_equal(ck["history"], np.array(res["history"]))
    assert ck["genomes"].shape[0] == len(res["checkpoints"])
    rep = ew.per_trial_report(res["best_genome"], steps=60)
    assert {"distance", "duty_factor", "stride_period", "mean_velocity"} <= set(rep)


def test_gait_metrics_synthetic():
    foot = np.array([0, 0, 1, 1] * 6, bool)
    m = ew.gait_metrics(foot, np.full(24, 0.2))
    assert m["duty_factor"] == 0.5
    assert m["stride_period"] == 4
    assert abs(m["mean_velocity"] - 0.2) < 1e-12
