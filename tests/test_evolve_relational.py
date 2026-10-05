import numpy as np
from viva_bbe_systems.tasks import evolve_relational as er
from viva_bbe_systems.bodies.relational_genome import rel_genome_length


def test_smoke_evolve_and_bounds_and_io(tmp_path):
    res = er.evolve_relational(pop_size=8, generations=3, seed=0, record_every=1)
    assert np.isfinite(res["best_fitness"])
    assert res["best_fitness"] >= res["history"][0] - 1e-12
    spec = er.relational_spec()
    lo, hi = er.relational_bounds()
    L = rel_genome_length(spec)
    assert len(lo) == len(hi) == L
    assert np.all(lo[:spec.n_neurons] == 0.5)

    p = tmp_path / "s.npz"
    er.save_seed(res, p)
    assert np.array_equal(er.load_seed(p), res["best_genome"])
    c = tmp_path / "c.npz"
    er.save_checkpoints(res, c)
    ck = er.load_checkpoints(c)
    assert np.array_equal(ck["genomes"][-1], res["checkpoints"][-1]["genome"])
    assert np.array_equal(ck["history"], np.array(res["history"], dtype=float))

    rep = er.accuracy_report(res["best_genome"])
    assert set(rep) == {"accuracy", "catch_accuracy", "avoid_accuracy", "n"}
    assert 0.0 <= rep["accuracy"] <= 1.0


def test_baselines():
    for f in (er.always_catch_accuracy, er.always_avoid_accuracy):
        assert 0.0 <= f() <= 1.0
