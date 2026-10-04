import numpy as np
from viva_bbe_systems.tasks.evolve_categorical import evolve_categorical, save_seed, load_seed


def test_evolution_smoke_improves(tmp_path):
    # tiny budget: just assert the GA runs end-to-end and improves on gen 0
    res = evolve_categorical(pop_size=8, generations=3, seed=0)
    assert res["best_fitness"] >= res["history"][0]
    p = tmp_path / "seed.npz"
    save_seed(res, p)
    g = load_seed(p)
    assert g.ndim == 1
    assert np.allclose(g, res["best_genome"])
