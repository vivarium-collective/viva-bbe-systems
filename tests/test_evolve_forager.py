import numpy as np
from viva_bbe_systems.tasks.evolve_forager import (
    evolve_forager, forager_bounds, save_seed, load_seed,
    save_checkpoints, load_checkpoints)
from viva_bbe_systems.genome import GenomeSpec, genome_length
from viva_bbe_systems.agents.forager_agent import MORPHOLOGY_SIZE


def test_bounds():
    n = MORPHOLOGY_SIZE["M2"]
    lo, hi = forager_bounds("M2")
    L = genome_length(GenomeSpec(n))
    assert len(lo) == len(hi) == L
    assert lo[:n].min() == 1.0 and hi[:n].max() == 10.0
    assert lo[n:].min() == -15.0 and hi[n:].max() == 15.0


def test_smoke_and_roundtrip(tmp_path):
    res = evolve_forager(pop_size=6, generations=2, seed=0, max_steps=50, record_every=1)
    assert res["best_fitness"] >= res["history"][0]
    p = tmp_path / "s.npz"
    save_seed(res, p)
    g = load_seed(p)
    assert g.ndim == 1 and np.allclose(g, res["best_genome"])
    save_checkpoints(res, tmp_path / "c.npz")
    assert "history" in load_checkpoints(tmp_path / "c.npz")
