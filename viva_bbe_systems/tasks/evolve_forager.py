"""Seeded evolution driver for the chemotactic forager (action switching)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..evolution import evolve_flat
from ..genome import GenomeSpec, genome_length, _bounds_arrays
from ..agents.forager_agent import MORPHOLOGY_SIZE
from .forager_fitness import make_forager_fitness

_DATA = Path(__file__).resolve().parent.parent / "data" / "genomes"
DEFAULT_PATH = _DATA / "action_switching_M2.npz"
CHECKPOINT_PATH = _DATA / "action_switching_M2_evolution.npz"


def forager_spec(morphology="M2") -> GenomeSpec:
    """Paper ranges: tau in [1,10] (tau>=1 keeps the CTRNN stable), theta/w in [-15,15]."""
    return GenomeSpec(size=MORPHOLOGY_SIZE[morphology], tau_range=(1.0, 10.0),
                      bias_range=(-15.0, 15.0), weight_range=(-15.0, 15.0))


def forager_bounds(morphology="M2"):
    return _bounds_arrays(forager_spec(morphology))


def evolve_forager(*, morphology="M2", pop_size=80, generations=120, seed=0,
                   mutation_sd=0.5, record_every=None, max_steps=None) -> dict:
    spec = forager_spec(morphology)
    lo, hi = forager_bounds(morphology)
    kw = {} if max_steps is None else {"max_steps": max_steps}
    return evolve_flat(make_forager_fitness(spec, morphology, **kw),
                       genome_length(spec), lo, hi, pop_size=pop_size,
                       generations=generations, mutation_sd=mutation_sd,
                       seed=seed, record_every=record_every)


def save_seed(result, path=DEFAULT_PATH, morphology="M2") -> None:
    np.savez(path, best_genome=np.asarray(result["best_genome"], dtype=float),
             best_fitness=float(result["best_fitness"]), morphology=morphology,
             size=MORPHOLOGY_SIZE[morphology])


def load_seed(path=DEFAULT_PATH) -> np.ndarray:
    with np.load(path) as z:
        return np.array(z["best_genome"], dtype=float)


def save_checkpoints(result, path=CHECKPOINT_PATH) -> None:
    cks = result["checkpoints"]
    np.savez(path,
             gens=np.array([c["gen"] for c in cks]),
             genomes=np.stack([c["genome"] for c in cks]),
             fitness=np.array([c["fitness"] for c in cks]),
             history=np.array(result["history"], dtype=float))


def load_checkpoints(path=CHECKPOINT_PATH) -> dict:
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in ("gens", "genomes", "fitness", "history")}


def per_config_report(genome, morphology="M2", max_steps=5000):
    """Per-config survival and whether both resources were visited (level rises)."""
    from ..agents.forager_agent import ForagerAgent
    from ..bodies.chemotactic_forager import ChemotacticForager
    from ..environments.chemotaxis_resources import ChemotaxisEnv, Resource
    from ..genome import decode
    from .forager_fitness import TRIAL_CONFIGS
    net_spec = forager_spec(morphology)
    out = []
    for c in TRIAL_CONFIGS:
        agent = ForagerAgent(decode(genome, net_spec), ChemotacticForager(morphology))
        env = ChemotaxisEnv(Resource(center=np.array(c["resource_a"], float), signal="A"),
                            Resource(center=np.array(c["resource_b"], float), signal="B"))
        r = agent.run_trial(env, c["init_levels"], start_pos=c["start_pos"],
                            start_angle=c["start_angle"], max_steps=max_steps)
        rises = np.diff(r["levels_hist"], axis=0) > 1e-9 if len(r["levels_hist"]) > 1 \
            else np.zeros((0, 2), bool)
        out.append({"survival": int(r["survival"]),
                    "visits_both": bool(rises.any(axis=0).all()) if len(rises) else False})
    return out


def main() -> None:
    res = evolve_forager()
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_seed(res, DEFAULT_PATH)
    rep = per_config_report(res["best_genome"])
    print(f"mean survival (fitness): {res['best_fitness']:.4f}")
    print("per-config survival:", [r["survival"] for r in rep])
    print("visits both resources:", [r["visits_both"] for r in rep])


if __name__ == "__main__":
    main()
