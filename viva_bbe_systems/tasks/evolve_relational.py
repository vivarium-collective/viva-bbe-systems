"""Seeded evolution driver for the relational-categorization agent
(catch obj2 iff larger than obj1; requires memory of obj1)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..evolution import evolve_flat
from ..bodies.relational_genome import (RelGenomeSpec, rel_genome_length, rel_bounds,
                                        decode_agent)
from .relational_fitness import make_relational_fitness, SIZE_PAIRS

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "genomes" / "relational.npz"
CHECKPOINT_PATH = DEFAULT_PATH.parent / "relational_evolution.npz"


def relational_spec() -> RelGenomeSpec:
    return RelGenomeSpec()


def relational_bounds():
    return rel_bounds(relational_spec())


def evolve_relational(*, pop_size=80, generations=120, seed=0, mutation_sd=0.5,
                      record_every=None) -> dict:
    spec = relational_spec()
    lo, hi = relational_bounds()
    return evolve_flat(make_relational_fitness(spec), rel_genome_length(spec), lo, hi,
                       pop_size=pop_size, generations=generations,
                       mutation_sd=mutation_sd, seed=seed, record_every=record_every)


def save_checkpoints(result, path=CHECKPOINT_PATH) -> None:
    cks = result["checkpoints"]
    np.savez(path,
             gens=np.array([c["gen"] for c in cks]),
             genomes=np.stack([c["genome"] for c in cks]),
             fitness=np.array([c["fitness"] for c in cks]),
             history=np.array(result["history"], dtype=float))


def load_checkpoints(path=CHECKPOINT_PATH) -> dict:
    with np.load(path) as z:
        return {"gens": np.array(z["gens"]), "genomes": np.array(z["genomes"]),
                "fitness": np.array(z["fitness"]), "history": np.array(z["history"])}


def save_seed(result, path=DEFAULT_PATH) -> None:
    spec = relational_spec()
    np.savez(path, best_genome=np.asarray(result["best_genome"], dtype=float),
             best_fitness=float(result["best_fitness"]),
             n_neurons=spec.n_neurons, n_sensors=spec.n_sensors)


def load_seed(path=DEFAULT_PATH) -> np.ndarray:
    with np.load(path) as z:
        return np.array(z["best_genome"], dtype=float)


def accuracy_report(genome, spec=None, dt=0.1) -> dict:
    spec = spec or relational_spec()
    c_ok, a_ok = [], []
    for s1, s2, off in SIZE_PAIRS:
        agent = decode_agent(genome, spec, dt=dt)
        out = agent.run_trial(s1, s2, offset2=off, dt=dt)
        (c_ok if out["should_catch"] else a_ok).append(bool(out["correct"]))
    n = len(c_ok) + len(a_ok)
    frac = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {"accuracy": (sum(c_ok) + sum(a_ok)) / n if n else 0.0,
            "catch_accuracy": frac(c_ok), "avoid_accuracy": frac(a_ok), "n": n}


def always_catch_accuracy() -> float:
    """Baseline: a policy that always catches is right on the should-catch (s2>s1) trials."""
    return float(np.mean([s2 > s1 for s1, s2, _ in SIZE_PAIRS]))


def always_avoid_accuracy() -> float:
    return float(np.mean([s2 < s1 for s1, s2, _ in SIZE_PAIRS]))


def main() -> None:
    res = evolve_relational()
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_seed(res, DEFAULT_PATH)
    save_checkpoints(res, CHECKPOINT_PATH)
    r = accuracy_report(res["best_genome"])
    print(f"best_fitness={res['best_fitness']:.4f}")
    print(f"accuracy={r['accuracy']:.3f} catch={r['catch_accuracy']:.3f} "
          f"avoid={r['avoid_accuracy']:.3f} (n={r['n']})")
    print(f"baselines: always-catch={always_catch_accuracy():.3f} "
          f"always-avoid={always_avoid_accuracy():.3f}")
    print(f"wrote {DEFAULT_PATH}")


if __name__ == "__main__":
    main()
