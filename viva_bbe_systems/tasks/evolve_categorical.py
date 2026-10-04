"""Seeded evolution driver for the categorical-perception agent."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..evolution import evolve_flat
from ..genome import _bounds_arrays
from ..bodies.categorical_genome import CatGenomeSpec, cat_genome_length, decode_agent
from .categorical_fitness import make_fitness

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "genomes" / "categorical_perception.npz"


def cat_bounds(spec: CatGenomeSpec):
    """Length-cat_genome_length lo/hi (tau>=0.5 is enforced via the CTRNN block)."""
    lo, hi = _bounds_arrays(spec.ctrnn_spec)
    n_sw = spec.n_neurons * spec.n_sensors
    lo = np.concatenate([lo, np.full(n_sw, spec.sensor_weight_range[0])])
    hi = np.concatenate([hi, np.full(n_sw, spec.sensor_weight_range[1])])
    return lo, hi


def evolve_categorical(*, pop_size=80, generations=100, seed=0, mutation_sd=0.8) -> dict:
    spec = CatGenomeSpec()
    lo, hi = cat_bounds(spec)
    return evolve_flat(make_fitness(spec), cat_genome_length(spec), lo, hi,
                       pop_size=pop_size, generations=generations,
                       mutation_sd=mutation_sd, seed=seed)


def save_seed(result, path) -> None:
    spec = CatGenomeSpec()
    np.savez(path, best_genome=np.asarray(result["best_genome"], dtype=float),
             best_fitness=float(result["best_fitness"]),
             n_neurons=spec.n_neurons, n_sensors=spec.n_sensors)


def load_seed(path) -> np.ndarray:
    with np.load(path) as z:
        return np.array(z["best_genome"], dtype=float)


def per_shape_final_distance(genome, spec=None, offsets=(-6.0, -3.0, 0.0, 3.0, 6.0),
                             dt=0.1, steps=200) -> dict:
    spec = spec or CatGenomeSpec()
    out = {}
    for shape in ("circle", "diamond"):
        ds = []
        for off in offsets:
            agent = decode_agent(genome, spec, dt=dt)
            ds.append(agent.run_trial(obj_offset=off, shape=shape, dt=dt,
                                      steps=steps)["final_distance"])
        out[shape] = float(np.mean(ds))
    return out


def main() -> None:
    res = evolve_categorical()
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_seed(res, DEFAULT_PATH)
    d = per_shape_final_distance(res["best_genome"])
    print(f"best_fitness={res['best_fitness']:.4f}")
    print(f"mean final distance: circle={d['circle']:.3f} diamond={d['diamond']:.3f}")
    print(f"wrote {DEFAULT_PATH}")


if __name__ == "__main__":
    main()
