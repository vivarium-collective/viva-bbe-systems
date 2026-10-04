"""Seeded evolution driver for the legged-locomotion CPG walker."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..evolution import evolve_flat
from ..genome import GenomeSpec, genome_length, decode, _bounds_arrays
from ..agents.walker_agent import WALKER_SIZE, make_walker
from .walk_fitness import make_walk_fitness

_DATA = Path(__file__).resolve().parent.parent / "data" / "genomes"
DEFAULT_PATH = _DATA / "legged_walker.npz"
CHECKPOINT_PATH = _DATA / "legged_walker_evolution.npz"


def walker_spec() -> GenomeSpec:
    """tau >= 0.5 keeps the CTRNN from diverging; theta/w in [-16, 16]."""
    return GenomeSpec(size=WALKER_SIZE, tau_range=(0.5, 10.0),
                      bias_range=(-16.0, 16.0), weight_range=(-16.0, 16.0))


def walker_bounds():
    return _bounds_arrays(walker_spec())


def evolve_walker(*, pop_size=60, generations=80, seed=0, mutation_sd=0.5,
                  record_every=None, steps=500) -> dict:
    spec = walker_spec()
    lo, hi = walker_bounds()
    return evolve_flat(make_walk_fitness(spec, steps=steps), genome_length(spec),
                       lo, hi, pop_size=pop_size, generations=generations,
                       mutation_sd=mutation_sd, seed=seed, record_every=record_every)


def save_seed(result, path=DEFAULT_PATH) -> None:
    np.savez(path, best_genome=np.asarray(result["best_genome"], dtype=float),
             best_fitness=float(result["best_fitness"]), size=WALKER_SIZE)


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


def gait_metrics(foot_hist, vx_hist=None) -> dict:
    """duty_factor = fraction of steps foot down; stride_period = mean steps between
    successive foot-plant onsets (up->down), nan if < 2 onsets; mean_velocity of vx_hist."""
    foot = np.asarray(foot_hist, bool)
    onsets = np.flatnonzero(~foot[:-1] & foot[1:]) if len(foot) > 1 else np.array([])
    stride = float(np.mean(np.diff(onsets))) if len(onsets) >= 2 else float("nan")
    vel = float(np.mean(vx_hist)) if vx_hist is not None and len(vx_hist) else float("nan")
    return {"duty_factor": float(foot.mean()) if len(foot) else float("nan"),
            "stride_period": stride, "mean_velocity": vel}


def per_trial_report(genome, *, steps=500, dt=0.1) -> dict:
    agent = make_walker(decode(genome, walker_spec()))
    r = agent.run_trial(dt=dt, steps=steps)
    return {"distance": float(r["distance"]), **gait_metrics(r["foot_hist"], r["vx_hist"])}


def main() -> None:
    res = evolve_walker(record_every=5)
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_seed(res, DEFAULT_PATH)
    save_checkpoints(res, CHECKPOINT_PATH)
    rep = per_trial_report(res["best_genome"])
    print(f"distance (fitness): {res['best_fitness']:.4f}")
    print(f"stride period (steps): {rep['stride_period']:.2f}")
    print(f"duty factor: {rep['duty_factor']:.3f}")


if __name__ == "__main__":
    main()
