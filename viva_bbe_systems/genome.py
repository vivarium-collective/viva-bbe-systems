"""Flat-vector genome encode/decode for a CTRNN (the GA's evolvable contract)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ctrnn import CTRNN


@dataclass(frozen=True)
class GenomeSpec:
    size: int
    tau_range: tuple[float, float] = (0.5, 10.0)
    bias_range: tuple[float, float] = (-16.0, 16.0)
    weight_range: tuple[float, float] = (-16.0, 16.0)


def genome_length(spec: GenomeSpec) -> int:
    n = spec.size
    return n + n + n * n  # tau, theta, weights


def encode(net: CTRNN, spec: GenomeSpec) -> np.ndarray:
    return np.concatenate([net.tau, net.theta, net.weights.reshape(-1)])


def decode(genome: np.ndarray, spec: GenomeSpec) -> CTRNN:
    genome = np.asarray(genome, dtype=float)
    if genome.size != genome_length(spec):
        raise ValueError(
            f"genome length {genome.size} != expected {genome_length(spec)} for size {spec.size}"
        )
    n = spec.size
    net = CTRNN(n)
    net.tau[:] = genome[:n]
    net.theta[:] = genome[n:2 * n]
    net.weights[:] = genome[2 * n:].reshape(n, n)
    return net


def _bounds_arrays(spec: GenomeSpec):
    n = spec.size
    lo = np.concatenate([
        np.full(n, spec.tau_range[0]),
        np.full(n, spec.bias_range[0]),
        np.full(n * n, spec.weight_range[0]),
    ])
    hi = np.concatenate([
        np.full(n, spec.tau_range[1]),
        np.full(n, spec.bias_range[1]),
        np.full(n * n, spec.weight_range[1]),
    ])
    return lo, hi


def random_genome(spec: GenomeSpec, rng: np.random.Generator) -> np.ndarray:
    lo, hi = _bounds_arrays(spec)
    return rng.uniform(lo, hi)


def clip_to_bounds(genome: np.ndarray, spec: GenomeSpec) -> np.ndarray:
    lo, hi = _bounds_arrays(spec)
    return np.clip(np.asarray(genome, dtype=float), lo, hi)
