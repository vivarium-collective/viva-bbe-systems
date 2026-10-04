"""Extended genome for the categorical-perception agent: the shared CTRNN genome
plus a (n_neurons x n_sensors) sensor-input weight block. Demonstrates the spec's
"composite declares its evolvable genome" contract for an embodied BBE model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..genome import GenomeSpec, genome_length, decode, random_genome
from ..bodies.categorical_perception import CategoricalBody
from ..agents.categorical_agent import CategoricalAgent


@dataclass(frozen=True)
class CatGenomeSpec:
    n_neurons: int = 5
    n_sensors: int = 7
    sensor_weight_range: tuple = (-5.0, 5.0)
    motor_gain: float = 5.0

    @property
    def ctrnn_spec(self) -> GenomeSpec:
        return GenomeSpec(size=self.n_neurons)


def cat_genome_length(spec: CatGenomeSpec) -> int:
    return genome_length(spec.ctrnn_spec) + spec.n_neurons * spec.n_sensors


def decode_agent(genome, spec: CatGenomeSpec, dt=0.1) -> CategoricalAgent:
    genome = np.asarray(genome, dtype=float)
    if genome.size != cat_genome_length(spec):
        raise ValueError(
            f"genome length {genome.size} != expected {cat_genome_length(spec)}")
    nctr = genome_length(spec.ctrnn_spec)
    net = decode(genome[:nctr], spec.ctrnn_spec)
    net.dt = dt
    sw = genome[nctr:].reshape(spec.n_neurons, spec.n_sensors)
    body = CategoricalBody(n_sensors=spec.n_sensors)
    return CategoricalAgent(net, body, sw,
                            motor_indices=(spec.n_neurons - 2, spec.n_neurons - 1),
                            motor_gain=spec.motor_gain)


def random_cat_genome(spec: CatGenomeSpec, rng) -> np.ndarray:
    ctr = random_genome(spec.ctrnn_spec, rng)
    sw = rng.uniform(*spec.sensor_weight_range,
                     size=spec.n_neurons * spec.n_sensors)
    return np.concatenate([ctr, sw])
