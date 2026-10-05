"""Extended genome for the relational-categorization agent: the shared CTRNN
genome (6 neurons, tau>=0.5) plus a (n_neurons x n_sensors) sensor-weight block."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..genome import GenomeSpec, genome_length, decode, random_genome, _bounds_arrays
from ..bodies.categorical_perception import CategoricalBody
from ..agents.relational_agent import RelationalAgent


@dataclass(frozen=True)
class RelGenomeSpec:
    n_neurons: int = 6
    n_sensors: int = 7
    sensor_weight_range: tuple = (-5.0, 5.0)
    motor_gain: float = 5.0
    catch_radius: float = 2.0
    avoid_margin: float = 6.0

    @property
    def ctrnn_spec(self) -> GenomeSpec:
        return GenomeSpec(size=self.n_neurons, tau_range=(0.5, 10.0),
                          bias_range=(-16.0, 16.0), weight_range=(-16.0, 16.0))


def rel_genome_length(spec: RelGenomeSpec) -> int:
    return genome_length(spec.ctrnn_spec) + spec.n_neurons * spec.n_sensors


def decode_agent(genome, spec: RelGenomeSpec, dt=0.1) -> RelationalAgent:
    genome = np.asarray(genome, dtype=float)
    if genome.size != rel_genome_length(spec):
        raise ValueError(
            f"genome length {genome.size} != expected {rel_genome_length(spec)}")
    nctr = genome_length(spec.ctrnn_spec)
    net = decode(genome[:nctr], spec.ctrnn_spec)
    net.dt = dt
    sw = genome[nctr:].reshape(spec.n_neurons, spec.n_sensors)
    body = CategoricalBody(n_sensors=spec.n_sensors)
    return RelationalAgent(net, body, sw,
                           motor_indices=(spec.n_neurons - 2, spec.n_neurons - 1),
                           motor_gain=spec.motor_gain,
                           catch_radius=spec.catch_radius,
                           avoid_margin=spec.avoid_margin)


def rel_bounds(spec: RelGenomeSpec):
    lo, hi = _bounds_arrays(spec.ctrnn_spec)
    n_sw = spec.n_neurons * spec.n_sensors
    lo = np.concatenate([lo, np.full(n_sw, spec.sensor_weight_range[0])])
    hi = np.concatenate([hi, np.full(n_sw, spec.sensor_weight_range[1])])
    return lo, hi


def random_rel_genome(spec: RelGenomeSpec, rng) -> np.ndarray:
    ctr = random_genome(spec.ctrnn_spec, rng)
    sw = rng.uniform(*spec.sensor_weight_range, size=spec.n_neurons * spec.n_sensors)
    return np.concatenate([ctr, sw])
