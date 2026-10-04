# Active Categorical Perception (Beer 2003) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The first embodied Beer model: a horizontal agent with a fan of ray sensors that catches falling circles and avoids falling diamonds, built on the shipped CTRNN brain + GA, with the categorization behaviour and its decision dynamics as studies.

**Architecture:** A process-bigraph composite of three processes — `FallingObjectEnvironment` (the world), `CategoricalPerceptionBody` (morphology: ray sensors + horizontal effector), and the shared `CTRNNProcess` (the brain) — wired Environment → Body(sense) → CTRNN → Body(act) → Body(pose) → Environment. An extended genome (CTRNN params + sensor-input weights) is evolved by the shipped `evolve()` against a catch/avoid fitness; the best genome is committed and replayed deterministically by the studies.

**Tech Stack:** Python ≥3.11, numpy, scipy, process-bigraph, the shipped `viva_bbe_systems` core (`ctrnn`, `genome`, `analysis`, `evolution`, `viz`), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-viva-bbe-systems-design.md` (§ active-categorical-perception)

## Global Constraints

- Reuse the shipped core unchanged: `CTRNNProcess` (ports `sensory_input`→`motor_output`/`neuron_states`, `"array[float]"`, addressed `local:CTRNNProcess`), `ctrnn.CTRNN`, `evolution.evolve(fitness_fn, spec)`, `analysis.*`, `viz.*`. Processes auto-register by class name via `build_core()` — do NOT edit `core.py`.
- Clean-room reproduction of Beer (2003), *The dynamics of active categorical perception in an evolved model agent*, Adaptive Behavior 11(4):209–243. Label it a reproduction.
- Agent geometry (fixed unless a step says otherwise): agent moves on a horizontal line at y=0; 7 ray sensors fanned symmetrically over a half-angle of π/6 about vertical (straight up); object falls from y=H at constant negative vy; horizontal offset is the trial parameter. Sensor returns `(max_range − dist)/max_range` clamped to [0,1] on intersection, else 0.
- Determinism: every run/evolution takes an explicit seed. The committed seeded genome makes the studies deterministic. Non-finite state fails loud (inherited from `CTRNNProcess`).
- Investigation + studies created on `main` as canonical YAML under `workspace/investigations/active-categorical-perception/` (single-branch repo convention, per the foundation). Figures rendered by a gallery module into the investigation's `figures/`.
- No AI attribution in any commit message. Dashboard visuals AI-free.

## Review Focus

- **Ray misses the object entirely:** a sensor whose ray never intersects the object must read exactly 0, not a NaN or a negative value. (Pinned in Task 1.)
- **Object horizontally far off-screen:** all sensors read 0; the agent must still integrate without error and simply not track. (Pinned in Task 2.)
- **Agent reaches the object / passes it:** final-distance metric and effector must stay finite when the agent is directly under or beyond the object. (Pinned in Task 3.)
- **Genome of wrong length for the extended encoding:** decode must raise, not silently truncate (the extended genome adds sensor weights to the CTRNN block). (Pinned in Task 4.)
- **Degenerate fitness (agent never moves):** the fitness must still rank a non-mover below a tracker, and be finite when no object is ever caught. (Pinned in Task 5.)

---

### Task 1: Falling-object environment + ray geometry

**Files:**
- Create: `viva_bbe_systems/environments/__init__.py`, `viva_bbe_systems/environments/falling_objects.py`
- Test: `tests/test_falling_objects.py`

**Interfaces:**
- Produces:
  - `Shape` = `"circle" | "diamond"`.
  - `ray_distance(origin: (2,), direction_unit: (2,), obj_center: (2,), size: float, shape: Shape) -> float | None` — distance from `origin` to the first intersection of the ray with the object (circle of radius `size`, or axis-aligned diamond / L1 ball of "radius" `size`), or `None` if the ray misses.
  - `class FallingObject(center, vy, size, shape)` with `step(dt)` advancing `center` by `vy*dt` in −y; read-only geometry via `distance_along(origin, direction_unit)` delegating to `ray_distance`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_falling_objects.py
import numpy as np
import pytest
from viva_bbe_systems.environments.falling_objects import ray_distance, FallingObject


def test_ray_straight_up_hits_circle_center():
    # object centered directly above origin at height 10, radius 1 -> near face at 9
    d = ray_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                     np.array([0.0, 10.0]), 1.0, "circle")
    assert d == pytest.approx(9.0, abs=1e-6)


def test_ray_misses_returns_none():
    d = ray_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                     np.array([5.0, 10.0]), 1.0, "circle")
    assert d is None


def test_ray_hits_diamond_point():
    # diamond (L1 ball) size 1 centered above at 10 -> bottom vertex at y=9
    d = ray_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                     np.array([0.0, 10.0]), 1.0, "diamond")
    assert d == pytest.approx(9.0, abs=1e-6)


def test_falling_object_descends():
    obj = FallingObject(center=np.array([0.0, 10.0]), vy=2.0, size=1.0, shape="circle")
    obj.step(0.5)
    assert obj.center[1] == pytest.approx(9.0)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_falling_objects.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/environments/falling_objects.py
"""Falling circle / diamond objects + ray-intersection geometry (Beer 2003)."""
from __future__ import annotations
import numpy as np

Shape = str  # "circle" | "diamond"


def ray_distance(origin, direction_unit, obj_center, size, shape) -> float | None:
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction_unit, dtype=float)
    c = np.asarray(obj_center, dtype=float)
    if shape == "circle":
        # |o + t d - c|^2 = size^2 ; smallest t >= 0
        f = o - c
        b = 2.0 * f.dot(d)
        cc = f.dot(f) - size * size
        disc = b * b - 4.0 * cc
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        for t in sorted(((-b - sq) / 2.0, (-b + sq) / 2.0)):
            if t >= 0:
                return float(t)
        return None
    if shape == "diamond":
        # L1 ball: |x-cx| + |y-cy| <= size. March the ray against the 4 edges.
        best = None
        verts = [c + np.array(v) * size for v in
                 ((1, 0), (0, 1), (-1, 0), (0, -1))]
        edges = [(verts[i], verts[(i + 1) % 4]) for i in range(4)]
        for p0, p1 in edges:
            t = _ray_segment(o, d, p0, p1)
            if t is not None and (best is None or t < best):
                best = t
        return best
    raise ValueError(f"unknown shape {shape!r}")


def _ray_segment(o, d, p0, p1):
    # solve o + t d = p0 + u (p1-p0), t>=0, 0<=u<=1
    e = p1 - p0
    denom = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(denom) < 1e-12:
        return None
    diff = p0 - o
    t = (diff[0] * (-e[1]) - diff[1] * (-e[0])) / denom
    u = (d[0] * diff[1] - d[1] * diff[0]) / denom
    if t >= 0 and 0.0 <= u <= 1.0:
        return float(t)
    return None


class FallingObject:
    def __init__(self, center, vy, size, shape):
        self.center = np.array(center, dtype=float)
        self.vy = float(vy)
        self.size = float(size)
        self.shape = shape

    def step(self, dt):
        self.center = self.center + np.array([0.0, -self.vy * dt])

    def distance_along(self, origin, direction_unit):
        return ray_distance(origin, direction_unit, self.center, self.size, self.shape)
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_falling_objects.py -v` → PASS (4).

- [ ] **Step 5: Commit** — `git add viva_bbe_systems/environments tests/test_falling_objects.py && git commit -m "feat: falling-object environment + ray-intersection geometry"`

---

### Task 2: Categorical-perception body (sensors + effector)

**Files:**
- Create: `viva_bbe_systems/bodies/__init__.py`, `viva_bbe_systems/bodies/categorical_perception.py`
- Test: `tests/test_categorical_body.py`

**Interfaces:**
- Consumes: `FallingObject` (Task 1).
- Produces:
  - `class CategoricalBody(n_sensors=7, half_angle=pi/6, max_range=20.0, x=0.0)`:
    - `sensor_rays() -> list[(2,)]` unit directions fanned symmetrically about vertical.
    - `sense(obj) -> np.ndarray` length `n_sensors` of `(max_range−dist)/max_range` clamped to [0,1], 0 on miss (origin = `(x, 0)`).
    - `act(motor_output: (2,), dt, gain) -> None` updates `x += gain*(motor[0]-motor[1])*dt`.
    - `distance_to(obj) -> float` horizontal `|x − obj.center[0]|` (the categorization metric).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_categorical_body.py
import numpy as np
import pytest
from viva_bbe_systems.bodies.categorical_perception import CategoricalBody
from viva_bbe_systems.environments.falling_objects import FallingObject


def test_centered_object_is_symmetric():
    body = CategoricalBody(n_sensors=7)
    obj = FallingObject(center=np.array([0.0, 10.0]), vy=1.0, size=1.0, shape="circle")
    s = body.sense(obj)
    assert s.shape == (7,)
    assert s[0] == pytest.approx(s[-1], abs=1e-6)        # symmetric
    assert s.argmax() == 3                                # center sensor strongest


def test_offset_object_is_asymmetric():
    body = CategoricalBody(n_sensors=7)
    obj = FallingObject(center=np.array([3.0, 10.0]), vy=1.0, size=1.0, shape="circle")
    s = body.sense(obj)
    assert s.argmax() > 3                                 # peak shifts toward the object


def test_far_object_all_zero():
    body = CategoricalBody(n_sensors=7)
    obj = FallingObject(center=np.array([100.0, 10.0]), vy=1.0, size=1.0, shape="circle")
    assert np.all(body.sense(obj) == 0.0)


def test_effector_moves_toward_more_active_motor():
    body = CategoricalBody()
    body.act(np.array([1.0, 0.0]), dt=1.0, gain=1.0)
    assert body.x > 0
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/bodies/categorical_perception.py
"""Beer-2003 categorical-perception agent body: a horizontal mover with a fan of
ray distance sensors and a two-motor horizontal effector."""
from __future__ import annotations
import numpy as np


class CategoricalBody:
    def __init__(self, n_sensors=7, half_angle=np.pi / 6, max_range=20.0, x=0.0):
        self.n_sensors = int(n_sensors)
        self.half_angle = float(half_angle)
        self.max_range = float(max_range)
        self.x = float(x)

    def sensor_rays(self):
        if self.n_sensors == 1:
            angles = [0.0]
        else:
            angles = np.linspace(-self.half_angle, self.half_angle, self.n_sensors)
        # angle 0 = straight up (+y); +angle tilts toward +x
        return [np.array([np.sin(a), np.cos(a)]) for a in angles]

    def sense(self, obj) -> np.ndarray:
        origin = np.array([self.x, 0.0])
        out = np.zeros(self.n_sensors)
        for i, d in enumerate(self.sensor_rays()):
            dist = obj.distance_along(origin, d)
            if dist is not None and dist <= self.max_range:
                out[i] = (self.max_range - dist) / self.max_range
        return out

    def act(self, motor_output, dt, gain) -> None:
        m = np.asarray(motor_output, dtype=float)
        self.x = self.x + gain * (m[0] - m[1]) * dt

    def distance_to(self, obj) -> float:
        return float(abs(self.x - obj.center[0]))
```

- [ ] **Step 4: Run to verify pass** — PASS (4).

- [ ] **Step 5: Commit** — `git add viva_bbe_systems/bodies tests/test_categorical_body.py && git commit -m "feat: categorical-perception body (ray sensors + horizontal effector)"`

---

### Task 3: Agent runner (brain+body+environment loop)

**Files:**
- Create: `viva_bbe_systems/agents/__init__.py`, `viva_bbe_systems/agents/categorical_agent.py`
- Test: `tests/test_categorical_agent.py`

**Interfaces:**
- Consumes: `CTRNN` (core), `CategoricalBody`, `FallingObject`.
- Produces:
  - `class CategoricalAgent(ctrnn, body, sensor_weights, motor_indices=(−2,−1), motor_gain=5.0)` — holds a `CTRNN`, a `CategoricalBody`, and a `sensor_weights` matrix `(N, n_sensors)` mapping sensor readings to the per-neuron current `I = sensor_weights @ sense`.
  - `run_trial(obj_offset, shape, *, H=20.0, vy=1.0, dt=0.1, steps=200) -> dict` with `{"final_distance": float, "trajectory": np.ndarray (steps,2) of (agent_x, obj_x), "caught": bool}`. Resets CTRNN + body each call (deterministic). `caught` = final_distance below a catch radius.

This runner is the plain-Python engine the fitness and studies use; the process-bigraph composite (Task 7) wraps the same pieces for the dashboard.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_categorical_agent.py
import numpy as np
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.bodies.categorical_perception import CategoricalBody
from viva_bbe_systems.agents.categorical_agent import CategoricalAgent


def _agent(n=5, n_sensors=7):
    net = CTRNN(n, dt=0.1)
    net.weights[:] = 0.0
    body = CategoricalBody(n_sensors=n_sensors)
    sw = np.zeros((n, n_sensors))
    return CategoricalAgent(net, body, sw, motor_indices=(n - 2, n - 1), motor_gain=5.0)


def test_run_trial_shape_and_finiteness():
    agent = _agent()
    out = agent.run_trial(obj_offset=2.0, shape="circle", steps=50)
    assert out["trajectory"].shape == (50, 2)
    assert np.isfinite(out["final_distance"])


def test_zero_weights_agent_does_not_move():
    # no CTRNN weights, no sensor weights -> motor outputs constant -> but equal -> no net motion
    agent = _agent()
    out = agent.run_trial(obj_offset=3.0, shape="circle", steps=50)
    assert out["trajectory"][0, 0] == 0.0
    assert abs(out["trajectory"][-1, 0]) < 1e-6   # symmetric motors => stays put


def test_agent_passes_object_stays_finite():
    agent = _agent()
    out = agent.run_trial(obj_offset=0.0, shape="circle", steps=200)
    assert np.all(np.isfinite(out["trajectory"]))
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/agents/categorical_agent.py
"""Plain-Python brain+body+environment loop for the categorical-perception agent."""
from __future__ import annotations
import numpy as np
from ..environments.falling_objects import FallingObject


class CategoricalAgent:
    def __init__(self, ctrnn, body, sensor_weights, motor_indices=(-2, -1),
                 motor_gain=5.0, catch_radius=1.5):
        self.ctrnn = ctrnn
        self.body = body
        self.sensor_weights = np.asarray(sensor_weights, dtype=float)
        self.motor_indices = tuple(motor_indices)
        self.motor_gain = float(motor_gain)
        self.catch_radius = float(catch_radius)

    def run_trial(self, obj_offset, shape, *, H=20.0, vy=1.0, dt=0.1, steps=200,
                  obj_size=1.0, start_x=0.0):
        self.ctrnn.reset(np.zeros(self.ctrnn.size))
        self.body.x = float(start_x)
        obj = FallingObject(center=np.array([float(obj_offset), H]), vy=vy,
                            size=obj_size, shape=shape)
        traj = np.zeros((steps, 2))
        for t in range(steps):
            I = self.sensor_weights @ self.body.sense(obj)
            o = self.ctrnn.step(external_input=I)
            motor = np.array([o[self.motor_indices[0]], o[self.motor_indices[1]]])
            self.body.act(motor, dt=dt, gain=self.motor_gain)
            obj.step(dt)
            traj[t] = (self.body.x, obj.center[0])
        final_distance = self.body.distance_to(obj)
        return {"final_distance": final_distance, "trajectory": traj,
                "caught": final_distance <= self.catch_radius}
```

- [ ] **Step 4: Run to verify pass** — PASS (3).

- [ ] **Step 5: Commit** — `git add viva_bbe_systems/agents tests/test_categorical_agent.py && git commit -m "feat: categorical-perception agent runner (brain+body+env loop)"`

---

### Task 4: Extended genome (CTRNN + sensor weights)

**Files:**
- Create: `viva_bbe_systems/bodies/categorical_genome.py`
- Test: `tests/test_categorical_genome.py`

**Interfaces:**
- Consumes: `genome.GenomeSpec/genome_length/encode/decode` (core), `CTRNN`, `CategoricalBody`, `CategoricalAgent`.
- Produces:
  - `CatGenomeSpec(n_neurons=5, n_sensors=7, sensor_weight_range=(-5,5), motor_gain=5.0)`.
  - `cat_genome_length(spec)` = `genome_length(GenomeSpec(n_neurons)) + n_neurons*n_sensors`.
  - `decode_agent(genome, spec, dt=0.1) -> CategoricalAgent` — first block is the CTRNN genome (reuses core `decode`), remaining block is the `(n_neurons, n_sensors)` sensor-weight matrix. **Raises `ValueError`** on wrong length.
  - `random_cat_genome(spec, rng)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_categorical_genome.py
import numpy as np
import pytest
from viva_bbe_systems.bodies.categorical_genome import (
    CatGenomeSpec, cat_genome_length, decode_agent, random_cat_genome)
from viva_bbe_systems.agents.categorical_agent import CategoricalAgent


def test_length():
    spec = CatGenomeSpec(n_neurons=5, n_sensors=7)
    assert cat_genome_length(spec) == (5 + 5 + 25) + 5 * 7


def test_decode_builds_agent():
    spec = CatGenomeSpec()
    g = random_cat_genome(spec, np.random.default_rng(0))
    agent = decode_agent(g, spec)
    assert isinstance(agent, CategoricalAgent)
    assert agent.sensor_weights.shape == (spec.n_neurons, spec.n_sensors)


def test_decode_rejects_wrong_length():
    spec = CatGenomeSpec()
    with pytest.raises(ValueError, match="length"):
        decode_agent(np.zeros(3), spec)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/bodies/categorical_genome.py
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
```

- [ ] **Step 4: Run to verify pass** — PASS (3).

- [ ] **Step 5: Commit** — `git add viva_bbe_systems/bodies/categorical_genome.py tests/test_categorical_genome.py && git commit -m "feat: extended categorical-perception genome (CTRNN + sensor weights)"`

---

### Task 5: Catch/avoid fitness

**Files:**
- Create: `viva_bbe_systems/tasks/__init__.py`, `viva_bbe_systems/tasks/categorical_fitness.py`
- Test: `tests/test_categorical_fitness.py`

**Interfaces:**
- Consumes: `decode_agent`, `CatGenomeSpec`.
- Produces:
  - `catch_avoid_fitness(genome, spec, *, offsets=(-6,-3,0,3,6), dt=0.1, steps=200) -> float` — for each (offset, shape∈{circle,diamond}) run a trial; reward = for circles `(span − final_distance)`, for diamonds `final_distance` (both normalized by `span`); averaged. Higher is better. Returns a finite float even when nothing is caught.
  - `make_fitness(spec, **kw) -> callable` returning `genome -> float` for the GA.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_categorical_fitness.py
import numpy as np
from viva_bbe_systems.bodies.categorical_genome import CatGenomeSpec, random_cat_genome
from viva_bbe_systems.tasks.categorical_fitness import catch_avoid_fitness


def test_fitness_is_finite_and_deterministic():
    spec = CatGenomeSpec()
    g = random_cat_genome(spec, np.random.default_rng(1))
    a = catch_avoid_fitness(g, spec)
    b = catch_avoid_fitness(g, spec)
    assert np.isfinite(a) and a == b


def test_nonmover_scores_below_perfect_tracker():
    spec = CatGenomeSpec()
    nonmover = np.zeros(  # all-zero genome => no motion
        __import__("viva_bbe_systems.bodies.categorical_genome",
                   fromlist=["cat_genome_length"]).cat_genome_length(spec))
    score = catch_avoid_fitness(nonmover, spec)
    # a non-mover catches nothing and avoids nothing perfectly; bounded, finite, < max
    assert np.isfinite(score)
    assert score < 1.0
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/tasks/categorical_fitness.py
"""Catch-circles / avoid-diamonds fitness for the categorical-perception agent."""
from __future__ import annotations
import numpy as np
from ..bodies.categorical_genome import decode_agent, CatGenomeSpec


def catch_avoid_fitness(genome, spec: CatGenomeSpec, *,
                        offsets=(-6.0, -3.0, 0.0, 3.0, 6.0),
                        dt=0.1, steps=200, span=12.0) -> float:
    agent = decode_agent(genome, spec, dt=dt)
    scores = []
    for shape in ("circle", "diamond"):
        for off in offsets:
            out = agent.run_trial(obj_offset=off, shape=shape, dt=dt, steps=steps)
            fd = min(out["final_distance"], span)
            if shape == "circle":
                scores.append((span - fd) / span)     # closer is better
            else:
                scores.append(fd / span)              # farther is better
    return float(np.mean(scores))


def make_fitness(spec: CatGenomeSpec, **kw):
    return lambda g: catch_avoid_fitness(g, spec, **kw)
```

- [ ] **Step 4: Run to verify pass** — PASS (2).

- [ ] **Step 5: Commit** — `git add viva_bbe_systems/tasks tests/test_categorical_fitness.py && git commit -m "feat: catch/avoid fitness for categorical-perception agent"`

---

### Task 6: Evolve + commit the seeded genome

**Files:**
- Create: `viva_bbe_systems/tasks/evolve_categorical.py` (a thin, seeded driver), `viva_bbe_systems/data/genomes/` (output dir)
- Test: `tests/test_evolve_categorical.py` (a FAST smoke test only — not the full evolution)

**Interfaces:**
- Consumes: `evolution.evolve`, `make_fitness`, `CatGenomeSpec`, `random_cat_genome`.
- Produces:
  - `evolve_categorical(*, pop_size=60, generations=80, seed=0, mutation_sd=0.8) -> dict` — runs the GA against `make_fitness(CatGenomeSpec())` over a `GenomeSpec`-compatible flat genome of length `cat_genome_length`, returns `{"best_genome", "best_fitness", "history"}`.
  - `save_seed(result, path)` / `load_seed(path)` — numpy `.npz` with the genome + spec metadata.
  - CLI `python -m viva_bbe_systems.tasks.evolve_categorical` → evolves and writes `viva_bbe_systems/data/genomes/categorical_perception.npz`, prints achieved fitness + per-shape mean catch/avoid distances.

Note: `evolve()` takes a `GenomeSpec` for mutation bounds. Use a `GenomeSpec` whose length equals `cat_genome_length(CatGenomeSpec())` by widening it — or add a `flat_length` override. Simplest: construct `GenomeSpec(size=k)` such that `genome_length==cat_length` is NOT generally possible, so **extend `evolve` usage** by passing a dedicated bounds object: add `evolve_flat(fitness_fn, length, lo, hi, **kw)` to `evolution.py` in this task (a thin wrapper around the same GA that takes explicit flat bounds instead of a `GenomeSpec`). Keep the existing `evolve` untouched.

- [ ] **Step 1: Write failing FAST smoke test**

```python
# tests/test_evolve_categorical.py
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
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement** `evolve_flat` in `evolution.py` (thin wrapper taking explicit `length, lo, hi`), then `evolve_categorical.py` using it with `make_fitness`. (Full code: mirror `evolve`'s loop but seed the population with `rng.uniform(lo, hi, (pop, length))` and clip to `[lo,hi]`.)

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_evolve_categorical.py -v` → PASS (fast; tiny budget).

- [ ] **Step 5: Produce the real seed (controller-run, not in the suite)**

Run `python -m viva_bbe_systems.tasks.evolve_categorical` (full budget). Inspect the printed per-shape catch/avoid means. If the agent does not separate circles from diamonds, tune: raise generations/pop, widen `sensor_weight_range`, or increase `motor_gain`. Commit the resulting `data/genomes/categorical_perception.npz` **and record its achieved metrics** (they become the study's acceptance band — report honestly, including partial separation).

- [ ] **Step 6: Commit** — `git add viva_bbe_systems/tasks/evolve_categorical.py viva_bbe_systems/evolution.py viva_bbe_systems/data/genomes/categorical_perception.npz tests/test_evolve_categorical.py && git commit -m "feat: evolve + commit seeded categorical-perception genome"`

---

### Task 7: Process-bigraph composite (dashboard-runnable BBE)

**Files:**
- Create: `viva_bbe_systems/processes/categorical_env_process.py`, `viva_bbe_systems/processes/categorical_body_process.py`, `viva_bbe_systems/composites/categorical-perception.composite.yaml`
- Test: `tests/test_categorical_composite.py`

**Interfaces:**
- Produces two auto-discovered Processes wrapping the Task-1/2 pieces, wired with `CTRNNProcess`:
  - `FallingObjectEnvironment` (Process): state object center/shape; outputs `obj_center`, `obj_size`, `obj_shape`; `update` advances the fall.
  - `CategoricalBodyProcess` (Process): inputs `obj_center`/`obj_size`/`obj_shape` + `motor_output`; computes `sensory_input` (= `sensor_weights @ sense`, sensor_weights from config/seed) and updates `x`; outputs `sensory_input`, `agent_x`.
  - Composite YAML wiring Environment → Body → `CTRNNProcess` → Body, loading the seeded genome from config.
- Test: the composite loads via `build_core()`, runs N steps, and the agent's `x` responds to a circle (moves toward it) — a genuine behavioral run.

- [ ] Steps 1–5 as the standard TDD cycle (RED composite-missing → implement processes + YAML → `build_core()` discovers both → composite runs and `agent_x` changes toward a circle → commit). Use `"array[float]"` ports and string shape via a scalar store. Reuse `CategoricalBody.sense/act` inside the body process so geometry is not duplicated.

---

### Task 8: Studies + visualizations + report

**Files:**
- Create: `viva_bbe_systems/categorical_gallery.py`; `workspace/investigations/active-categorical-perception/investigation.yaml`; three `studies/<slug>/study.yaml`; figures.
- Test: `tests/test_categorical_gallery.py` (renders at low resolution; asserts non-empty figures).

- [ ] **Step 1 (gallery):** render, from the committed seed, (a) **catch/avoid trajectories** (agent_x vs time for circles and diamonds at several offsets), (b) the **categorization map** (final_distance vs object offset, circle vs diamond — the Beer categorization plot), (c) a **decision-dynamics** figure (CTRNN neuron-state trajectories / a 2-projection phase plot during a catch vs an avoid). Reuse `viz`/`analysis` where applicable.
- [ ] **Step 2 (investigation + studies):** write `investigation.yaml` (v2 spine: executive/scientific_argument/biological_story, competing_frameworks, glossary) + three studies (`catch-and-avoid`, `categorization-map`, `decision-dynamics`) with the achieved metrics from Task 6 as acceptance bands, each citing Beer (2003). Mark honestly if separation is partial.
- [ ] **Step 3 (report):** `/viva-report` (Pass A audit + Pass B lint + render). Resolve blocking findings. Commit.
- [ ] **Step 4 (commit):** `git add` the gallery, figures, investigation, studies, rendered report; commit.

---

## Self-review notes (author)

- Coverage: environment geometry (T1), body sensors/effector (T2), agent loop (T3), extended genome (T4), fitness (T5), evolved seed (T6), dashboard composite (T7), studies+viz+report (T8) — every spec bullet for this model is covered.
- The one open risk is scientific, not structural: **whether evolution finds an agent that genuinely separates circles from diamonds** within a reasonable budget (Task 6). The plan handles this by (a) making the fitness/metrics honest and (b) requiring the study to report the *achieved* separation rather than asserting a pre-decided number. If evolution underperforms, that is a finding to surface, not a test to weaken.
- `evolve_flat` is the only addition to the shipped core (a thin wrapper); `evolve` itself is untouched, honoring the foundation's reviewed GA.
