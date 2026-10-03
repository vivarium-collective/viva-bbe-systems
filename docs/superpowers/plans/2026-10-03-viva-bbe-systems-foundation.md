# viva-bbe-systems Foundation + CTRNN Parameter-Space — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `viva-bbe-systems` workspace with the shared CTRNN brain, the dynamical-systems analysis library, a generic composite-evolving GA, and a complete brain-only **CTRNN parameter-space** investigation (Beer 2006/2022) — a runnable, analyzable, publishable slice that every later BBE model builds on.

**Architecture:** A `viva_bbe_systems` process-bigraph package. The CTRNN is a pure-numpy class wrapped as a `CTRNNProcess`; `analysis.py` computes equilibria/stability/nullclines/bifurcations over that class; `evolution.py` evolves any composite exposing a flat genome; `viz.py` renders the phase portraits and bifurcation maps. The parameter-space investigation uses the brain + analysis directly (no body/environment).

**Tech Stack:** Python ≥3.11, numpy, scipy (root-finding + eigenvalues), process-bigraph / bigraph-schema, pbg-superpowers, vivarium-dashboard (dev), matplotlib (dev), pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-viva-bbe-systems-design.md`

## Global Constraints

- Python requires-python `>=3.11`; package importable as `viva_bbe_systems`; repo at `~/code/viva-bbe-systems`.
- Runtime deps: `process-bigraph`, `bigraph-schema`, `numpy`, `scipy`, `pbg-superpowers @ git+https://github.com/vivarium-collective/pbg-superpowers.git@main`. Dev extra: `vivarium-dashboard @ git+…@main`, `matplotlib`. Test extra: `pytest`. `[tool.hatch.metadata] allow-direct-references = true`.
- CTRNN convention: `tau_i ẏ_i = −y_i + Σ_j w[i,j]·σ(y_j+θ_j) + I_i`, `σ(x)=1/(1+e^{-x})`, output `o_i=σ(y_i+θ_i)`. Weight index convention: `w[i,j]` is the connection **from** neuron `j` **to** neuron `i` (so the drive term is `w @ o`).
- All randomness takes an explicit seed; studies are deterministic. Non-finite CTRNN state fails loud (raise), never emits.
- Processes are labeled clean-room reproductions of the cited Beer papers. **No AI attribution** in any commit message or PR. Dashboard visuals are AI-free.
- Studies/investigations are created through `/viva-study` and `/viva-investigation` (canonical `study.yaml` + provenance), never hand-edited around the skills.

## Review Focus

- **CTRNN overflow:** large activations push `exp(-x)` to overflow/`inf`→`nan`. Expected: integration raises a clear error rather than emitting NaNs. (Pinned in Task 2.)
- **Genome shape mismatch:** a genome vector of the wrong length is decoded into a CTRNN of a different size. Expected: loader raises with a shape message, not a silent reshape. (Pinned in Task 4.)
- **Non-convergent root-finding:** `equilibria()` on a system where a start point doesn't converge. Expected: non-converged starts are discarded, not returned as spurious fixed points. (Pinned in Task 5.)
- **Duplicate equilibria:** multiple starts converge to the same fixed point. Expected: results are de-duplicated within a tolerance. (Pinned in Task 5.)
- **Bifurcation sweep ordering:** sweep values given unsorted / with stability changes. Expected: each value's equilibria+stability are reported against that value, order-independent. (Pinned in Task 6.)

---

### Task 1: Scaffold workspace + package skeleton

**Files:**
- Create (via skill): repo `~/code/viva-bbe-systems` with `workspace.yaml`, `pyproject.toml`, `viva_bbe_systems/__init__.py`, `viva_bbe_systems/core.py`
- Move in: `docs/superpowers/specs/2026-10-03-viva-bbe-systems-design.md`, `docs/superpowers/plans/2026-10-03-viva-bbe-systems-foundation.md`, `references/AgmonBeer2014.pdf` (already staged under `~/code/viva-bbe-systems/`)

**Interfaces:**
- Produces: an importable `viva_bbe_systems` package; `viva_bbe_systems.core.build_core()` returning a process-bigraph `Core` with types/processes registered (empty registry for now).

- [ ] **Step 1: Scaffold the repo via the workspace skill**

Invoke `/viva-workspace` in **standalone** mode to clone the pbg-template into `~/code/viva-bbe-systems` on a workspace branch. Set workspace `name: bbe-systems`, `package_path: viva_bbe_systems`. (The `docs/` and `references/` already staged in that directory are preserved / merged.)

- [ ] **Step 2: Set package metadata**

Edit `pyproject.toml` to match Global Constraints (name `viva-bbe-systems`, `requires-python>=3.11`, deps + extras + `allow-direct-references`, `[tool.hatch.build.targets.wheel] packages=["viva_bbe_systems"]`).

- [ ] **Step 3: Minimal `build_core`**

```python
# viva_bbe_systems/core.py
"""Workspace core: register viva-bbe-systems processes + composites."""
from process_bigraph import ProcessTypes


def build_core() -> ProcessTypes:
    core = ProcessTypes()
    # Processes are registered by later tasks:
    #   from .ctrnn import register_ctrnn; register_ctrnn(core)
    return core
```

```python
# viva_bbe_systems/__init__.py
from .core import build_core

__all__ = ["build_core"]
```

- [ ] **Step 4: Install editable + verify import**

Run: `cd ~/code/viva-bbe-systems && uv venv && uv pip install -e '.[dev,test]'`
Run: `python -c "import viva_bbe_systems; viva_bbe_systems.build_core(); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
cd ~/code/viva-bbe-systems
git add -A
git commit -m "chore: scaffold viva-bbe-systems workspace + package skeleton"
```

---

### Task 2: CTRNN integration core

**Files:**
- Create: `viva_bbe_systems/ctrnn.py`
- Test: `tests/test_ctrnn.py`

**Interfaces:**
- Produces:
  - `sigmoid(x: np.ndarray) -> np.ndarray`
  - `class CTRNN(size: int, dt: float = 0.01)` with attributes `tau (N,)`, `theta (N,)`, `weights (N,N)` (`weights[i,j]` = j→i), `y (N,)`; methods `outputs() -> np.ndarray` (= `σ(y+theta)`), `step(external_input=None) -> np.ndarray` (one Euler step of size `dt`, returns new outputs, raises `FloatingPointError` on non-finite state), `reset(y0=None)`.
  - `center_crossing_biases(weights) -> np.ndarray` (`θ_i = −Σ_j weights[i,j]/2`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ctrnn.py
import numpy as np
import pytest
from viva_bbe_systems.ctrnn import CTRNN, sigmoid, center_crossing_biases


def test_single_neuron_relaxes_to_input():
    # tau ẏ = -y + I  (no self-weight) => y* = I
    net = CTRNN(1, dt=0.01)
    net.weights[:] = 0.0
    net.theta[:] = 0.0
    net.reset(np.zeros(1))
    for _ in range(5000):
        net.step(external_input=np.array([5.0]))
    assert net.y[0] == pytest.approx(5.0, abs=1e-3)


def test_center_crossing_biases():
    w = np.array([[4.5, 1.0], [-1.0, 4.5]])
    theta = center_crossing_biases(w)
    assert theta == pytest.approx([-2.75, -1.75])


def test_two_neuron_oscillator_does_not_settle():
    net = CTRNN(2, dt=0.01)
    net.weights[:] = np.array([[4.5, 1.0], [-1.0, 4.5]])
    net.theta[:] = center_crossing_biases(net.weights)
    net.reset(np.array([0.1, -0.1]))
    traj = []
    for _ in range(20000):
        traj.append(net.step()[0])
    tail = np.array(traj[-5000:])
    assert tail.max() - tail.min() > 0.05  # sustained oscillation, not a fixed point


def test_non_finite_state_raises():
    net = CTRNN(1, dt=10.0)
    net.weights[:] = 0.0
    net.reset(np.array([0.0]))
    with pytest.raises(FloatingPointError):
        for _ in range(10000):
            net.step(external_input=np.array([1e6]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ctrnn.py -v`
Expected: FAIL (module/class not defined).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/ctrnn.py
"""Continuous-time recurrent neural network (Beer).

Clean-room reproduction of the CTRNN used across Beer's minimal-cognition
models. State equation (per neuron i):

    tau_i * dy_i/dt = -y_i + sum_j weights[i,j] * sigma(y_j + theta_j) + I_i

with sigma the logistic function and output o_i = sigma(y_i + theta_i).
weights[i,j] is the connection FROM neuron j TO neuron i.
"""
from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def center_crossing_biases(weights: np.ndarray) -> np.ndarray:
    """theta_i = - sum_j weights[i,j] / 2  (Beer's center-crossing condition)."""
    return -0.5 * np.asarray(weights, dtype=float).sum(axis=1)


class CTRNN:
    def __init__(self, size: int, dt: float = 0.01):
        self.size = int(size)
        self.dt = float(dt)
        self.tau = np.ones(self.size)
        self.theta = np.zeros(self.size)
        self.weights = np.zeros((self.size, self.size))
        self.y = np.zeros(self.size)

    def reset(self, y0=None) -> None:
        self.y = np.zeros(self.size) if y0 is None else np.array(y0, dtype=float)

    def outputs(self) -> np.ndarray:
        return sigmoid(self.y + self.theta)

    def derivatives(self, y: np.ndarray, external_input: np.ndarray) -> np.ndarray:
        o = sigmoid(y + self.theta)
        return (-y + self.weights @ o + external_input) / self.tau

    def step(self, external_input=None) -> np.ndarray:
        I = np.zeros(self.size) if external_input is None else np.asarray(external_input, dtype=float)
        self.y = self.y + self.dt * self.derivatives(self.y, I)
        if not np.all(np.isfinite(self.y)):
            raise FloatingPointError("CTRNN state became non-finite (dt too large / unbounded input)")
        return self.outputs()
```

(Note: `sigmoid` clips its argument, so overflow surfaces through the state `y` diverging under too-large `dt`/input, which `step` catches — the `test_non_finite_state_raises` case.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ctrnn.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/ctrnn.py tests/test_ctrnn.py
git commit -m "feat: CTRNN integration core with center-crossing + overflow guard"
```

---

### Task 3: Genome encode/decode contract

**Files:**
- Create: `viva_bbe_systems/genome.py`
- Test: `tests/test_genome.py`

**Interfaces:**
- Produces:
  - `GenomeSpec(size: int, tau_range=(0.5,10.0), bias_range=(-16.0,16.0), weight_range=(-16.0,16.0))` — bounds per Beer's usual ranges.
  - `genome_length(spec) -> int` (= `size + size + size*size`, order: tau, theta, weights-flattened row-major).
  - `encode(ctrnn, spec) -> np.ndarray` (flat genome in [−1,1] normalized to bounds? No — store raw values; normalization handled by GA via bounds).
  - `decode(genome, spec) -> CTRNN` — builds a CTRNN; **raises `ValueError`** if `len(genome) != genome_length(spec)`.
  - `random_genome(spec, rng) -> np.ndarray` — uniform within bounds.
  - `clip_to_bounds(genome, spec) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_genome.py
import numpy as np
import pytest
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.genome import (
    GenomeSpec, genome_length, encode, decode, random_genome, clip_to_bounds,
)


def test_roundtrip_preserves_params():
    spec = GenomeSpec(size=3)
    net = CTRNN(3)
    rng = np.random.default_rng(0)
    net.tau[:] = rng.uniform(0.5, 10.0, 3)
    net.theta[:] = rng.uniform(-5, 5, 3)
    net.weights[:] = rng.uniform(-5, 5, (3, 3))
    g = encode(net, spec)
    net2 = decode(g, spec)
    assert net2.tau == pytest.approx(net.tau)
    assert net2.theta == pytest.approx(net.theta)
    assert net2.weights == pytest.approx(net.weights)


def test_length():
    assert genome_length(GenomeSpec(size=4)) == 4 + 4 + 16


def test_decode_rejects_wrong_length():
    spec = GenomeSpec(size=3)
    with pytest.raises(ValueError, match="length"):
        decode(np.zeros(5), spec)


def test_clip_enforces_bounds():
    spec = GenomeSpec(size=2, tau_range=(1.0, 2.0))
    g = random_genome(spec, np.random.default_rng(1))
    g[0] = 999.0  # first tau out of range
    clipped = clip_to_bounds(g, spec)
    assert 1.0 <= clipped[0] <= 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_genome.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/genome.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_genome.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/genome.py tests/test_genome.py
git commit -m "feat: CTRNN genome encode/decode with bounds + shape validation"
```

---

### Task 4: `CTRNNProcess` + registration + brain-only composite smoke

**Files:**
- Create: `viva_bbe_systems/processes/__init__.py`, `viva_bbe_systems/processes/ctrnn_process.py`
- Modify: `viva_bbe_systems/core.py` (register the process)
- Test: `tests/test_ctrnn_process.py`

**Interfaces:**
- Consumes: `CTRNN`, `center_crossing_biases` (Task 2); `decode`, `GenomeSpec` (Task 3).
- Produces:
  - `class CTRNNProcess(Process)` with config `{size:int, dt:float, tau, theta, weights, motor_indices:list[int]}` (params may be given directly or via `genome`+`GenomeSpec`); ports — inputs `{sensory_input: array[size]}`, outputs `{motor_output: array[len(motor_indices)], neuron_outputs: array[size], neuron_states: array[size]}`; `update(state, interval)` steps the net over `interval` in `dt` increments.
  - `register_ctrnn(core)` registering it under process id `"ctrnn"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ctrnn_process.py
import numpy as np
from viva_bbe_systems.core import build_core


def test_ctrnn_process_runs_in_composite():
    core = build_core()
    spec = {
        "state": {
            "brain": {
                "_type": "process",
                "address": "local:ctrnn",
                "config": {"size": 2, "dt": 0.01,
                           "weights": [[4.5, 1.0], [-1.0, 4.5]],
                           "motor_indices": [1]},
                "inputs": {"sensory_input": ["sensory"]},
                "outputs": {"motor_output": ["motor"], "neuron_states": ["states"]},
            },
            "sensory": [0.0, 0.0],
            "motor": [0.0],
            "states": [0.0, 0.0],
        }
    }
    from process_bigraph import Composite
    sim = Composite(spec, core=core)
    sim.run(1.0)
    results = sim.gather_results()
    assert results is not None  # ran without error; motor produced a value
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ctrnn_process.py -v`
Expected: FAIL (process id `ctrnn` not registered).

- [ ] **Step 3: Implement the process + register it**

```python
# viva_bbe_systems/processes/ctrnn_process.py
"""CTRNNProcess: the shared Beer CTRNN brain as a process-bigraph Process."""
from __future__ import annotations
import numpy as np
from process_bigraph import Process
from ..ctrnn import CTRNN, center_crossing_biases
from ..genome import GenomeSpec, decode


class CTRNNProcess(Process):
    config_schema = {
        "size": "integer",
        "dt": {"_type": "float", "_default": 0.01},
        "tau": "maybe[list[float]]",
        "theta": "maybe[list[float]]",
        "weights": "maybe[list[list[float]]]",
        "genome": "maybe[list[float]]",
        "center_crossing": {"_type": "boolean", "_default": False},
        "motor_indices": {"_type": "list[integer]", "_default": []},
        "initial_state": "maybe[list[float]]",
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        n = self.config["size"]
        if self.config.get("genome") is not None:
            self.net = decode(np.asarray(self.config["genome"]), GenomeSpec(size=n))
            self.net.dt = self.config["dt"]
        else:
            self.net = CTRNN(n, dt=self.config["dt"])
            if self.config.get("weights") is not None:
                self.net.weights[:] = np.asarray(self.config["weights"])
            if self.config.get("tau") is not None:
                self.net.tau[:] = np.asarray(self.config["tau"])
            if self.config.get("theta") is not None:
                self.net.theta[:] = np.asarray(self.config["theta"])
            elif self.config.get("center_crossing"):
                self.net.theta[:] = center_crossing_biases(self.net.weights)
        self.net.reset(self.config.get("initial_state"))
        self.motor_indices = list(self.config["motor_indices"])

    def inputs(self):
        return {"sensory_input": f"array[({self.config['size']},),float]"}

    def outputs(self):
        return {
            "motor_output": f"array[({max(len(self.motor_indices),1)},),float]",
            "neuron_outputs": f"array[({self.config['size']},),float]",
            "neuron_states": f"array[({self.config['size']},),float]",
        }

    def update(self, state, interval):
        I = np.asarray(state.get("sensory_input", np.zeros(self.config["size"])), dtype=float)
        steps = max(1, int(round(interval / self.net.dt)))
        o = self.net.outputs()
        for _ in range(steps):
            o = self.net.step(external_input=I)
        motor = o[self.motor_indices] if self.motor_indices else np.array([0.0])
        return {"motor_output": motor, "neuron_outputs": o, "neuron_states": self.net.y.copy()}


def register_ctrnn(core):
    core.register_process("ctrnn", CTRNNProcess)
    return core
```

```python
# viva_bbe_systems/processes/__init__.py
from .ctrnn_process import CTRNNProcess, register_ctrnn
__all__ = ["CTRNNProcess", "register_ctrnn"]
```

Modify `core.py`:

```python
# viva_bbe_systems/core.py  (update build_core)
from process_bigraph import ProcessTypes
from .processes import register_ctrnn


def build_core() -> ProcessTypes:
    core = ProcessTypes()
    register_ctrnn(core)
    return core
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ctrnn_process.py -v`
Expected: PASS. (If the array type grammar differs in the installed bigraph-schema, adjust the `array[(N,),float]` strings to the registered form; verify with `python -c "from viva_bbe_systems import build_core; build_core()"`.)

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/processes viva_bbe_systems/core.py tests/test_ctrnn_process.py
git commit -m "feat: CTRNNProcess brain + build_core registration"
```

---

### Task 5: Equilibria, Jacobian, stability

**Files:**
- Create: `viva_bbe_systems/analysis.py`
- Test: `tests/test_analysis_equilibria.py`

**Interfaces:**
- Consumes: `CTRNN`, `sigmoid` (Task 2).
- Produces:
  - `equilibria(net, I, n_starts=50, rng=None, tol=1e-8, dedupe_tol=1e-4) -> list[np.ndarray]` — multi-start root-find of `derivatives=0`; **discards non-converged starts**, **de-dupes** within `dedupe_tol`.
  - `jacobian(net, y) -> np.ndarray` — `J[i,j] = (-δ_ij + weights[i,j]·σ'(y_j+θ_j))/τ_i`, `σ'=σ(1−σ)`.
  - `eigenvalues(net, y) -> np.ndarray`.
  - `is_stable(net, y) -> bool` — all eigenvalue real parts < 0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analysis_equilibria.py
import numpy as np
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.analysis import equilibria, jacobian, eigenvalues, is_stable


def _net1(I_self=0.0):
    net = CTRNN(1)
    net.weights[:] = 0.0
    net.theta[:] = 0.0
    return net


def test_single_neuron_unique_equilibrium():
    net = _net1()
    eqs = equilibria(net, np.array([3.0]), rng=np.random.default_rng(0))
    assert len(eqs) == 1
    np.testing.assert_allclose(eqs[0][0], 3.0, atol=1e-4)


def test_jacobian_matches_finite_difference():
    net = CTRNN(2)
    rng = np.random.default_rng(2)
    net.weights[:] = rng.uniform(-3, 3, (2, 2))
    net.theta[:] = rng.uniform(-1, 1, 2)
    y = rng.uniform(-1, 1, 2)
    J = jacobian(net, y)
    eps = 1e-6
    Jfd = np.zeros((2, 2))
    for j in range(2):
        dy = np.zeros(2); dy[j] = eps
        Jfd[:, j] = (net.derivatives(y + dy, np.zeros(2)) - net.derivatives(y - dy, np.zeros(2))) / (2 * eps)
    np.testing.assert_allclose(J, Jfd, atol=1e-5)


def test_stable_fixed_point_detected():
    net = _net1()
    assert is_stable(net, np.array([3.0])) is True


def test_bistable_has_three_equilibria():
    # strong self-excitation -> bistable (two stable, one unstable)
    net = CTRNN(1)
    net.weights[:] = np.array([[8.0]])
    net.theta[:] = np.array([-4.0])  # center-crossing for a single self-excitatory neuron
    eqs = equilibria(net, np.array([0.0]), n_starts=200, rng=np.random.default_rng(3))
    assert len(eqs) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_analysis_equilibria.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/analysis.py
"""Dynamical-systems analysis of CTRNNs: equilibria, stability, nullclines,
bifurcation sweeps. Pure numpy/scipy; no plotting."""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve
from .ctrnn import CTRNN, sigmoid


def equilibria(net: CTRNN, I, n_starts=50, rng=None, tol=1e-8, dedupe_tol=1e-4,
               start_range=20.0):
    I = np.asarray(I, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    found: list[np.ndarray] = []

    def f(y):
        return net.derivatives(y, I)

    starts = [np.zeros(net.size)] + [
        rng.uniform(-start_range, start_range, net.size) for _ in range(n_starts)
    ]
    for y0 in starts:
        sol, info, ier, _ = fsolve(f, y0, full_output=True)
        if ier != 1:
            continue  # did not converge -> discard
        if np.max(np.abs(f(sol))) > tol:
            continue
        if not any(np.linalg.norm(sol - e) < dedupe_tol for e in found):
            found.append(sol)
    return found


def jacobian(net: CTRNN, y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = sigmoid(y + net.theta)
    dsig = s * (1.0 - s)                      # sigma'(y_j + theta_j)
    J = net.weights * dsig[np.newaxis, :]     # weights[i,j] * dsig[j]
    J = J - np.eye(net.size)
    J = J / net.tau[:, np.newaxis]
    return J


def eigenvalues(net: CTRNN, y) -> np.ndarray:
    return np.linalg.eigvals(jacobian(net, y))


def is_stable(net: CTRNN, y) -> bool:
    return bool(np.all(eigenvalues(net, y).real < 0.0))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_analysis_equilibria.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/analysis.py tests/test_analysis_equilibria.py
git commit -m "feat: CTRNN equilibria, Jacobian, and stability analysis"
```

---

### Task 6: Nullclines + bifurcation sweep

**Files:**
- Modify: `viva_bbe_systems/analysis.py`
- Test: `tests/test_analysis_bifurcation.py`

**Interfaces:**
- Consumes: `equilibria`, `is_stable` (Task 5).
- Produces:
  - `nullcline_grid(net, I, neuron, y_range, resolution=200) -> (Y1, Y2, Z)` — for a 2-neuron net, the `dy_{neuron}/dt` field over a grid, for contouring the nullcline at `Z=0`.
  - `bifurcation_sweep(net_factory, param_values, I=0.0, rng=None) -> list[dict]` — `net_factory(value)` returns a configured `CTRNN`; returns, **per value (in input order)**, `{"value":v, "equilibria":[...], "stability":[bool,...]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analysis_bifurcation.py
import numpy as np
from viva_bbe_systems.ctrnn import CTRNN
from viva_bbe_systems.analysis import nullcline_grid, bifurcation_sweep


def test_nullcline_grid_shapes():
    net = CTRNN(2)
    net.weights[:] = np.array([[4.5, 1.0], [-1.0, 4.5]])
    Y1, Y2, Z = nullcline_grid(net, np.zeros(2), neuron=0, y_range=(-5, 5), resolution=30)
    assert Y1.shape == (30, 30) and Z.shape == (30, 30)


def test_bifurcation_sweep_tracks_pitchfork():
    # single self-excitatory neuron: 1 equilibrium at low self-weight, 3 at high
    def factory(w_self):
        net = CTRNN(1)
        net.weights[:] = np.array([[w_self]])
        net.theta[:] = np.array([-w_self / 2.0])  # center-crossing
        return net

    values = [1.0, 8.0]  # given in order; low then high
    out = bifurcation_sweep(factory, values, I=0.0, rng=np.random.default_rng(0))
    assert [r["value"] for r in out] == values
    assert len(out[0]["equilibria"]) == 1
    assert len(out[1]["equilibria"]) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_analysis_bifurcation.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement (append to `analysis.py`)**

```python
def nullcline_grid(net: CTRNN, I, neuron, y_range, resolution=200):
    assert net.size == 2, "nullcline_grid is for 2-neuron systems"
    I = np.asarray(I, dtype=float)
    axis = np.linspace(y_range[0], y_range[1], resolution)
    Y1, Y2 = np.meshgrid(axis, axis)
    Z = np.zeros_like(Y1)
    for a in range(resolution):
        for b in range(resolution):
            y = np.array([Y1[a, b], Y2[a, b]])
            Z[a, b] = net.derivatives(y, I)[neuron]
    return Y1, Y2, Z


def bifurcation_sweep(net_factory, param_values, I=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    results = []
    for v in param_values:
        net = net_factory(v)
        Ivec = np.full(net.size, I) if np.isscalar(I) else np.asarray(I)
        eqs = equilibria(net, Ivec, rng=rng)
        results.append({
            "value": v,
            "equilibria": eqs,
            "stability": [is_stable(net, e) for e in eqs],
        })
    return results
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_analysis_bifurcation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/analysis.py tests/test_analysis_bifurcation.py
git commit -m "feat: nullcline grid + bifurcation sweep"
```

---

### Task 7: Generic composite-evolving GA (`EvolutionStep`)

**Files:**
- Create: `viva_bbe_systems/evolution.py`
- Test: `tests/test_evolution.py`

**Interfaces:**
- Consumes: `GenomeSpec`, `random_genome`, `clip_to_bounds` (Task 3).
- Produces:
  - `evolve(fitness_fn, spec, *, pop_size=50, generations=30, mutation_sd=0.5, seed=0, elitism=1) -> dict` with keys `best_genome`, `best_fitness`, `history` (per-gen best). `fitness_fn(genome: np.ndarray) -> float` (higher is better); the Step is **model-agnostic** — all task/composite logic lives in `fitness_fn`.
  - `class EvolutionStep(Step)` wrapping `evolve` for in-workspace runs: config `{genome_size, fitness_ref, pop_size, generations, mutation_sd, seed}` where `fitness_ref` names a registered fitness callable; outputs `{best_genome, best_fitness}`. (The in-process `evolve` is what tests and `tasks/fitness.py` use; `EvolutionStep` is the dashboard-facing wrapper.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evolution.py
import numpy as np
from viva_bbe_systems.genome import GenomeSpec
from viva_bbe_systems.evolution import evolve


def test_ga_improves_on_trivial_fitness():
    # fitness maximized by driving the genome toward zero
    spec = GenomeSpec(size=2)

    def fitness(genome):
        return -float(np.sum(genome ** 2))

    out = evolve(fitness, spec, pop_size=30, generations=40, mutation_sd=0.5, seed=1)
    assert out["best_fitness"] > out["history"][0]
    assert out["best_fitness"] > -50.0  # got meaningfully close to zero


def test_ga_is_deterministic_under_seed():
    spec = GenomeSpec(size=2)
    f = lambda g: -float(np.sum(g ** 2))
    a = evolve(f, spec, seed=7, generations=10, pop_size=20)
    b = evolve(f, spec, seed=7, generations=10, pop_size=20)
    assert a["best_fitness"] == b["best_fitness"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_evolution.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/evolution.py
"""Generic genetic algorithm that evolves any genome scored by a fitness fn.

Model-agnostic: all task/composite logic lives in `fitness_fn`, which maps a
flat genome to a scalar score (higher = better). A model evolves its BBE
composite by passing a fitness fn that instantiates the composite from the
genome, runs it, and returns the task score (see tasks/fitness.py)."""
from __future__ import annotations
import numpy as np
from .genome import GenomeSpec, random_genome, clip_to_bounds


def evolve(fitness_fn, spec: GenomeSpec, *, pop_size=50, generations=30,
           mutation_sd=0.5, seed=0, elitism=1) -> dict:
    rng = np.random.default_rng(seed)
    pop = np.array([random_genome(spec, rng) for _ in range(pop_size)])
    fits = np.array([fitness_fn(g) for g in pop])
    history = [float(fits.max())]

    for _ in range(generations):
        order = np.argsort(fits)[::-1]       # best first
        pop, fits = pop[order], fits[order]
        new = [pop[i].copy() for i in range(elitism)]   # keep elites
        while len(new) < pop_size:
            # rank-based (truncation) selection from the top half
            parent = pop[rng.integers(0, max(1, pop_size // 2))]
            child = clip_to_bounds(parent + rng.normal(0.0, mutation_sd, parent.shape), spec)
            new.append(child)
        pop = np.array(new)
        fits = np.array([fitness_fn(g) for g in pop])
        history.append(float(fits.max()))

    best = int(np.argmax(fits))
    return {"best_genome": pop[best], "best_fitness": float(fits[best]), "history": history}
```

(The `EvolutionStep` Step wrapper is thin and dashboard-facing; implement it in this same file once a fitness registry exists in a model plan. The foundation ships `evolve`, which the parameter-space plan does not need but the model plans do.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_evolution.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/evolution.py tests/test_evolution.py
git commit -m "feat: generic composite-evolving genetic algorithm"
```

---

### Task 8: Dynamical-analysis visualizations

**Files:**
- Create: `viva_bbe_systems/viz.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `CTRNN`, `analysis.equilibria`, `analysis.nullcline_grid`, `analysis.bifurcation_sweep`.
- Produces (each a `@visualization`-decorated function returning a matplotlib Figure):
  - `phase_portrait_2d(net, I, y_range=(-10,10), resolution=25)` — vector field + both nullclines + equilibria (stable filled, unstable open).
  - `bifurcation_diagram(net_factory, param_values, param_name, I=0.0)` — equilibrium branches vs parameter, colored by stability.

- [ ] **Step 1: Write the failing test** (figure is produced, has expected artists)

```python
# tests/test_viz.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
from viva_bbe_systems.ctrnn import CTRNN, center_crossing_biases
from viva_bbe_systems.viz import phase_portrait_2d, bifurcation_diagram


def test_phase_portrait_returns_figure():
    net = CTRNN(2)
    net.weights[:] = np.array([[4.5, 1.0], [-1.0, 4.5]])
    net.theta[:] = center_crossing_biases(net.weights)
    fig = phase_portrait_2d(net, np.zeros(2), resolution=15)
    assert fig.axes and fig.axes[0].collections  # quiver + contours drawn


def test_bifurcation_diagram_returns_figure():
    def factory(w):
        net = CTRNN(1); net.weights[:] = [[w]]; net.theta[:] = [-w / 2]
        return net
    fig = bifurcation_diagram(factory, np.linspace(0.5, 8.0, 12), "self-weight")
    assert fig.axes and fig.axes[0].lines or fig.axes[0].collections
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_viz.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement** (use the workspace's `@visualization` decorator; if its import path differs, verify with `grep -r "def visualization" $(python -c "import pbg_superpowers,os;print(os.path.dirname(pbg_superpowers.__file__))")` and match it)

```python
# viva_bbe_systems/viz.py
"""Dynamical-systems visualizations (Beer's signature plots). AI-free."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .analysis import equilibria, nullcline_grid, bifurcation_sweep, is_stable

try:
    from pbg_superpowers.viz import visualization
except Exception:  # decorator optional for unit tests
    def visualization(*a, **k):
        def deco(fn):
            return fn
        return deco if (a and callable(a[0]) is False) else (a[0] if a else deco)


@visualization(name="phase_portrait_2d")
def phase_portrait_2d(net, I, y_range=(-10, 10), resolution=25):
    I = np.asarray(I, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    axis = np.linspace(y_range[0], y_range[1], resolution)
    Y1, Y2 = np.meshgrid(axis, axis)
    U = np.zeros_like(Y1); V = np.zeros_like(Y2)
    for a in range(resolution):
        for b in range(resolution):
            d = net.derivatives(np.array([Y1[a, b], Y2[a, b]]), I)
            U[a, b], V[a, b] = d
    ax.quiver(Y1, Y2, U, V, color="0.6", pivot="mid")
    for neuron, color in [(0, "#1f77b4"), (1, "#d62728")]:
        G1, G2, Z = nullcline_grid(net, I, neuron, y_range, resolution=max(resolution, 60))
        ax.contour(G1, G2, Z, levels=[0.0], colors=[color], linewidths=1.5)
    for e in equilibria(net, I, rng=np.random.default_rng(0)):
        stable = is_stable(net, e)
        ax.plot(e[0], e[1], "o", mfc=("k" if stable else "none"), mec="k", ms=9)
    ax.set_xlabel("y1"); ax.set_ylabel("y2"); ax.set_title("CTRNN phase portrait")
    return fig


@visualization(name="bifurcation_diagram")
def bifurcation_diagram(net_factory, param_values, param_name, I=0.0):
    sweep = bifurcation_sweep(net_factory, list(param_values), I=I,
                              rng=np.random.default_rng(0))
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in sweep:
        for e, st in zip(row["equilibria"], row["stability"]):
            ax.plot(row["value"], e[0], ".",
                    color=("k" if st else "r"), ms=6)
    ax.set_xlabel(param_name); ax.set_ylabel("equilibrium y1")
    ax.set_title("Bifurcation diagram (black=stable, red=unstable)")
    return fig
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_viz.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add viva_bbe_systems/viz.py tests/test_viz.py
git commit -m "feat: phase-portrait + bifurcation-diagram visualizations"
```

---

### Task 9: CTRNN parameter-space investigation + studies

**Files:**
- Create (via skills): `investigations/ctrnn-parameter-space/`, `studies/param-space-equilibria-and-nullclines/`, `studies/param-space-codim1-bifurcations/`, `studies/param-space-codim2-structure/`
- Create: `viva_bbe_systems/param_space.py` (small helpers the studies call: `single_neuron_net(self_weight)`, `two_neuron_net(...)`, codim-2 grid over two params)
- Test: `tests/test_param_space.py`

**Interfaces:**
- Consumes: `CTRNN`, `analysis.*`, `viz.*`.
- Produces: `single_neuron_net(self_weight, I=0.0)`, `two_neuron_net(w, I=0.0, center_crossing=True)`, `codim2_equilibria_count(p1_values, p2_values, make_net) -> np.ndarray` (grid of equilibrium counts).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_param_space.py
import numpy as np
from viva_bbe_systems.param_space import single_neuron_net, codim2_equilibria_count


def test_codim2_count_grid_shape_and_bistable_region():
    # vary self-weight and bias of a single neuron; high self-weight near
    # center-crossing -> 3 equilibria (bistable) region exists
    w_vals = np.linspace(0.5, 10.0, 8)
    b_vals = np.linspace(-6.0, 6.0, 8)

    def make(w, b):
        net = single_neuron_net(w)
        net.theta[:] = [b]
        return net

    grid = codim2_equilibria_count(w_vals, b_vals, make)
    assert grid.shape == (8, 8)
    assert grid.max() == 3  # a bistable region is found
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_param_space.py -v`
Expected: FAIL (module not defined).

- [ ] **Step 3: Implement**

```python
# viva_bbe_systems/param_space.py
"""Helpers for the CTRNN parameter-space investigation (Beer 2006, 2022)."""
from __future__ import annotations
import numpy as np
from .ctrnn import CTRNN, center_crossing_biases
from .analysis import equilibria


def single_neuron_net(self_weight, I=0.0):
    net = CTRNN(1)
    net.weights[:] = np.array([[self_weight]])
    net.theta[:] = center_crossing_biases(net.weights)
    return net


def two_neuron_net(w, I=0.0, center_crossing=True):
    net = CTRNN(2)
    net.weights[:] = np.asarray(w, dtype=float)
    if center_crossing:
        net.theta[:] = center_crossing_biases(net.weights)
    return net


def codim2_equilibria_count(p1_values, p2_values, make_net, I=0.0):
    rng = np.random.default_rng(0)
    grid = np.zeros((len(p1_values), len(p2_values)), dtype=int)
    for a, p1 in enumerate(p1_values):
        for b, p2 in enumerate(p2_values):
            net = make_net(p1, p2)
            Ivec = np.full(net.size, I)
            grid[a, b] = len(equilibria(net, Ivec, n_starts=80, rng=rng))
    return grid
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_param_space.py -v`
Expected: PASS.

- [ ] **Step 5: Build the investigation + studies via skills**

Invoke `/viva-investigation new` → `ctrnn-parameter-space` (overview: reproduce Beer's parameter-space structure of small CTRNNs). Then `/viva-study` to create the three studies, each with acceptance bands cited to the source paper via `/viva-cite-bands`:
  - `param-space-equilibria-and-nullclines` — for the 2-neuron center-crossing net, assert equilibrium count + render `phase_portrait_2d`. Band: equilibria match the analytically expected count.
  - `param-space-codim1-bifurcations` — single-neuron self-weight sweep; assert transition 1→3 equilibria; render `bifurcation_diagram`. Cite Beer (2006).
  - `param-space-codim2-structure` — `codim2_equilibria_count` over (self-weight × bias); render the count grid as a parameter-plane heatmap (add a `codim2_heatmap` viz via `/viva-viz`). Cite Beer (2022).

- [ ] **Step 6: Commit**

```bash
git add viva_bbe_systems/param_space.py tests/test_param_space.py investigations studies
git commit -m "feat: CTRNN parameter-space investigation with 3 studies"
```

---

### Task 10: Publish read-only workbench + report

**Files:**
- Modify: none (uses `/viva-report`)

- [ ] **Step 1: Full test sweep**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 2: Reviewer-readiness + render**

Invoke `/viva-report` (Pass A audit, Pass B lint, then render) over the `ctrnn-parameter-space` investigation. Resolve any flagged drift.

- [ ] **Step 3: Publish**

Invoke `/viva-workbench` to serve, confirm the investigation + three studies + visualizations render, then publish the read-only workbench per the ecosystem publish flow.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: publish CTRNN parameter-space investigation report"
```

---

## Later plans (written just-ahead of execution)

Each reuses `CTRNNProcess`, `analysis.*`, `evolution.evolve`, `viz.*` unchanged, adding a body + environment + seeded genome + studies:

- **Plan 2 — active-categorical-perception** (Beer 2003): `bodies/categorical_perception.py` (ray-sensor fan + 2 motors), `environments/falling_objects.py` (circle vs diamond), `tasks/fitness.py` catch/avoid fitness, seeded genome, studies catch-and-avoid / categorization-map / decision-dynamics.
- **Plan 3 — legged-locomotion-cpg** (Beer & Gallagher 1992; Chiel/Beer/Gallagher 1999).
- **Plan 4 — relational-categorization** (Williams, Beer, Gasser 2008).
- **Plan 5 — action-switching** (Agmon & Beer 2014): `bodies/chemotactic_forager.py` (3 morphologies), `environments/chemotaxis_resources.py`, `metabolism.py` (internal nutrient state), action-switching analysis.
