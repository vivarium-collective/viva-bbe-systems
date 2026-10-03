# viva-bbe-systems — Design Spec

**Date:** 2026-10-03
**Status:** Approved design (pre-plan)
**Author:** Eran Agmon (with Claude)

## Purpose

A new sibling viva repo, `viva-bbe-systems`, dedicated to **Randy Beer's minimal
cognitive models** — evolved continuous-time recurrent neural network (CTRNN)
agents studied as coupled **brain–body–environment (BBE)** dynamical systems.

The repo reproduces several of Beer's canonical models as process-bigraph
composites and presents each as an investigation whose studies are Beer's
signature **dynamical-systems analyses** (phase portraits, nullclines,
equilibria, bifurcations, trajectory/categorization maps, gait diagrams).

Success = each model runs end-to-end in the workbench, reproduces the paper's
qualitative behavior from a seeded genome, and ships at least one interactive
dynamical-analysis visualization; the whole set is published as a read-only
workbench + reports.

### Who it's for

The vivarium/viva ecosystem as a teaching + demonstration artifact: the clearest
possible illustration that cognition in these models is a property of the
*coupled* brain–body–environment system, built from independently-understandable
process-bigraph parts.

## Scope

Four models, each an investigation:

1. **Active categorical perception** — Beer (2003), *The dynamics of active
   categorical perception in an evolved model agent*, Adaptive Behavior
   11(4):209–243; precursor Beer (1996), *Toward the evolution of dynamical
   neural networks for minimally cognitive behavior*.
2. **Legged locomotion / CPG walking** — Beer & Gallagher (1992), *Evolving
   dynamical neural networks for adaptive behavior*; Chiel, Beer & Gallagher
   (1999) and Beer, Chiel & Gallagher (1999), *Evolution and analysis of model
   CPGs for walking I & II*.
3. **Relational categorization** — Williams, Beer & Gasser (2008), *An embodied
   dynamical approach to relational categorization*.
4. **CTRNN parameter-space structure** — Beer (2006), *Parameter space structure
   of continuous-time recurrent neural networks*, Neural Computation
   18:3009–3051; Beer (2022), *Codimension-2 parameter space structure of
   continuous-time recurrent neural networks*, Biological Cybernetics 116:501–515.
5. **Action switching in embodied agents** — Agmon & Beer (2014), *The evolution
   and analysis of action switching in embodied agents*, Adaptive Behavior
   22(1):3–20 (builds on the chemotaxis model of Beer 1995 / Beer & Gallagher
   1992). A food-foraging agent that autonomously switches between approaching two
   resources to sustain two internal nutrient levels. The capstone model: the only
   one with **internal metabolic state** and explicit action-switching analysis.
   PDF: `references/AgmonBeer2014.pdf`.

### Decisions (locked)

- **Implementation:** clean-room **numpy** reproduction (not a bridge to Beer's
  C++ `Evolutionary Agents`). Processes are labeled as reproductions of the
  published models.
- **Parameters:** ship a **seeded known-good genome per model** as committed data
  so studies are deterministic and fast; include an **opt-in re-evolution study**
  using the bundled GA.
- **Architecture:** shared BBE core + one investigation per model (approaches A/B/C
  considered; A chosen).

### Out of scope (YAGNI)

- Faithful bit-for-bit reproduction of Beer's exact evolved weights. We reproduce
  *qualitative behavior + dynamical structure*, seeding genomes we evolve/curate
  ourselves.
- The full hexapod (6-leg) locomotion controller. We do the single-leg CPG model;
  a hexapod extension is a documented future study, not built now.
- Information-theoretic analyses (transfer entropy, etc.). Future work.
- GPU / performance work. Models are tiny (≤ ~14 neurons); numpy + Euler/RK is
  ample.

## Architecture

### The BBE seam

Each model (except #4) is a process-bigraph composite of three process types
wired through typed ports:

```
EnvironmentProcess  --geometry/world state-->  BodyProcess  --sensory_input-->  CTRNNProcess
       ^                                            |                                |
       |----------- body pose / coupling -----------|<------- motor_output ----------|
```

- **`CTRNNProcess`** (the brain) — shared, reused by all four models.
  - State: neuron activation vector `y` (length N).
  - Parameters (the genome): time-constants `tau` (N), biases `theta` (N),
    weight matrix `w` (N×N), and the sensor→neuron input gain/weights.
  - Dynamics: `tau_i * dy_i/dt = -y_i + Σ_j w_ji * σ(y_j + theta_j) + I_i`,
    with logistic `σ(x) = 1/(1+e^-x)`. Fixed-step integration (Euler default,
    RK4 optional) at step `dt`.
  - Ports: input `sensory_input` (current vector `I`); output `motor_output`
    (outputs `σ(y_i+theta_i)` of a designated set of motor neurons).
  - Also exposes full state `y` and outputs `o` for analysis.

- **`BodyProcess`** (per model) — morphology + sensor model + effector model.
  - Reads environment geometry and its own pose; computes `sensory_input`.
  - For models with internal state (action-switching), the body assembles
    `sensory_input` as the concatenation of *external* sensors and *internal*
    (metabolic) sensors, so the shared `CTRNNProcess` needs no change.
  - Consumes `motor_output`; maps to actuation; integrates its own pose.

- **`EnvironmentProcess`** (per model) — world state + the geometry the body's
  sensors query (object positions/shapes, ground, object stream).

Composites are defined as `*.composite.yaml` under the package `composites/`
dir, consistent with other viva repos.

### Package layout

```
viva-bbe-systems/
  workspace.yaml                 # schema_version 2, package_path: viva_bbe_systems
  pyproject.toml                 # name viva-bbe-systems; deps below
  README.md
  viva_bbe_systems/
    __init__.py
    core.py                      # build_core(): register processes + composites
    ctrnn.py                     # CTRNNProcess (shared brain)
    evolution.py                 # EvolutionStep (GA) + fitness protocol
    analysis.py                  # dynamical-systems helpers (equilibria, nullclines,
                                 #   bifurcation sweep, Jacobian/eigenvalues)
    viz.py                       # @visualization functions (phase portraits, maps, gaits)
    bodies/
      categorical_perception.py  # mover + ray-sensor fan + 2 motors
      legged.py                  # single leg: joint, foot, force; 3 motors
      relational.py              # vertical catcher
      chemotactic_forager.py     # chemosensor stalks + internal nutrient sensors
                                 #   + 2 force effectors; 3 morphology variants
    environments/
      falling_objects.py         # circles vs diamonds, constant fall
      ground.py                  # flat ground + gravity for legged
      object_stream.py           # two sequential objects for relational
      chemotaxis_resources.py    # bounded plane, 2 resources, exp chemical gradients
    metabolism.py                # internal nutrient-level dynamics (eat/drain/death)
    tasks/
      fitness.py                 # per-task fitness functions used by EvolutionStep
    composites/
      categorical-perception.composite.yaml
      legged-locomotion.composite.yaml
      relational-categorization.composite.yaml
      action-switching.composite.yaml
    data/
      genomes/                   # committed seeded genomes (.npz / .yaml) per model
  studies/                       # per-study dirs (study.yaml etc.) — created via /viva-study
  investigations/                # four investigations — created via /viva-investigation
  references/                    # Beer PDFs / citation metadata
  tests/
  docs/
    superpowers/specs/2026-10-03-viva-bbe-systems-design.md   # this file
```

- **Package name:** `viva_bbe_systems` (viva_ prefix per ecosystem convention;
  repo `viva-bbe-systems`).
- `build_core()` registers every process + composite so the workbench Registry
  and `/viva-run` can resolve them (standard workspace `build_core` registration
  pattern).

### Dependencies

`process-bigraph`, `bigraph-schema`, `numpy`, `pbg-superpowers @ git+…@main`
(runtime). Dev extra: `vivarium-dashboard @ git+…@main`, `matplotlib`, `scipy`
(for eigenvalues / root-finding in `analysis.py`). Test extra: `pytest`.

## The four investigations

### 1. active-categorical-perception

- **Body:** horizontal position `x`; fan of 7 distance ("ray") sensors over a
  limited angular spread; 2 motor neurons mapped to left/right horizontal
  velocity (`v ∝ o_left − o_right`).
- **Environment:** one object per trial falling at constant vertical velocity;
  shape ∈ {circle, diamond} (line as an alternative distractor); horizontal
  offset is the trial parameter.
- **Sensors:** each ray cast upward from the body intersects the object; sensor
  current ∝ inverse distance to intersection (clamped), 0 if no intersection.
- **Studies:**
  - `catch-and-avoid` — agent catches circles (minimize final horizontal
    distance) and avoids diamonds (maximize it) from the seeded genome.
  - `categorization-map` — success/behavior as a function of object horizontal
    offset and shape (Beer's categorization plot).
  - `decision-dynamics` — phase-space / state-trajectory analysis of how the
    coupled system commits to catch vs avoid.
  - `re-evolution` *(opt-in)* — evolve a fresh genome via `EvolutionStep`.

### 2. legged-locomotion-cpg

- **Body:** single leg with joint angle, foot state (up/down) and foot force;
  3 motor neurons (backward-swing torque, forward-swing torque, foot). Body
  advances horizontally when the foot is planted and the leg swings back.
- **Environment:** flat ground + gravity; measures forward body velocity.
- **Studies:**
  - `walking` — sustained forward locomotion from the seeded CPG genome.
  - `cpg-limit-cycle` — phase-portrait / limit-cycle analysis of the pattern
    generator (and the "dynamical modules" framing of Chiel/Beer/Gallagher 1999).
  - `gait-and-velocity` — stride period, duty factor, mean velocity.
  - `re-evolution` *(opt-in)*.

### 3. relational-categorization

- **Body:** vertical/horizontal catcher with a small sensor set.
- **Environment:** two objects presented in sequence; the agent must respond to
  the *second* based on its size *relative* to the first (requires holding the
  first in CTRNN state → memory).
- **Studies:**
  - `relative-size-discrimination` — catch/avoid the second object by relative
    size, from the seeded genome.
  - `memory-dynamics` — how the first object's size is encoded in and read out of
    neural state over the inter-stimulus interval.
  - `re-evolution` *(opt-in)*.

### 4. ctrnn-parameter-space

- **Brain only** (no body/env composite). Analyze small CTRNNs directly.
- **Studies:**
  - `equilibria-and-nullclines` — equilibria + nullclines of 1- and 2-neuron
    center-crossing CTRNNs.
  - `codim1-bifurcations` — saddle-node / pitchfork structure as a bias/weight
    parameter varies (Beer 2006).
  - `codim2-structure` — codimension-2 organizing centers (Beer 2022), rendered
    as a parameter-plane map.

### 5. action-switching (capstone)

Agmon & Beer (2014). A food-foraging agent that autonomously switches between
approaching two resources to keep two internal nutrient levels above zero. The
only model with **internal metabolic state**, so the CTRNN's `sensory_input` is
the concatenation of *external* (chemosensor) and *internal* (nutrient-level)
readings assembled by the body.

- **Body (`chemotactic_forager`):** chemosensors on stalks + internal nutrient
  sensors + interneurons + 2 motor neurons driving 2 force effectors. Effector
  outputs combine as a force vector: direction ∝ (o_left − o_right), magnitude ∝
  (o_left + o_right), with friction (agent coasts to rest at zero force). Three
  **morphology variants** (per the paper's Fig. 2): (1) two chemosensor stalks +
  nutrient sensors, (2) one front stalk + nutrient sensors, (3) two stalks, no
  nutrient sensors. Morphology is a study variant.
- **Environment (`chemotaxis_resources`):** bounded 100×100 plane, two resources
  (radius 7) at fixed locations; chemical concentration flat inside a resource and
  decaying exponentially with distance outside. Two chemical signals engaging
  different chemosensors.
- **Metabolism (`metabolism.py`):** two nutrient levels in [0, 10]; a level rises
  at a fixed eating rate while the agent is inside the matching resource and drains
  at a constant metabolic rate everywhere; a nutrient reaching 0 ends the trial
  (death). Survival time (capped at 5000 steps) is the fitness, over the paper's
  battery of 11 environment configurations.
- **Studies:**
  - `foraging-and-survival` — the seeded agent sustains both nutrients and survives
    the trial battery.
  - `morphology-comparison` — the three morphologies' evolved strategies (the
    paper's high-level view of the strategy space); variant study.
  - `action-switching-dynamics` — analyze one evolved agent in depth: transient
    modes of sensorimotor coordination (which sensors/effectors engage/disengage),
    and the rapid transitions through state space that correspond to switches
    between the two foraging actions.
  - `re-evolution` *(opt-in)* — evolve a fresh forager via the generic
    `EvolutionStep` (survival-time fitness).

## Evolution

`EvolutionStep` is **generic: it wraps a whole BBE composite and evolves it.** It
takes (a) a composite spec that exposes an evolvable genome and (b) a fitness
function, and runs a genetic algorithm over the composite — no model-specific GA
code. It is model-agnostic; adding a fifth model requires no change to the GA.

- **Genome extraction/injection.** A BBE composite declares which parameters are
  evolvable — the CTRNN genome (`tau`, `theta`, `w`, sensor gains) and any body
  params a model opts to evolve — via a stable flat-vector encoding with published
  bounds. `EvolutionStep` reads the genome from a candidate composite, writes a
  mutated genome back into a fresh composite instance, runs it, and scores it. The
  encode/decode contract lives on the processes (e.g. a `genome` schema + helpers)
  so the Step never hard-codes a model's parameter layout.
- **Fitness.** Each model supplies a fitness function in `tasks/fitness.py` that
  instantiates the composite from a genome, runs it over a trial battery
  (offsets/shapes/object sequences), and returns a scalar score. The Step calls
  this callable; it owns no task logic itself.
- **Algorithm.** Mutation (Gaussian on the real-valued genome, clipped to bounds)
  + rank/truncation selection; microbial-GA-style is fine. Population size,
  generations, mutation variance and RNG seed are Step parameters.

Seeded genomes live in `data/genomes/` as committed arrays; studies load them so
runs are deterministic. Re-evolution studies are opt-in, fixed-seed where
possible, and emit a new genome artifact (they do not overwrite the seed).

## Dynamical-systems analysis

`analysis.py` provides reusable, numpy/scipy helpers (no plotting):

- `equilibria(ctrnn, I)` — fixed points via multi-start root finding.
- `jacobian(ctrnn, y)` / `eigenvalues(...)` — local stability.
- `nullclines(ctrnn, I, grid)` — for 1–2 neuron systems.
- `bifurcation_sweep(ctrnn, param, values)` — track equilibria/stability.
- trajectory + success metrics for the embodied tasks.

`viz.py` provides `@visualization`-decorated functions (built/extended via
`/viva-viz`) rendering: phase portraits with nullclines + fixed points, agent
trajectory & categorization maps, CPG limit cycles / gait diagrams, and
bifurcation / parameter-plane plots. Dashboard visuals stay AI-free per ecosystem
rule.

## Error handling

- CTRNN integration guards against NaN/overflow (clip activations; fail loud on
  non-finite state rather than emitting garbage).
- Sensor ray casting handles the no-intersection case explicitly (current = 0).
- Genome loaders validate shape/length against the target CTRNN size and fail
  with a clear message on mismatch.
- `analysis.equilibria` reports when root-finding does not converge rather than
  returning spurious points.

## Testing (TDD)

- **CTRNN:** integrate a 1-neuron system to its analytic equilibrium; reproduce a
  known 2-neuron oscillator's limit cycle; verify logistic/center-crossing
  conventions. Jacobian checked by finite differences.
- **Bodies:** sensor geometry (ray/object intersection) and effector mapping unit
  tested against hand-computed cases; no-intersection → 0 current.
- **Environments:** object motion + geometry queries deterministic given a seed.
- **Composites:** each loads, runs N steps via `/viva-run` / `composite-test-run`,
  emits expected observables.
- **Seeded genomes:** each asserts the paper's qualitative behavior above a
  threshold (e.g., mean catch-distance for circles ≪ for diamonds; forward
  velocity > 0 and periodic; relative-size discrimination > chance).
- **Evolution:** GA smoke test improves a trivial fitness within a few
  generations (fixed seed).

## Provenance & conventions

- References: Beer PDFs / citation metadata under `references/`; acceptance bands
  in studies cite the originating paper via `/viva-cite-bands`.
- Studies/investigations created through `/viva-study` and `/viva-investigation`
  (canonical `study.yaml`, provenance) — never hand-edited around the skills.
- Processes are explicitly labeled clean-room reproductions of Beer's published
  models.
- No AI attribution in commits/PRs.

## Execution outline (for the plan)

1. `/viva-workspace` (standalone) → scaffold `~/code/viva-bbe-systems` on a
   workspace branch; move this spec into the repo and commit.
2. `/viva-expert --reproduce` → `CTRNNProcess`, then per-model body/environment
   processes + composites, each with tests (TDD).
3. `EvolutionStep` + fitness + seeded genomes.
4. `/viva-investigation` × 4 and `/viva-study` for each study above.
5. `/viva-viz` for the dynamical-analysis visualizations.
6. `/viva-report` → publish read-only workbench + reports.

Each model is independently shippable; the plan sequences shared core first, then
models in the order: ctrnn-parameter-space → categorical-perception →
legged-locomotion → relational-categorization → action-switching
(parameter-space first because it is brain-only and validates the CTRNN + analysis
tooling the others depend on; categorical-perception next as the iconic BBE case;
action-switching last as the capstone, since it adds internal metabolic state and
the richest analysis on top of everything else).
