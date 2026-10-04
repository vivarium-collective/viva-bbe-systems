# Action Switching in Embodied Agents (Agmon & Beer 2014) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The capstone BBE model: a chemotactic foraging agent with *internal metabolic state* that autonomously switches between approaching two resources to keep two nutrient levels alive, built on the shared CTRNN brain + GA, with videos, dynamical (action-switching) analyses, and evolution-progress visualizations.

**Architecture:** A brain–body–environment loop where the shared `CTRNN` brain receives external (chemosensor) and internal (nutrient-level) currents assembled by a `ChemotacticForager` body, drives two effectors as torque+thrust through a friction model, moves in a bounded 2-D plane with two chemical-gradient resources, and is kept alive by a `Metabolism` that eats inside resources and drains everywhere. Agents are evolved for longevity by the shipped `evolve_flat`. Three morphologies (sensor placement / presence of nutrient sensors) are a config. All numeric constants are from Agmon & Beer (2014) Appendix.

**Tech Stack:** Python ≥3.11, numpy, scipy, the shipped `viva_bbe_systems` core (`ctrnn`, `genome`, `evolution`, `anim`, `viz`, `analysis`), matplotlib, pytest. Reuses the viz pattern established for categorical-perception (agent videos via `anim.save_gif`, evolution-progress via `evolve_flat(record_every=...)` checkpoints).

**Spec:** `docs/superpowers/specs/2026-10-03-viva-bbe-systems-design.md` (§ action-switching) + the committed paper `references/AgmonBeer2014.pdf`.

## Global Constants (Agmon & Beer 2014 Appendix — use verbatim)

- Plane: bounded 100×100. Two resources, radius 7, fixed per trial. Δt (StepSize) = 1.
- Chemical field of a resource at distance `d` from its center: `conc = 10·exp(−0.05·d)`, **capped at 10 inside the radius** (flat max 10 within boundary). Each resource emits its own signal (A, B).
- Chemosensors project `6` units out from the body center; projection direction is morphology-specific (relative to heading). A chemosensor is tuned to one signal (A or B) and reads that resource's `conc` at the sensor location.
- Nutrients: two levels in [0,10]; `+0.02`/step inside the matching resource boundary; `−0.0045`/step always (constant metabolism). A level reaching 0 → death (trial ends). Trial halts at 5000 steps if alive.
- Motion: `MaxAngle=π/12`, `MaxThrust=0.008`; `torque=(o_right−o_left)·MaxAngle`; `thrust=(o_right+o_left)·MaxThrust`; `velocity_t = 0.9·velocity_{t−1} + Δt·thrust`; `angle_t = angle_{t−1} + Δt·torque`; position advances by `velocity` along `angle`. Clamp position to the plane.
- Brain: shared `CTRNN` (Eq 1). Chemosensor neurons get `I=conc`; nutrient-sensor neurons get `I=nutrient level`; all other neurons `I=0`. Genome = CTRNN `(τ,θ,w)` via the shared `GenomeSpec` with `tau_range=(1,10)`, `bias_range=(-15,15)`, `weight_range=(-15,15)`. GA: longevity fitness over 11 configs; mutation variance 0.25.
- EVERY study wires its figures via the canonical `visualizations: [{name, chart: ../../figures/<file>, config:{caption}}]` block (NOT a `figures:` key — the workbench ignores it); GIFs + PNGs both render. AND a non-empty `baseline: [{name, composite|step|process, params}]` (the workbench study-detail render-guarantee rejects `baseline: []`). Action-switching studies reference the forager BBE composite (T7) or `process: ForagerAgent`/`CTRNNProcess`.
- Reuse the shipped core unchanged (no edits to `ctrnn.py`, `genome.py`, `evolution.py`, `core.py`). No AI attribution in commits. Investigation + studies on `main` under `workspace/investigations/action-switching/`.

## Review Focus

- **Nutrient hits exactly 0 at the boundary step:** death must trigger (≤0), the trial end deterministically, and survival time recorded. (Task 2.)
- **Agent pinned against the plane wall:** position clamps without NaN; chemosensors beyond the wall still read a finite conc. (Task 3.)
- **Both resources out of chemosensor range:** all chemo currents ~0; the agent still integrates and the nutrient drain still runs (so it can starve). (Task 4.)
- **CTRNN divergence during evolution:** a bad genome that diverges scores minimal longevity, never crashes the GA (fitness guard). (Task 5.)
- **Morphology without nutrient sensors (M3):** the internal state is absent from `I`; the body must still assemble a correct-length current vector. (Task 3/4.)

---

### Task 1: Chemotaxis environment (two gradient resources)

**Files:** Create `viva_bbe_systems/environments/chemotaxis_resources.py`; Test `tests/test_chemotaxis_env.py`.

**Interfaces:**
- `Resource(center: (2,), radius=7.0, signal: str)`.
- `concentration(resource, point) -> float` = `10.0` if `dist ≤ radius` else `10·exp(-0.05·dist)`.
- `class ChemotaxisEnv(resource_a, resource_b, width=100.0, height=100.0)`: `conc_at(point, signal) -> float` (the matching resource's concentration); `inside(point, signal) -> bool`; `clamp(point) -> (2,)`.

- [ ] Step 1 (RED): tests — concentration is 10 at center and within radius; decays as `10·exp(-0.05·d)` outside (e.g. d=20 → 10·exp(-1)=3.679); `conc_at` picks the signal's resource; `clamp` keeps points in [0,width]×[0,height].
- [ ] Step 2: run → FAIL. Step 3: implement. Step 4: PASS. Step 5: commit `feat: chemotaxis environment (two gradient resources)`.

---

### Task 2: Metabolism (internal nutrient state)

**Files:** Create `viva_bbe_systems/metabolism.py`; Test `tests/test_metabolism.py`.

**Interfaces:**
- `class Metabolism(levels=(nutrient_a, nutrient_b), eat_rate=0.02, drain_rate=0.0045, cap=10.0)`:
  - `step(inside_a: bool, inside_b: bool) -> None`: each level `+= eat_rate` if inside its resource, `-= drain_rate` always, clipped to `[0, cap]`.
  - `alive -> bool` (both levels `> 0`).
  - `levels -> np.ndarray (2,)`.

- [ ] Step 1 (RED): tests — inside A raises nutrient A net `+0.02-0.0045`; outside both drains both by 0.0045; a level reaching 0 makes `alive` False (verify death triggers when a level hits ≤0); cap at 10.
- [ ] Step 2–4 TDD. Step 5: commit `feat: metabolism (two nutrient levels, eat/drain/death)`.

---

### Task 3: Chemotactic forager body (sensors + effectors + motion), 3 morphologies

**Files:** Create `viva_bbe_systems/bodies/chemotactic_forager.py`; Test `tests/test_forager_body.py`.

**Interfaces:**
- `MORPHOLOGIES = {"M1","M2","M3"}`; a morphology defines chemosensor offsets (angles relative to heading, each tuned to signal A or B) and whether nutrient sensors are present:
  - `M1`: two side stalks at `±π/2` from heading, each carrying an A and a B chemosensor (4 chemo) + 2 nutrient sensors.
  - `M2`: one front stalk at `0`, carrying an A and a B chemosensor (2 chemo) + 2 nutrient sensors.
  - `M3`: same chemosensors as M1 (4), **no** nutrient sensors.
- `class ChemotacticForager(morphology="M2", sensor_dist=6.0, max_angle=pi/12, max_thrust=0.008, friction=0.9)`:
  - state: `pos (2,)`, `angle`, `velocity` (scalar).
  - `sensor_layout() -> list[(angle_offset, signal)]` and `n_chemo`, `n_nutrient`, `n_sensors`.
  - `sense(env, metabolism) -> np.ndarray` length `n_sensors`: chemosensor currents (conc at each sensor's world location = `pos + sensor_dist·unit(angle+offset)`, via `env.conc_at`) followed by nutrient currents (`metabolism.levels`, omitted for M3).
  - `act(o_right, o_left, env) -> None`: compute torque/thrust per the Global Constants, integrate `velocity` (friction), `angle`, advance `pos` along heading, `env.clamp` the position.
  - `inside(env) -> (bool, bool)` for resources A, B at `pos`.

- [ ] Step 1 (RED): tests — M1 has 6 sensors, M2 has 4, M3 has 4 (no nutrient); a chemosensor pointed at a nearby resource reads a higher conc than one pointed away; `act` with `o_right>o_left` turns one way and with equal outputs drives straight; friction decays velocity toward 0 at zero thrust; position clamps at the wall.
- [ ] Step 2–4 TDD (verify the motion math against the pseudocode). Step 5: commit `feat: chemotactic forager body (3 morphologies, torque/thrust motion)`.

---

### Task 4: Forager agent runner (brain+body+env+metabolism loop)

**Files:** Create `viva_bbe_systems/agents/forager_agent.py`; Test `tests/test_forager_agent.py`.

**Interfaces:**
- `class ForagerAgent(ctrnn, body, *, motor_right_index=-1, motor_left_index=-2)`.
- `run_trial(env, init_levels, *, start_pos, start_angle=0.0, max_steps=5000, record=False) -> dict`:
  - resets ctrnn (zeros) + body (pos/angle/velocity) + a fresh `Metabolism(init_levels)`.
  - loop each step: `I = concat(body.sense(env, metab))` placed into the CTRNN's sensor-neuron inputs (sensor neurons are the first `n_sensors`; rest 0); `o = ctrnn.step(I_full)`; `body.act(o[mr], o[ml], env)`; `inside_a, inside_b = body.inside(env)`; `metab.step(inside_a, inside_b)`; break when `not metab.alive` or `max_steps`.
  - returns `{"survival": int, "alive_at_end": bool, "path": (T,2), "levels_hist": (T,2)}`; with `record=True` also `"outputs": (T,N)` (reuse the record pattern from the categorical agent — a single loop, optional recorder, no duplicate loops).
- The full `I` vector has length `ctrnn.size`; sensor currents fill indices `0..n_sensors-1`, the rest are 0. `ctrnn.size` must equal `n_sensors + n_inter + 2` for the chosen morphology (a module constant maps morphology → size).

- [ ] Step 1 (RED): tests — a trial returns finite survival ≤ max_steps; an agent placed far from both resources with low nutrients dies (survival < max_steps); `record=True` adds outputs of shape (survival, N); determinism (same inputs → same survival/path).
- [ ] Step 2–4 TDD. Step 5: commit `feat: forager agent runner (BBE + metabolism loop)`.

---

### Task 5: Longevity fitness over 11 configs

**Files:** Create `viva_bbe_systems/tasks/forager_fitness.py`; Test `tests/test_forager_fitness.py`.

**Interfaces:**
- `TRIAL_CONFIGS` — 11 deterministic configs (resource A/B positions sampling near/far separations, agent start pos/angle, initial nutrient levels making the agent "hungrier" for one or the other), per the paper's intent (sample a variety of scenarios).
- `longevity_fitness(genome, spec, morphology="M2", *, max_steps=5000) -> float`: decode CTRNN from `genome`, build the forager, run all 11 configs, return **mean survival** (normalized by `max_steps` → [0,1]). Guard `FloatingPointError` → that config scores 0 survival (never crash). Deterministic.
- `make_forager_fitness(spec, morphology="M2", **kw)`.

- [ ] Step 1 (RED): tests — fitness finite & deterministic; an immortal-ish hand-set agent (or the best of a few random) scores higher than an agent that never moves (which starves at ~`min(init_level)/drain` steps); divergent genome scores low, no raise.
- [ ] Step 2–4 TDD. Step 5: commit `feat: longevity fitness over 11 forager configs`.

---

### Task 6: Evolve + commit the seeded forager (morphology M2)

**Files:** Create `viva_bbe_systems/tasks/evolve_forager.py`; Test `tests/test_evolve_forager.py` (fast smoke only).

**Interfaces:**
- `forager_bounds(spec)` — τ∈[1,10], θ/w∈[−15,15] arrays of length `genome_length(spec)`.
- `evolve_forager(*, morphology="M2", pop_size=80, generations=120, seed=0, mutation_sd=0.5, record_every=None)` → `evolve_flat` with `make_forager_fitness`. (`evolve_flat` already exists; **do not modify it**.)
- `save_seed/load_seed` → `data/genomes/action_switching_M2.npz`; `save_checkpoints/load_checkpoints` → `data/genomes/action_switching_M2_evolution.npz`.
- CLI evolves (full budget) and prints mean survival + per-config survival + whether the agent sustains both nutrients.

- [ ] Steps 1–4 (subagent): fast smoke test (`pop_size=8, generations=3`) asserts the GA runs + improves; save/load round-trip; **do not run full evolution**.
- [ ] Step 5 (controller): run the CLI (full budget). The success criterion is **survival far above the passive-starvation baseline, with evidence of switching** (the agent visits both resources). Tune if needed — this is a harder GA than catch/avoid: longer evolution, best-of-seeds, or a shaped longevity+balance fitness (reward keeping the *minimum* nutrient high) are allowed; **report the achieved survival honestly** (full 5000-step survival may or may not be reached). Commit the seed `.npz` + its achieved metrics as the study band.
- [ ] Step 6: commit.

---

### Task 7: Process-bigraph composite (optional dashboard BBE)

Mirror the categorical-perception composite: `ChemotaxisEnvProcess` + `ChemotacticForagerProcess` + `CTRNNProcess`, `overwrite[...]` ports, float-coded where needed, seed-loaded builder + a genuine behavioral test (the forager moves toward a resource and nutrient rises). One-step delay acceptable + documented. Create processes + a `.composite.yaml` builder reference + `tests/test_forager_composite.py`.

---

### Task 8: Investigation + studies + full visualization suite

**Files:** Create `viva_bbe_systems/forager_gallery.py`, `viva_bbe_systems/forager_anim.py`, `viva_bbe_systems/forager_evo_viz.py`; `workspace/investigations/action-switching/investigation.yaml` + 3 studies; figures; `tests/test_forager_viz.py`.

The three required visualizations per study (as established for categorical-perception):
- **Video (agent solving):** `forage.gif` — the forager navigating the 2-D plane (both resources drawn as chemical-gradient discs, the agent with its chemosensor stalks, its path), with the two nutrient levels shown as live bars/traces, over a full trial — showing it approach one resource, eat, then **switch** to the other. Use `anim.save_gif`.
- **Dynamical analysis (action switching):** reproduce the paper's reading — plot the agent's state-space trajectory and/or the **distance from sensorimotor-coordination modes over time** (the paper's Fig 12 idea: segments where different sensor/effector subsets are engaged), and a nutrient-vs-nutrient phase plot showing the switching cycle. Static figure + an animated state-trajectory GIF.
- **Evolution-progress:** `evolution_curves.png` (mean survival vs generation) + `evolution_progress.gif` (best agent's trial improving — e.g. survival / path at checkpoint generations), from `evolve_forager(record_every=...)` checkpoints whose final genome == the committed seed (pin with a test).

Studies: `forage-and-survival` (the agent sustains both nutrients / achieved survival), `morphology-comparison` (M1/M2/M3 evolved strategies — variant study; M3 lacks internal sensors so must internalize nutrient state), `action-switching-dynamics` (the deep single-agent dynamical analysis, morphology M2). Investigation v2 spine (executive/scientific_argument/biological_story, competing_frameworks, glossary), honest reporting of achieved survival. `/viva-report` render. Commit.

---

## Self-review notes (author)
- Coverage: env (T1), metabolism (T2), body+motion+morphologies (T3), agent loop (T4), fitness (T5), evolved seed (T6), composite (T7), studies+full viz (T8) — every spec bullet for action-switching is covered, with exact paper constants.
- Reuse: the shared `CTRNN`/`GenomeSpec`/`evolve_flat`/`anim`/`analysis` are used unchanged (this model needs **no** extended genome — sensors feed designated neurons directly, unlike categorical-perception). The viz pattern (videos, evolution checkpoints, dynamical analyses) is reused from the categorical model.
- Open risk (honest): evolving a long-surviving *switching* forager is the hardest GA in the repo. The plan budgets for it and requires reporting the **achieved** survival + switching evidence rather than asserting full 5000-step survival; morphology M2 (the paper's deep-analysis target) is the primary seed.
```
