# Legged Locomotion CPG (Beer & Gallagher 1992) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CTRNN central pattern generator driving a single leg to walk — the classic rhythmic brain-body model — built on the shared CTRNN brain + GA, with the walk video, the CPG limit-cycle / phase-portrait analysis, a gait diagram, and evolution-progress visualizations.

**Architecture:** A brain-body loop where the shared `CTRNN` drives a `LeggedBody` (one leg: angle + foot up/down; three motor outputs — foot, backward-swing, forward-swing) over flat ground. When the foot is planted and the leg swings backward, the body translates forward; when the foot is up, the leg swings freely (recovery). The leg angle is fed back to the CTRNN as proprioception, so the CPG can be sensory-entrained. Agents are evolved for forward distance by the shipped `evolve_flat`. Clean-room reproduction with documented constants (the spec sanctions reproduce-mode).

**Tech Stack:** Python ≥3.11, numpy, scipy, the shipped `viva_bbe_systems` core (`ctrnn`, `genome`, `evolution`, `analysis`, `anim`, `viz`), matplotlib, pytest. Reuses the forager model's agent/fitness/evolution/viz patterns (single-loop `run_trial` with optional recording; `evolve_flat(record_every=...)` checkpoints; `anim.save_gif`).

**Spec:** `docs/superpowers/specs/2026-10-03-viva-bbe-systems-design.md` (§ legged-locomotion)

## Global Constraints

- Reuse the shipped core UNCHANGED (no edits to `ctrnn.py`, `genome.py`, `evolution.py`, `core.py`).
- Leg model (clean-room, documented constants matching the Beer walker's structure): leg angle `θ ∈ [−π/6, +π/6]` (±30°), `leg_length = 15.0`, foot down iff `o_foot > 0.5`, angular-velocity control `ω = omega_gain·(o_bs − o_fs)` with `omega_gain = 1.0`, `dt = 0.1`. When the foot is DOWN the planted foot translates the body: `Δx_body = −Δθ · leg_length` (leg swings back → body forward), with `θ` clamped to its range (no body motion past the limit). When the foot is UP the leg swings freely and the body coasts with friction `0.9`. The leg angle is fed back to the CTRNN (one proprioceptive input, normalized to [−1,1]).
- Determinism: every run/evolution takes an explicit seed; the committed seed makes studies deterministic; non-finite state fails loud (inherited from the CTRNN).
- Investigation + studies on `main` under `workspace/investigations/legged-locomotion/`.
- EVERY study wires its figures via the canonical `visualizations: [{name, chart: ../../figures/<file>, config:{caption}}]` block (NOT a `figures:` key), AND has a non-empty `baseline: [{name, composite|step|process, params}]` (the workbench render-guarantee rejects `baseline: []`); baselines reference `process: CTRNNProcess` (the shared brain) unless the optional composite is built.
- GIFs + PNGs both render. No AI attribution in commits. Dashboard visuals AI-free.

## Review Focus

- **Leg at the angle limit while planted:** when `θ` is clamped at ±π/6 and the foot stays down, the body must stop advancing (no motion past the limit), not jump. (Task 1.)
- **Foot lifts mid-swing:** switching foot down→up must stop body translation cleanly and begin free swing without a discontinuity in `x`. (Task 1.)
- **Agent that never lifts its foot / never swings:** distance ≈ 0; fitness finite; no divide-by-zero. (Task 3.)
- **CTRNN divergence during evolution:** a bad genome scores minimal distance, never crashes the GA (fitness guard). (Task 3.)
- **Backward walking:** an agent that moves the body backward should score ≤ a stationary one (fitness rewards forward distance only, or net signed distance — pick one and pin it). (Task 3.)

---

### Task 1: Legged body (leg + foot + body dynamics)

**Files:** Create `viva_bbe_systems/bodies/legged.py`; Test `tests/test_legged_body.py`.

**Interfaces:**
- `class LeggedBody(angle_range=(-pi/6, pi/6), leg_length=15.0, omega_gain=1.0, friction=0.9, foot_threshold=0.5)`:
  - state: `angle` (θ), `foot_down` (bool), `x` (body position), `vx` (body velocity), `omega` (ω).
  - `sense() -> float`: the leg angle normalized to [−1, 1] over its range (proprioceptive feedback).
  - `act(o_foot, o_bs, o_fs, dt) -> None`: set `foot_down = o_foot > foot_threshold`; `omega = omega_gain*(o_bs − o_fs)`; `new = angle + omega*dt`; if foot down → clamp `new` to the range, `Δθ = new − angle`, `x += −Δθ*leg_length`, `vx = −Δθ*leg_length/dt`, `angle = new`; else (foot up) → `angle = clamp(new)`, `vx *= friction`, `x += vx*dt`.
  - `reset(angle=0.0)`.

- [ ] Step 1 (RED): tests — foot DOWN + swing backward (o_bs>o_fs → +ω → +Δθ) moves the body forward (`x` increases); foot UP → the leg swings (angle changes) but the body only coasts (no drive), velocity decays by 0.9; angle clamps at ±π/6 and the body stops advancing while planted at the limit; `sense()` maps angle to [−1,1].
- [ ] Steps 2-4 TDD (verify the body-translation sign + clamp). Step 5: commit `feat: legged body (leg + foot + body dynamics)`.

---

### Task 2: Walker agent runner (CPG brain + leg loop)

**Files:** Create `viva_bbe_systems/agents/walker_agent.py`; Test `tests/test_walker_agent.py`.

**Interfaces:**
- `class WalkerAgent(ctrnn, body, *, foot_index=-3, bs_index=-2, fs_index=-1, angle_input_index=0)`.
- `run_trial(*, start_angle=0.0, dt=0.1, steps=500, record=False) -> dict`:
  - reset ctrnn (zeros) + body (`x=0`, `angle=start_angle`, `vx=0`).
  - each step (single loop): build `I_full` length `ctrnn.size`, all 0 except `I_full[angle_input_index] = body.sense()`; `o = ctrnn.step(I_full)`; `body.act(o[foot_index], o[bs_index], o[fs_index], dt)`.
  - returns `{"distance": float (final body.x), "x_hist": (T,), "angle_hist": (T,), "foot_hist": (T,) bool, "vx_hist": (T,)}`; with `record=True` also `"outputs": (T, ctrnn.size)`. ONE loop, optional recording (no duplicate loops — the categorical/forager models' reviews flagged duplication). Record copies where arrays are stored.
- Module-level `WALKER_SIZE` (CTRNN size) = `n_inter + 3 motor` with the angle fed to neuron 0; choose a small net, e.g. 5 (2 inter + 3 motor) → size 5; motor indices (−3,−2,−1) = foot/BS/FS; angle input index 0. Provide `make_walker(ctrnn=None)`.

- [ ] Step 1 (RED): tests — a trial returns finite `distance` and histories of shape (steps,); a hand-built CPG (e.g. a strong 2-neuron oscillator wired to drive foot + swing in antiphase) produces net forward motion (`distance > 0`) OR at least nonzero motion; determinism (same ctrnn+body+args → identical distance + x_hist); `record=True` adds outputs (steps, size).
- [ ] Steps 2-4 TDD. Step 5: commit `feat: walker agent runner (CPG brain + leg loop)`.

---

### Task 3: Walk fitness (forward distance)

**Files:** Create `viva_bbe_systems/tasks/walk_fitness.py`; Test `tests/test_walk_fitness.py`.

**Interfaces:**
- `walk_fitness(genome, spec, *, steps=500, dt=0.1) -> float`: decode a CTRNN of size `WALKER_SIZE` from `genome`; build `WalkerAgent`; run one trial; return the forward distance traveled (`body.x`, i.e. net signed distance so backward walking scores ≤ a stationary agent). GUARD `FloatingPointError`/`OverflowError` → score 0. Deterministic. (A single environment — flat ground — so no config battery is needed; optionally average over a few start angles for robustness.)
- `make_walk_fitness(spec, **kw)`.

- [ ] Step 1 (RED): tests — fitness finite & deterministic; a divergent genome (tiny tau) scores ~0, no raise; a stationary agent (zero motor drive) scores ~0; a known forward-walking hand-built genome (or best-of-a-few-random) scores > 0 (positive distance). Backward motion scores ≤ stationary.
- [ ] Steps 2-4 TDD. Step 5: commit `feat: walk fitness (forward distance)`.

---

### Task 4: Evolve + commit the walker seed

**Files:** Create `viva_bbe_systems/tasks/evolve_walker.py`; Test `tests/test_evolve_walker.py` (fast smoke only).

**Interfaces:**
- `walker_bounds()` → length-`genome_length(GenomeSpec(WALKER_SIZE, tau_range=(0.5,10), bias_range=(-16,16), weight_range=(-16,16)))` lo/hi (tau ≥ 0.5 so the CTRNN can't diverge).
- `evolve_walker(*, pop_size=60, generations=80, seed=0, mutation_sd=0.5, record_every=None, steps=500)` → `evolve_flat(make_walk_fitness(spec, steps=steps), genome_length(spec), lo, hi, ...)`. (`evolve_flat` UNCHANGED.)
- `save_seed`/`load_seed` → `data/genomes/legged_walker.npz`; `save_checkpoints`/`load_checkpoints` → `data/genomes/legged_walker_evolution.npz`.
- CLI evolves (full budget), prints distance + stride period + duty factor.

- [ ] Steps 1-4 (subagent): fast smoke test (pop 8, gen 3, small steps) runs + improves; save/load round-trip; `walker_bounds` tau≥0.5. Do NOT run the full evolution.
- [ ] Step 5 (controller): run the full evolution; a walker should evolve (forward distance ≫ 0 with a rhythmic gait). If it doesn't walk, tune (budget, omega_gain, leg_length, or a shaped fitness rewarding rhythmic foot cycling). Report the ACHIEVED distance + gait honestly. Commit the seed + a `test_walker_seed.py` pinning `distance > <achieved threshold>` and a non-trivial gait (foot cycles up and down).
- [ ] Step 6: commit.

---

### Task 5: Investigation + studies + full visualization suite

**Files:** Create `viva_bbe_systems/walker_anim.py`, `viva_bbe_systems/walker_gallery.py`, `viva_bbe_systems/walker_evo_viz.py`; `workspace/investigations/legged-locomotion/investigation.yaml` + 3 studies; figures; `tests/test_walker_viz.py`.

The three required visualizations per study (as established for the other models, wired as `visualizations:`):
- **Video (agent solving):** `walk.gif` — a side view of the leg: the body (a box) advancing, the leg pivoting at the hip, the foot planting (filled) / lifting (open) over the ground, with the body position advancing — the walk in motion. Use `anim.save_gif`.
- **Dynamical analysis (CPG):** the CPG limit cycle — a phase portrait of the CTRNN state (project onto two neurons) showing the rhythmic limit cycle (reuse `viz.phase_portrait_2d`/`analysis` where applicable, or plot the recorded neuron-output trajectory), PLUS a **gait diagram**: foot up/down bands + leg angle + body velocity over time (showing the stance/swing rhythm and stride period / duty factor).
- **Neural activity (CPG firing):** `neural_activity.gif` — the CTRNN node-ring graph (neuron outputs + weight edges) + a time raster, showing the central pattern generator's rhythm driving the leg (reuse the forager's `animate_neural` pattern).
- **Evolution-progress:** `evolution_curves.png` (distance vs generation) + `evolution_progress.gif` (the gait improving — e.g. the leg-angle/foot trace or the walk at checkpoint generations), from `evolve_walker(record_every=...)` checkpoints whose final genome == the committed seed (pin with a test).

Studies: `walk-forward` (the agent walks forward; achieved distance + gait), `cpg-limit-cycle` (the CPG's rhythmic dynamics / limit cycle), `gait-and-velocity` (stride period, duty factor, mean velocity). Investigation v2 spine (executive/scientific_argument/biological_story, competing_frameworks, glossary), honest reporting of the achieved gait. Each study: non-empty baseline + `visualizations:` block. `/viva-report` render. Commit.

---

## Self-review notes (author)
- Coverage: leg body (T1), walker agent (T2), fitness (T3), evolved seed (T4), studies+full viz (T5) — every spec bullet for legged-locomotion, reusing the shared core unchanged.
- Reuse: the shared `CTRNN`/`GenomeSpec`/`evolve_flat`/`anim`/`analysis`/`viz` are used unchanged (no extended genome — proprioception feeds a designated neuron directly, like the forager). The single-loop `run_trial` + optional record and the viz pattern mirror the forager model, avoiding the duplicated-loop issue prior reviews flagged.
- Open risk (honest): evolving a clean forward walker is a classic but non-trivial GA; the plan budgets for it and requires reporting the ACHIEVED gait (distance, stride period, duty factor) rather than asserting a pre-decided number. The leg constants are documented clean-room choices; the CPG limit-cycle + gait analysis are the scientific payoff regardless of walk speed.
```
