# Relational Categorization (Williams, Beer & Gasser 2008) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A CTRNN agent that categorizes the SECOND of two sequentially-presented objects by its size RELATIVE to the first — catching it if larger, avoiding it if smaller — which requires holding the first object's size in neural state across an inter-stimulus interval (memory). Built on the shared CTRNN + GA, reusing the active-categorical-perception body/object, with the behaviour video, the memory-dynamics analysis, a neural-activation GIF, evolution-progress viz, and a BBE composite.

**Architecture:** A horizontal catcher with ray distance-sensors (reuse `CategoricalBody` + `FallingObject`) sees two circles fall one after another. Phase 1: object 1 (size s1) falls — the agent perceives its size via the sensor shadow and must encode it. Inter-stimulus interval. Phase 2: object 2 (size s2) falls — the agent moves to CATCH it (final horizontal distance small) if s2 > s1, or AVOID it (move away) if s2 < s1. Because object 1 is gone during phase 2, s1 can only inform the decision via the CTRNN's internal state → relational MEMORY. Evolved for correct relative-size categorization by the shipped `evolve_flat`. Clean-room reproduction with documented constants.

**Tech Stack:** Python ≥3.11, numpy, the shipped `viva_bbe_systems` core (`ctrnn`, `genome`, `evolution`, `analysis`, `anim`, `viz`), the categorical `CategoricalBody`/`FallingObject`, matplotlib, pytest, process_bigraph (composite). Reuses the categorical model's agent/fitness/evolution/viz patterns (single-loop run_trial with optional recording; `evolve_flat(record_every=...)`; `anim.save_gif`; the `animate_neural` node-ring+raster) and the BBE composite pattern (`@composite_generator` + a body Process + an env Process + `CTRNNProcess`).

**Spec:** `docs/superpowers/specs/2026-10-03-viva-bbe-systems-design.md` (§ 3 relational-categorization)

## Global Constraints

- Reuse the shipped core UNCHANGED (no edits to `ctrnn.py`, `genome.py`, `evolution.py`, `core.py`), and reuse `CategoricalBody` + `FallingObject` unchanged (wrap/compose, don't edit).
- Relation rule (documented, fixed): CATCH object 2 iff `s2 > s1` (larger), AVOID iff `s2 < s1` (smaller). Equal sizes are excluded from the battery (ambiguous).
- Sizes in `[2.0, 6.0]`; battery pairs differ by a margin ≥ ~1.0 so the discrimination is well-posed; both s2>s1 and s2<s1 cases covered in balance.
- Determinism: every run/evolution takes an explicit seed; committed seed makes studies deterministic; non-finite CTRNN state fails loud.
- CTRNN size: a small net with a designated memory capacity — use size 6 (4 interneurons + 2 motor), motor_indices (−2,−1); the ray-sensor currents feed the first `n_sensors` neurons (zero-pad the rest), mirroring the categorical model's `sensor_weights @ body.sense(obj)` → input.
- Investigation + studies on `main` under `workspace/investigations/relational-categorization/`.
- EVERY study wires figures via the canonical `visualizations: [{name, chart: ../../figures/<file>, config:{caption}}]` block AND has a non-empty `baseline: [{name, composite|process, params}]`; baselines reference the registered `relational` composite (Task 6) or `process: CTRNNProcess` if the composite is deferred.
- GIFs + PNGs both render. No AI attribution in commits. Dashboard visuals AI-free.

## Review Focus

- **Memory actually used (not a within-phase-2 cue):** object 1 must be GONE during phase 2, so the only way to respond relative to s1 is internal state. A test must confirm the agent cannot see object 1 during phase 2 (the env returns no object / out-of-range during the ISI and phase 2 shows only object 2). (Task 1.)
- **Both relations covered + balanced:** the battery must contain both s2>s1 and s2<s1 pairs; an agent that always catches (or always avoids) scores ~50%, not passing. (Task 3.)
- **Degenerate "ignore s1" strategy scores chance:** fitness must reward RELATIVE correctness, so a fixed catch/avoid policy or a policy keyed only on s2 cannot score high. (Task 3.)
- **CTRNN divergence:** a bad genome scores minimally, never crashes the GA (FloatingPointError/OverflowError guard → 0). (Task 3.)
- **Catch vs avoid both well-defined:** "caught" = final |agent.x − obj2.x| ≤ catch_radius; "avoided" = final distance ≥ an avoid_margin. Both pinned. (Task 2/3.)

---

### Task 1: Two-object sequential environment (object stream)

**Files:** Create `viva_bbe_systems/environments/object_stream.py`; Test `tests/test_object_stream.py`.

**Interfaces:**
- `class TwoObjectStream(s1, s2, *, offset1=0.0, offset2=0.0, H=20.0, vy=1.0, phase1_steps=160, isi_steps=40, phase2_steps=160)`: manages two `FallingObject`s presented in sequence.
  - `visible(step) -> FallingObject | None`: object 1 while `0 <= step < phase1_steps` (falling from H), `None` during the ISI (`phase1_steps <= step < phase1_steps+isi_steps`), object 2 while phase 2, `None` after. Each object's `center[1]` falls from `H` at `vy*dt` within its phase (reset to H at the start of its phase).
  - `total_steps` property = phase1+isi+phase2.
  - `phase(step) -> str` ∈ {"obj1","isi","obj2","done"}.
  - `sizes` = (s1, s2).

- [ ] Step 1 (RED): tests — `visible` returns obj1 (size s1) in phase 1, `None` in the ISI, obj2 (size s2) in phase 2, `None` after; objects fall (center[1] decreases within a phase); during phase 2 object 1 is NOT returned (memory-forcing). `total_steps` correct.
- [ ] Steps 2-4 TDD. Step 5: commit `feat: two-object sequential environment (relational object stream)`.

---

### Task 2: Relational agent runner (two-phase trial + memory)

**Files:** Create `viva_bbe_systems/agents/relational_agent.py`; Test `tests/test_relational_agent.py`.

**Interfaces:**
- `class RelationalAgent(ctrnn, body, sensor_weights, *, motor_indices=(-2,-1), motor_gain=5.0, catch_radius=2.0, avoid_margin=6.0)`. `RELATIONAL_SIZE = 6`; `make_relational(ctrnn=None, n_sensors=7)` builds a default agent (CategoricalBody(n_sensors) + a size-6 CTRNN + sensor_weights of shape (6, n_sensors)).
- `run_trial(s1, s2, *, offset1=0.0, offset2=0.0, dt=0.1, record=False) -> dict`:
  - reset ctrnn (zeros) + body (`x=0`); build a `TwoObjectStream(s1, s2, offset1, offset2, ...)`.
  - single loop over `stream.total_steps`: `obj = stream.visible(t)`; `I = sensor_weights @ body.sense(obj)` when `obj is not None` else `sensor_weights @ zeros(n_sensors)` (no object → zero sensory shadow); `o = ctrnn.step(external_input=I)`; `motor = o[motor_indices]`; `body.act(motor, dt, motor_gain)`; advance stream timing. Record x, outputs, obj2 center when recording.
  - returns `{"caught": bool (final |x-obj2_x| ≤ catch_radius), "final_distance": float, "should_catch": bool (s2>s1), "correct": bool (caught == should_catch), "x_hist": (T,)}`; with `record=True` also `"outputs": (T, size)`, `"obj_x_hist"`, `"phase_hist"`.

- [ ] Step 1 (RED): tests — a trial returns the dict with finite final_distance + histories length total_steps; determinism (same ctrnn+args → identical x_hist + caught); `should_catch == (s2>s1)`; `caught` reflects final distance vs catch_radius (construct a body/ctrnn that parks under obj2 → caught True; one that parks far → caught False). record adds outputs (T,size).
- [ ] Steps 2-4 TDD. Step 5: commit `feat: relational agent runner (two-phase trial + memory)`.

---

### Task 3: Relational fitness (relative-size categorization)

**Files:** Create `viva_bbe_systems/tasks/relational_fitness.py`; Test `tests/test_relational_fitness.py`.

**Interfaces:**
- Module-level `SIZE_PAIRS`: a deterministic balanced battery of (s1, s2) pairs — both `s2>s1` (should-catch) and `s2<s1` (should-avoid), sizes in [2,6], |s2−s1| ≥ ~1.0, a handful of offsets. Provide `_make_pairs(seed=...)`.
- `relational_fitness(genome, spec, *, record=False) -> float`: decode a size-`RELATIONAL_SIZE` CTRNN; build `RelationalAgent`; run one trial per pair; score the mean RELATIVE correctness with CONTINUOUS shaping — for a should-catch pair reward `catch_closeness = scale/(scale+final_distance)`; for a should-avoid pair reward `avoid = min(final_distance, avoid_margin)/avoid_margin`. So catching-when-should-catch and fleeing-when-should-avoid both score ~1; the opposite ~0. GUARD FloatingPointError/OverflowError → 0. Deterministic.
- `make_relational_fitness(spec, **kw)`.

- [ ] Step 1 (RED): tests — fitness finite & deterministic; a divergent genome (tiny tau) scores ~0, no raise; a fixed "always park at 0" policy scores ~chance (can't beat relative task); battery has both relations balanced (count should_catch ≈ count should_avoid); |s2−s1| ≥ margin for all pairs.
- [ ] Steps 2-4 TDD. Step 5: commit `feat: relational fitness (relative-size categorization)`.

---

### Task 4: Evolve + commit the relational seed

**Files:** Create `viva_bbe_systems/tasks/evolve_relational.py`; Test `tests/test_evolve_relational.py` (fast smoke only).

**Interfaces:**
- `relational_spec()` → `GenomeSpec(RELATIONAL_SIZE, tau_range=(0.5,10), bias_range=(-16,16), weight_range=(-16,16))`; the genome also carries the sensor_weights (shape size×n_sensors) appended after the CTRNN block (mirror the categorical genome's layout — see `bodies/categorical_genome.py` / `evolve_categorical.py`).
- `evolve_relational(*, pop_size=80, generations=120, seed=0, mutation_sd=0.5, record_every=None)` → `evolve_flat(make_relational_fitness(spec), genome_length, lo, hi, ...)`.
- `save_seed`/`load_seed` → `data/genomes/relational.npz`; `save_checkpoints`/`load_checkpoints` → `data/genomes/relational_evolution.npz`. `per_pair_report(genome)` → accuracy + per-pair correctness.
- CLI evolves (full budget), prints accuracy (fraction correct) split by should-catch / should-avoid.

- [ ] Steps 1-4 (subagent): fast smoke (pop 8, gen 3) runs + improves; save/load round-trip; bounds tau≥0.5. Do NOT run the full evolution.
- [ ] Step 5 (controller): run the full evolution; aim for an agent that categorizes by RELATIVE size (accuracy ≫ chance, both relations). If it doesn't discriminate, tune (budget, ISI length, size margin, shaped fitness). Report the ACHIEVED accuracy honestly (overall + per-relation). Commit the seed + `test_relational_seed.py` pinning accuracy > <achieved threshold> AND that it beats the always-catch / always-avoid baselines (proving relational memory).
- [ ] Step 6: commit.

---

### Task 5: Investigation + studies + full visualization suite

**Files:** Create `viva_bbe_systems/relational_anim.py`, `viva_bbe_systems/relational_gallery.py`, `viva_bbe_systems/relational_evo_viz.py`; `workspace/investigations/relational-categorization/investigation.yaml` + 2-3 studies; figures; `tests/test_relational_viz.py`.

Visualizations per study (wired as `visualizations:`):
- **Video (agent solving):** `relational.gif` — the two objects falling in sequence (obj1 then, after the ISI, obj2), the catcher moving, and the final CATCH (object 2 larger) vs AVOID (object 2 smaller) outcome, shown for one should-catch and one should-avoid trial. Use `anim.save_gif`.
- **Memory dynamics (THE payoff):** `memory_dynamics.png` — how object 1's size is encoded in and read out of neural state. Plot, for a sweep of s1 values, the CTRNN state (one or two neurons) during the ISI (object gone) vs s1 — showing a monotonic/separable encoding that persists until object 2 arrives. PLUS a decision plot: final response (catch/avoid) over the (s1, s2) plane (the relational decision boundary ≈ the diagonal s2=s1).
- **Neural activity:** `neural_activity.gif` — the CTRNN node-ring graph + time raster across the two phases + ISI (reuse the `animate_neural` pattern), showing the memory trace holding over the ISI.
- **Evolution-progress:** `evolution_curves.png` (accuracy vs generation) + `evolution_progress.gif` (the decision boundary sharpening across generations).

Studies: `relative-size-discrimination` (catch larger / avoid smaller; achieved accuracy + per-relation), `memory-dynamics` (s1 encoded in state over the ISI; the decision boundary ≈ diagonal). Investigation v2 spine (executive/scientific_argument/biological_story, competing_frameworks, glossary), honest reporting. Each study: non-empty baseline + `visualizations:`. `/viva-report` render. Commit.

---

### Task 6: BBE composite (register `relational`)

**Files:** Create `viva_bbe_systems/processes/relational_body_process.py` (+ object-stream Process); register in `viva_bbe_systems/composites/__init__.py`; Test in `tests/test_bbe_composites.py` (add a relational case).

- `RelationalBodyProcess(Process)` wraps `CategoricalBody` + the sensor_weights: reads `motor_output`, outputs `sensory_input` (zero-padded ray-sensor currents for the currently-visible object) + readouts (`agent_x`, `phase`). A `TwoObjectStreamProcess(Process)` (or bake the stream into the body process config, like the categorical composite bakes the falling object) presents obj1→ISI→obj2 and outputs the visible object's geometry. `build_relational_composite(s1=.., s2=.., seed_path=None)` wires env→body→`CTRNNProcess` through stores (one-step delay), from the committed seed. `@composite_generator(name="relational", ...)` → `viva_bbe_systems.composites.relational`. Repoint the relational studies' baselines to it.
- Engine-run test (mirror `tests/test_bbe_composites.py`): builds via `Composite({"state": build_relational_composite()}, core=build_core())`, runs the full `total_steps`, no error, agent_x changes, phase advances obj1→isi→obj2.
- Commit `feat: relational BBE composite (registered) + study baselines`.

---

## Self-review notes (author)
- Coverage: env (T1), agent (T2), fitness (T3), seed (T4), studies+viz (T5), composite (T6) — the spec's relational-categorization bullets (relative-size-discrimination, memory-dynamics, re-evolution-noted) plus a registered composite, reusing the shared core + categorical body/object unchanged.
- Memory is structurally forced: object 1 is absent during phase 2 (T1 test pins it), so relative response ⇒ internal memory. The memory-dynamics study makes the held trace visible.
- Open risk (honest): evolving robust relative-size discrimination is non-trivial; the plan budgets for it and requires reporting ACHIEVED accuracy (overall + per-relation) and beating always-catch/always-avoid baselines, rather than asserting a pre-decided number.
