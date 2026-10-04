# BBE composites: walker, forager, ctrnn — report

## Status: DONE_WITH_CONCERNS

Three real process-bigraph BBE composites built, registered via
`@composite_generator`, and the 9 study baselines repointed to them. Full suite
green; all 9 repointed baselines resolve AND build against the registry (the
"composite not found in registry" regression does NOT reappear).

Worked in a dedicated worktree: `~/code/viva-bbe-systems--bbe-composites`
(branch `bbe-composites`, off local `main` ed3963f — repo has no remote). Tests
run with `PYTHONPATH=<worktree>` prepended because the `.venv` editable install
points at the canonical checkout.

## What was built

- `viva_bbe_systems/processes/walker_body_process.py` — `WalkerBodyProcess`
  (wraps `LeggedBody` unchanged) + `build_walker_composite()`.
- `viva_bbe_systems/processes/forager_body_process.py` — `ForagerBodyProcess`
  (wraps `ChemotacticForager` + `ChemotaxisEnv`/`Resource` + `Metabolism`
  unchanged, env+metabolism baked in) + `build_forager_composite()`.
- `viva_bbe_systems/processes/ctrnn_process.py` — ADDED `build_ctrnn_composite()`
  (CTRNNProcess class untouched).
- `viva_bbe_systems/processes/__init__.py` — exposed the two new Process classes
  so `build_core()` auto-discovers them.
- `viva_bbe_systems/composites/__init__.py` — 3 new `@composite_generator`s:
  `walker`, `forager`, `ctrnn_parameter_space` (clean ids
  `viva_bbe_systems.composites.{walker,forager,ctrnn_parameter_space}`).
- `tests/test_bbe_composites.py` — engine-run smoke tests (mirror
  `test_categorical_composite.py`): discovery, registration, resolve+build, and
  behaviour.
- 9 study.yaml baselines repointed from `process: CTRNNProcess` to
  `composite: viva_bbe_systems.composites.<name>` (keeping a non-empty baseline).

## Engine-run readouts (`Composite({"state": spec}, core=build_core())`)

- **walker** (80 steps @ dt=0.1): `body_x` 0.0 -> 27.48 (mid) -> **62.51** final,
  monotone increasing (> 50). The committed seed walks.
- **forager** (60 steps): ran without error; `agent_pos` moved
  [50, 50] -> **[56.82, 46.44]**; `nutrient_levels` = **[4.73, 4.73]** (length-2
  vector), `alive` = 1.0.
- **ctrnn** (bare, size 5, 30–60 steps): `neuron_outputs` length-5 and **finite**
  ([30, 30, 30, 30, 30] — the pure-sink readout store accumulates the constant
  0.5 output each step, same accumulate semantics as the categorical composite's
  `neuron_outputs`).

## Tests

`142 passed, 1 skipped` (135 pre-existing + 7 new). All 9 repointed baselines
resolve via `build_generator(_REGISTRY[id])` and build a `Composite` with no
error.

## Concerns / deviations (all deliberate, with reasons)

1. **Bare-CTRNN default genome is NOT all-zeros.** The brief specified default
   `genome = np.zeros(genome_length(...))`, but an all-zero genome sets `tau=0`,
   so the CTRNN's `-y/tau` term divides by zero and the engine raises
   `FloatingPointError` on the first step (verified). `build_ctrnn_composite`
   instead defaults to the *default CTRNN* encoded for `size` (`tau=1, theta=0,
   zero weights`), which runs finite with no seed dependency. Studies pass a real
   genome via the `genome` arg. This was required to meet the "finite neuron
   outputs" success criterion.

2. **Body-process port types are `overwrite[...]`, not the plain `array[float]`/
   `float` in the brief's signature text.** The store's apply semantics come from
   the port wired to it: a store read/written through a plain `array[float]` port
   ACCUMULATES (add), while an `overwrite[...]` port SETS. The categorical body
   (the authoritative template) uses `overwrite` for exactly this reason. Using
   plain types would accumulate `sensory_input`/`motor_output` and break the
   feedback loop (and inflate the readouts). Verified empirically against the
   categorical composite. The walker/forager therefore mirror the categorical
   body and use `overwrite[...]`.

3. **Forager list-config params are `maybe[list[float]]` with null defaults, not
   `list[float]` with list defaults.** A `list[float]` config `_default` is
   applied by concatenation during schema realization, which silently DOUBLES the
   default (`[50,50]` -> `[50,50,50,50]`) and crashes the body. The real defaults
   are resolved in `__init__`. (resource_a/resource_b/init_levels/start_pos.)

4. **Pre-existing (NOT introduced): study.yaml jsonschema drift.** A full
   jsonschema validation of every study.yaml against `.pbg/schemas/study.schema.json`
   fails on a `behavior_tests` observable-entry shape — but this affects ALL 12
   studies identically, including the 3 categorical studies I never touched, so
   it predates this work. There is no jsonschema gate in the pytest suite (which
   is green); the baseline blocks themselves are schema-correct (`name` +
   `composite`) and resolve+build. Out of scope for this task, flagged for
   awareness.

5. **ctrnn baseline name.** Brief said `<model>-bbe`; used `ctrnn-bbe` (walker ->
   `walker-bbe`, forager -> `forager-bbe`).
