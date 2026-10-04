# Build 3 BBE composites + register + repoint study baselines

GOAL: Only `categorical_perception` currently appears in the workbench Composites
registry. Build real process-bigraph BBE composites for the WALKER, the FORAGER,
and a bare CTRNN (parameter-space), register them with `@composite_generator`, and
repoint those investigations' study baselines to them so each appears in the
registry and is runnable/explorable.

## The template (READ THESE FIRST — copy the pattern exactly)
- `viva_bbe_systems/composites/__init__.py` — the `@composite_generator` registration pattern (clean id `viva_bbe_systems.composites.<name>`).
- `viva_bbe_systems/processes/categorical_env_process.py` — `FallingObjectEnvironment(Process)` + `build_categorical_composite()` returning the process-bigraph state-dict (env/body/brain wired through stores with one-step delay).
- `viva_bbe_systems/processes/categorical_body_process.py` — a body Process wrapping a plain body class.
- `tests/test_categorical_composite.py` — how a composite is RUN THROUGH THE ENGINE: `Composite({"state": spec}, core=build_core())` then `sim.run(0.1)`; and how processes are discovered (`build_core().link_registry`).
- `viva_bbe_systems/core.py` — `build_core()` process auto-discovery (your new Process classes must be discovered the same way the categorical ones are — put them in `viva_bbe_systems/processes/` and follow whatever registration the existing ones use).

## The CTRNNProcess contract (the shared brain — DO NOT edit it)
`viva_bbe_systems/processes/ctrnn_process.py`: inputs `sensory_input` = an array of length `size` that enters the CTRNN as the full input-current vector `I` (zero-pad non-sensor neurons). Config: `size, dt, genome (list), motor_indices`. Outputs `motor_output = o[motor_indices]`, `neuron_outputs`, `neuron_states`.

## Exact per-model specs (confirmed)

### 1. WALKER composite (simplest — no env; flat ground is in the body)
Create `viva_bbe_systems/processes/walker_body_process.py`:
- `class WalkerBodyProcess(Process)` wrapping `viva_bbe_systems.bodies.legged.LeggedBody`.
  - config_schema: `size` (int, default 5), `angle_input_index` (int, 0), `dt` (float, 0.1), `start_angle` (float, 0.0), and the leg params `leg_length` (15.0), `omega_gain` (1.0), `friction` (0.9), `foot_threshold` (0.5). (angle_range ±pi/6 is LeggedBody's default — fine.)
  - `__init__`: build `self.body = LeggedBody(leg_length=..., omega_gain=..., friction=..., foot_threshold=...)`; `self.body.reset(angle=start_angle)`.
  - `inputs()`: `{"motor_output": "array[float]"}`  (the [foot, bs, fs] from the brain).
  - `outputs()`: `{"sensory_input": "array[float]", "body_x": "float", "foot_down": "float", "leg_angle": "float", "body_vx": "float"}`.
  - `update(state, interval)`: `m = np.asarray(state["motor_output"], float)`; `self.body.act(m[0], m[1], m[2], self.config["dt"])`; build `cur = np.zeros(size); cur[angle_input_index] = self.body.sense()`; return `{"sensory_input": cur, "body_x": float(body.x), "foot_down": float(body.foot_down), "leg_angle": float(body.angle), "body_vx": float(body.vx)}`.
- `build_walker_composite(seed_path=None, dt=0.1) -> dict`: load the walker seed via `viva_bbe_systems.tasks.evolve_walker.load_seed` + `walker_spec`; brain = `local:CTRNNProcess` config `{size:5, dt, genome: seed.tolist(), motor_indices:[2,3,4]}` reading `sensory_input` → outputs `motor_output`/`neuron_outputs`/`neuron_states`; body = `local:WalkerBodyProcess` reading `motor_output` → outputs `sensory_input` + readout stores. Initial stores: `sensory_input=[0.0]*5`, `motor_output=[0.0,0.0,0.0]`, `body_x=0.0`, `foot_down=0.0`, `leg_angle=0.0`, `body_vx=0.0`, `neuron_outputs=[0.0]*5`, `neuron_states=[0.0]*5`. (Mirror the categorical builder's dict shape EXACTLY — `_type: process, address, config, interval, inputs, outputs` + the store initial values at top level.)

### 2. FORAGER composite (env + metabolism baked into the body process, like categorical bakes the falling object)
Create `viva_bbe_systems/processes/forager_body_process.py`:
- `class ForagerBodyProcess(Process)` wrapping `ChemotacticForager` (`viva_bbe_systems.bodies.chemotactic_forager`) + `ChemotaxisEnv`/`Resource` (`viva_bbe_systems.environments.chemotaxis_resources`) + `Metabolism` (`viva_bbe_systems.metabolism`).
  - config_schema: `morphology` (string, "M2"), `size` (int, 9), `n_sensors` (int, 4), `dt` (float), `resource_a` (array, default [30,70]), `resource_b` (array, [70,30]), `radius_a` (float, 7.0), `radius_b` (float, 7.0), `init_levels` (array, [5.0,5.0]), `start_pos` (array, [50,50]), `start_angle` (float, 0.0).
  - `__init__`: `self.body = ChemotacticForager(morphology)`; `self.env = ChemotaxisEnv(Resource(center=resource_a, radius=radius_a, signal="A"), Resource(center=resource_b, radius=radius_b, signal="B"))`; `self.metab = Metabolism(init_levels)`; set `self.body.pos = array(start_pos)`, `self.body.angle = start_angle`, `self.body.velocity = 0.0`.
  - `inputs()`: `{"motor_output": "array[float]"}` (brain outputs `o[[8,7]]` = [right, left]).
  - `outputs()`: `{"sensory_input": "array[float]", "agent_pos": "array[float]", "nutrient_levels": "array[float]", "alive": "float"}`.
  - `update(state, interval)`: `m = asarray(state["motor_output"], float)`; `self.body.act(m[0], m[1], self.env)`; `ia, ib = self.body.inside(self.env)`; `self.metab.step(ia, ib)`; `cur = zeros(size); cur[:n_sensors] = self.body.sense(self.env, self.metab)`; return `{"sensory_input": cur, "agent_pos": self.body.pos.copy(), "nutrient_levels": self.metab.levels.copy(), "alive": float(self.metab.alive)}`. (Match the run_trial loop in `agents/forager_agent.py`: act→inside→metab.step→sense-for-next.)
- `build_forager_composite(morphology="M2", seed_path=None, dt=0.1, **resource_kw) -> dict`: load the forager seed via `viva_bbe_systems.tasks.evolve_forager.load_seed`+`forager_spec(morphology)`; brain = `CTRNNProcess` `{size:9, dt, genome, motor_indices:[8,7]}`; body = `ForagerBodyProcess` with morphology/size/n_sensors (M2: size 9, n_sensors 4) + resource config. Initial stores: `sensory_input=[0.0]*9`, `motor_output=[0.0,0.0]`, plus `agent_pos`, `nutrient_levels`, `alive`, `neuron_outputs=[0.0]*9`, `neuron_states=[0.0]*9`.

### 3. CTRNN parameter-space composite (bare brain)
Add `build_ctrnn_composite(size=5, genome=None, dt=0.1, motor_indices=None) -> dict` to `viva_bbe_systems/processes/ctrnn_process.py` (or a small new module): a composite with ONE `brain` process = `CTRNNProcess` config `{size, dt, genome: (genome or zeros(genome_length)).tolist(), motor_indices: motor_indices or []}`, reading `sensory_input=[0.0]*size` → outputs `neuron_outputs`/`neuron_states` (and `motor_output`). A self-driven CTRNN exploring its own dynamics (no body/env). Default genome = `np.zeros(genome_length(GenomeSpec(size)))` so it builds with no seed dependency.

## Register (composites/__init__.py — add 3 decorators)
```
@composite_generator(name="walker", description="Beer & Gallagher 1992 legged CPG: CTRNNProcess -> WalkerBodyProcess (one leg, flat ground), from the committed walker seed. Walks forward with a rhythmic gait.", parameters={"dt":{"type":"float","default":0.1}})
def walker(core=None, *, dt=0.1, **kw): return build_walker_composite(dt=dt)

@composite_generator(name="forager", description="Agmon & Beer 2014 action-switching forager: ChemotaxisEnv+Metabolism -> ForagerBodyProcess -> CTRNNProcess, from the committed M2 seed. Shuttles between two resources to keep both nutrients alive.", parameters={"morphology":{"type":"string","default":"M2"}})
def forager(core=None, *, morphology="M2", **kw): return build_forager_composite(morphology=morphology)

@composite_generator(name="ctrnn_parameter_space", description="A bare CTRNN (Beer 1995) exploring its own dynamics — the substrate for the parameter-space bifurcation/equilibria studies.", parameters={"size":{"type":"integer","default":5}})
def ctrnn_parameter_space(core=None, *, size=5, **kw): return build_ctrnn_composite(size=size)
```
Add all to `__all__`. Import the builders at top.

## Repoint study baselines (9 files)
In each study.yaml, replace the `baseline:` entry's `process: CTRNNProcess` (+ its `params:`) with a composite reference, KEEPING a non-empty baseline:
```
baseline:
- name: <model>-bbe
  composite: viva_bbe_systems.composites.<name>
  params: {}
```
- `workspace/investigations/legged-locomotion/studies/{walk-forward,cpg-limit-cycle,gait-and-velocity}/study.yaml` → `walker`
- `workspace/investigations/action-switching/studies/{as-forage-and-survival,as-switching-dynamics,as-morphology-comparison}/study.yaml` → `forager`
- `workspace/investigations/ctrnn-parameter-space/studies/{param-space-codim1-bifurcations,param-space-codim2-structure,param-space-equilibria-and-nullclines}/study.yaml` → `ctrnn_parameter_space`
(Edit ONLY the baseline block; leave visualizations/findings/etc untouched. Use a YAML round-trip or careful text edit.)

## Tests (tests/test_bbe_composites.py — MANDATORY engine-run, mirror test_categorical_composite.py)
For EACH composite:
- processes discovered: `build_core().link_registry` contains `WalkerBodyProcess` / `ForagerBodyProcess` (and `CTRNNProcess`).
- generator registered: the `@composite_generator` name resolves (import `viva_bbe_systems.composites`; check the generator is registered — mirror `tests/test_core_registration.py` for how).
- builds + RUNS through the engine: `Composite({"state": build_*_composite()}, core=build_core())`, `sim.run(0.1)` for ~30-60 steps, no error, and a sensible readout changes:
  - walker: `sim.state["body_x"]` increases over the run (the committed seed walks; assert final > 50).
  - forager: `sim.state["nutrient_levels"]` is a length-2 vector and the agent moves (`agent_pos` changes from start); assert it runs 60 steps without error.
  - ctrnn: `sim.state["neuron_outputs"]` is length-size and finite after the run.
Keep tests reasonably fast.

## Constraints
- Do NOT edit the shared core (`ctrnn.py`, `genome.py`, `evolution.py`, `core.py`) or `ctrnn_process.py`'s CTRNNProcess class (you may ADD `build_ctrnn_composite` to that file). Reuse everything else unchanged.
- No AI attribution in the commit.
- Strict TDD where practical: write the engine-run smoke tests, watch them fail, implement, go green, run the FULL suite (must stay green — the repointed studies must still validate).
- After green, verify each composite RESOLVES (the earlier "composite not found in registry" failure must NOT reappear): the repointed baselines must resolve against the registered generators.
- Commit `feat: BBE composites (walker, forager, ctrnn) registered + study baselines repointed`.

Write a report to `.notes/bbe-composites-report.md` (what you built, each composite's engine-run readout, any concerns).
