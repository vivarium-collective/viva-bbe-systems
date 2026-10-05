"""The committed forager seed reproduces Agmon & Beer (2014) action switching.

Pins the action-switching investigation's acceptance band: the evolved agent
forages BOTH resources (switches) across a battery of SPREAD environments
(separations ~38-65 units, varied orientation/size/start), surviving well beyond
a passive non-mover by keeping both nutrients going. The agent genuinely
navigates/searches for the far resource — a single tight circle cannot cover
these layouts. It forages both in all near+mid layouts (sep up to ~58); the far
tier (~58-70) is the reachability frontier, so a few of those fail. Honest
partial reproduction — it is not immortal.
"""
import numpy as np

from viva_bbe_systems.tasks.evolve_forager import load_seed, DEFAULT_PATH, forager_spec
from viva_bbe_systems.tasks.forager_fitness import TRIAL_CONFIGS
from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.agents.forager_agent import ForagerAgent
from viva_bbe_systems.environments.chemotaxis_resources import ChemotaxisEnv, Resource
from viva_bbe_systems.genome import decode

MAX_STEPS = 2000


def _run_all(genome):
    from viva_bbe_systems.tasks.forager_fitness import run_config
    spec = forager_spec("M2")
    agent = ForagerAgent(decode(genome, spec), ChemotacticForager("M2"))
    both, survs = 0, []
    for c in TRIAL_CONFIGS:
        r = run_config(agent, c, max_steps=MAX_STEPS)
        lh = r["levels_hist"]
        ate_a = float(np.diff(lh[:, 0]).max()) > 0   # ate from A at least once
        ate_b = float(np.diff(lh[:, 1]).max()) > 0   # ate from B at least once
        if ate_a and ate_b:
            both += 1
        survs.append(r["survival"])
    return both, float(np.mean(survs)), survs


def test_seed_switches_and_outlives_nonmover():
    assert DEFAULT_PATH.exists(), "committed forager seed missing"
    g = load_seed(DEFAULT_PATH)
    both, mean_surv, _ = _run_all(g)
    n = len(TRIAL_CONFIGS)
    # action switching: forages BOTH resources in most of the SPREAD battery
    # (achieved 12/16 — all near+mid layouts; the far tier is the frontier).
    # 11/16 leaves a one-config margin against the deterministic result.
    assert both >= 11, f"only forages both in {both}/{n} spread configs"
    # survives well beyond a passive non-mover (which starves at ~min_level/drain)
    nm = np.zeros(len(g)); nm[:forager_spec("M2").size] = 1.0
    _, nm_surv, _ = _run_all(nm)
    assert mean_surv > 1.5 * nm_surv, f"mean survival {mean_surv:.0f} not >> non-mover {nm_surv:.0f}"


def test_seed_is_deterministic():
    g = load_seed(DEFAULT_PATH)
    assert _run_all(g)[2] == _run_all(g)[2]
