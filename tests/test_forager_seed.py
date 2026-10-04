"""The committed forager seed reproduces Agmon & Beer (2014) action switching.

Pins the action-switching investigation's acceptance band: the evolved agent
forages BOTH resources (switches) across the trial battery, surviving well beyond
a passive non-mover by keeping both nutrients going. It is an honest partial
reproduction — the agent switches robustly but is not immortal (nutrients
net-decline, so it eventually dies after ~1-2.5k steps).
"""
import numpy as np

from viva_bbe_systems.tasks.evolve_forager import load_seed, DEFAULT_PATH, forager_spec
from viva_bbe_systems.tasks.forager_fitness import TRIAL_CONFIGS
from viva_bbe_systems.bodies.chemotactic_forager import ChemotacticForager
from viva_bbe_systems.agents.forager_agent import ForagerAgent
from viva_bbe_systems.environments.chemotaxis_resources import ChemotaxisEnv, Resource
from viva_bbe_systems.genome import decode

MAX_STEPS = 2500


def _run_all(genome):
    spec = forager_spec("M2")
    agent = ForagerAgent(decode(genome, spec), ChemotacticForager("M2"))
    both, survs = 0, []
    for c in TRIAL_CONFIGS:
        env = ChemotaxisEnv(
            Resource(center=np.array(c["resource_a"], float), signal="A"),
            Resource(center=np.array(c["resource_b"], float), signal="B"))
        r = agent.run_trial(env, c["init_levels"], start_pos=c["start_pos"],
                            start_angle=c["start_angle"], max_steps=MAX_STEPS)
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
    # action switching: forages BOTH resources in (almost) every config
    assert both >= 10, f"only forages both in {both}/11 configs"
    # survives well beyond a passive non-mover (which starves at ~min_level/drain)
    nm = np.zeros(len(g)); nm[:forager_spec("M2").size] = 1.0
    _, nm_surv, _ = _run_all(nm)
    assert mean_surv > 1.5 * nm_surv, f"mean survival {mean_surv:.0f} not >> non-mover {nm_surv:.0f}"


def test_seed_is_deterministic():
    g = load_seed(DEFAULT_PATH)
    assert _run_all(g)[2] == _run_all(g)[2]
