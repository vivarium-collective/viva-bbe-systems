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


def _run_all_ablated(genome):
    """Same, but with the agent's INTERNAL NUTRIENT SENSORS zeroed — it can no
    longer perceive its own metabolic state, only the chemical gradients."""
    from viva_bbe_systems.tasks.forager_fitness import run_config
    spec = forager_spec("M2")
    agent = ForagerAgent(decode(genome, spec), ChemotacticForager("M2"))
    orig, nchemo = agent.body.sense, agent.body.n_chemo

    def sense_no_nutrient(env, metab):
        s = orig(env, metab).copy(); s[nchemo:] = 0.0; return s
    agent.body.sense = sense_no_nutrient
    both, survs = 0, []
    for c in TRIAL_CONFIGS:
        r = run_config(agent, c, max_steps=MAX_STEPS)
        lh = r["levels_hist"]
        if float(np.diff(lh[:, 0]).max()) > 0 and float(np.diff(lh[:, 1]).max()) > 0:
            both += 1
        survs.append(r["survival"])
    return both, float(np.mean(survs))


def test_seed_switches_and_outlives_nonmover():
    assert DEFAULT_PATH.exists(), "committed forager seed missing"
    g = load_seed(DEFAULT_PATH)
    both, mean_surv, _ = _run_all(g)
    n = len(TRIAL_CONFIGS)
    # forages BOTH resources in most of the far, high-drain, asymmetric battery
    assert both >= 11, f"only forages both in {both}/{n} spread configs"
    # outlives a passive non-mover (the high-drain battery + starting buffer lifts
    # the non-mover baseline, so the margin is modest but real)
    nm = np.zeros(len(g)); nm[:forager_spec("M2").size] = 1.0
    _, nm_surv, _ = _run_all(nm)
    assert mean_surv > 1.3 * nm_surv, f"mean survival {mean_surv:.0f} not >> non-mover {nm_surv:.0f}"


def test_seed_uses_internal_nutrient_state():
    """The agent genuinely uses its internal nutrient sensors (not blind
    circling): ablating them measurably hurts foraging. This is the signature of
    the state-dependent action switching the far+high-drain pressure selects for."""
    g = load_seed(DEFAULT_PATH)
    both, surv = _run_all(g)[0], _run_all(g)[1]
    abl_both, abl_surv = _run_all_ablated(g)
    # removing the internal state sense costs survival and/or forages-both
    assert (surv - abl_surv) > 50 or (both - abl_both) >= 1, (
        f"ablating nutrient sensors barely changed behaviour "
        f"(intact {both}/{surv:.0f} vs ablated {abl_both}/{abl_surv:.0f}) "
        f"=> the agent is NOT using internal state")


def test_seed_is_deterministic():
    g = load_seed(DEFAULT_PATH)
    assert _run_all(g)[2] == _run_all(g)[2]
