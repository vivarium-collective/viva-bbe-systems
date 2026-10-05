"""Static figures for the relational-categorization agent: how it remembers
object 1's size across the ISI, and the (s1, s2) decision boundary it learned.

    python -m viva_bbe_systems.relational_gallery [OUTDIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tasks.evolve_relational import load_seed, relational_spec, accuracy_report
from .bodies.relational_genome import decode_agent
from .environments.object_stream import TwoObjectStream

CATCH_COLOR = "#2ca02c"
AVOID_COLOR = "#d62728"
NAVY = "#11355e"


def _default_outdir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "workspace" / "investigations" / "relational-categorization" / "figures"


def trial_states(agent, s1, s2, *, offset1=0.0, offset2=0.0):
    """Step the CTRNN through a full trial by hand; return (internal state y (T,N), stream)."""
    stream = TwoObjectStream(s1, s2, offset1=offset1, offset2=offset2)
    agent.ctrnn.reset(np.zeros(agent.ctrnn.size))
    agent.body.x = 0.0
    outs = []
    for t in range(stream.total_steps):
        obj = stream.visible(t)
        shadow = (agent.body.sense(obj) if obj is not None
                  else np.zeros(agent.body.n_sensors))
        o = agent.ctrnn.step(external_input=agent.sensor_weights @ shadow)
        outs.append(agent.ctrnn.y.copy())
        motor = np.array([o[agent.motor_indices[0]], o[agent.motor_indices[1]]])
        agent.body.act(motor, 0.1, agent.motor_gain)
    return np.array(outs), stream


def memory_encoding(agent, s1_values, *, s2=4.0):
    """Mid-ISI internal neural state (y) for each s1. Returns (mid (n_s1, N), corr (N,), best idx)."""
    mid, traces, stream = [], [], None
    for s1 in s1_values:
        outs, stream = trial_states(agent, s1, s2)
        traces.append(outs)
        mid.append(outs[stream.phase1_steps + stream.isi_steps // 2])
    mid = np.array(mid)
    corr = np.zeros(mid.shape[1])
    for j in range(mid.shape[1]):
        if np.std(mid[:, j]) > 1e-9:
            corr[j] = np.corrcoef(s1_values, mid[:, j])[0, 1]
    n = mid.shape[1]
    cand = [j for j in range(n) if j not in [k % n for k in agent.motor_indices]] or list(range(n))
    best = max(cand, key=lambda j: abs(corr[j]))
    return mid, corr, int(best), np.array(traces), stream


def fig_memory_dynamics(agent, n_s1=9):
    s1_values = np.linspace(2.0, 6.0, n_s1)
    mid, corr, j, traces, stream = memory_encoding(agent, s1_values)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    axl.plot(s1_values, mid[:, j], "-o", color=NAVY)
    axl.set_xlabel("object 1 size  s1"); axl.set_ylabel(f"neuron {j} state y, mid-ISI")
    axl.set_title("Object 1's size is encoded in neural state")
    axl.text(0.05, 0.93, f"corr(s1, state) = {corr[j]:+.2f}", transform=axl.transAxes,
             va="top", fontsize=11, bbox=dict(fc="w", ec="0.7"))
    axl.grid(alpha=0.25)

    cmap = plt.get_cmap("viridis")
    p1, isi = stream.phase1_steps, stream.isi_steps
    t = np.arange(stream.total_steps)
    axr.axvspan(0, p1, color="0.9"); axr.axvspan(p1, p1 + isi, color="#ffe9a8")
    axr.axvspan(p1 + isi, stream.total_steps, color="0.9")
    dev = traces[:, :, j] - traces[:, :, j].mean(axis=0)   # remove the shared ramp
    for k, s1 in enumerate(s1_values):
        axr.plot(t, dev[k], color=cmap(k / max(1, n_s1 - 1)), lw=1.4)
    axr.axvline(p1 + isi, color="k", lw=0.8, ls=":")
    top = axr.get_ylim()[1]
    for x, lb in ((p1 / 2, "object 1"), (p1 + isi / 2, "ISI\n(no object)"),
                  (p1 + isi + stream.phase2_steps / 2, "object 2")):
        axr.text(x, top, lb, ha="center", va="top", fontsize=9)
    axr.set_xlabel("time step"); axr.set_ylabel(f"neuron {j} state y, minus mean over s1")
    axr.set_title("The trace is set in phase 1 and persists through the ISI")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(2, 6))
    fig.colorbar(sm, ax=axr, label="s1")
    fig.tight_layout()
    return fig


def decision_grid(agent, n=7, offsets=(0.0, 3.0), lo=2.0, hi=6.0):
    sizes = np.linspace(lo, hi, n)
    frac = np.zeros((n, n))   # [i_s2, j_s1]
    for i, s2 in enumerate(sizes):
        for j, s1 in enumerate(sizes):
            frac[i, j] = np.mean([agent.run_trial(s1, s2, offset2=o)["caught"]
                                  for o in offsets])
    return sizes, frac


def draw_decision_map(ax, sizes, frac):
    lo, hi = sizes[0], sizes[-1]
    h = (sizes[1] - sizes[0]) / 2 if len(sizes) > 1 else 0.5
    ax.imshow(frac, origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
              extent=[lo - h, hi + h, lo - h, hi + h], aspect="equal")
    ax.plot([lo - h, hi + h], [lo - h, hi + h], "k--", lw=1.5)
    ax.set_xlabel("object 1 size  s1"); ax.set_ylabel("object 2 size  s2")


def fig_decision_map(agent, n=7, offsets=(0.0, 3.0)):
    sizes, frac = decision_grid(agent, n, offsets)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    draw_decision_map(ax, sizes, frac)
    ax.text(0.04, 0.96, "should CATCH\n(s2 > s1)", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.96, 0.04, "should AVOID\n(s2 < s1)", transform=ax.transAxes, ha="right", fontsize=9)
    ax.set_title("Learned decision map (green = catch, red = avoid)\ndashed = ideal boundary s2 = s1")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1)), ax=ax,
                 label="fraction caught")
    fig.tight_layout()
    return fig


def render(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    agent = decode_agent(load_seed(), relational_spec())
    written = []
    for name, fig in (("memory_dynamics.png", fig_memory_dynamics(agent)),
                      ("decision_map.png", fig_decision_map(agent))):
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); written.append(p)
    return written


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else _default_outdir()
    for p in render(outdir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
