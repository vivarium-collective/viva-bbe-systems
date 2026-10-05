"""Composite generators for viva-bbe-systems.

Importing this package fires the ``@composite_generator`` decorators below so the
generators register in the process-bigraph composite registry and resolve in the
workbench. Defining them here (module ``viva_bbe_systems.composites``) gives clean
ids like ``viva_bbe_systems.composites.categorical_perception``.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from ..processes.categorical_env_process import build_categorical_composite
from ..processes.ctrnn_process import build_ctrnn_composite
from ..processes.forager_body_process import build_forager_composite
from ..processes.relational_body_process import build_relational_composite
from ..processes.walker_body_process import build_walker_composite


@composite_generator(
    name="categorical_perception",
    description=(
        "Beer-2003 active categorical perception: a brain-body-environment agent "
        "(FallingObjectEnvironment -> CategoricalBodyProcess -> CTRNNProcess) "
        "configured from the committed evolved seed. It catches circles and "
        "avoids diamonds. offset/shape select the trial."
    ),
    parameters={
        "offset": {"type": "float", "default": 3.0},
        "shape": {"type": "string", "default": "circle"},
    },
)
def categorical_perception(core=None, *, offset=3.0, shape="circle", **kwargs) -> dict:
    return build_categorical_composite(offset=offset, shape=shape)


@composite_generator(
    name="walker",
    description=(
        "Beer & Gallagher 1992 legged CPG: CTRNNProcess -> WalkerBodyProcess "
        "(one leg, flat ground), from the committed walker seed. Walks forward "
        "with a rhythmic gait."
    ),
    parameters={"dt": {"type": "float", "default": 0.1}},
)
def walker(core=None, *, dt=0.1, **kwargs) -> dict:
    return build_walker_composite(dt=dt)


@composite_generator(
    name="forager",
    description=(
        "Agmon & Beer 2014 action-switching forager: ChemotaxisEnv+Metabolism -> "
        "ForagerBodyProcess -> CTRNNProcess, from the committed M2 seed. Shuttles "
        "between two resources to keep both nutrients alive."
    ),
    parameters={"morphology": {"type": "string", "default": "M2"}},
)
def forager(core=None, *, morphology="M2", **kwargs) -> dict:
    return build_forager_composite(morphology=morphology)


@composite_generator(
    name="relational",
    description=(
        "Williams Beer Gasser 2008 relational categorization: TwoObjectStream -> "
        "RelationalBodyProcess -> CTRNNProcess, from the committed seed. Catches "
        "the 2nd object iff larger than the 1st (requires memory of the 1st "
        "across the ISI)."
    ),
    parameters={"s1": {"type": "float", "default": 3.0},
                "s2": {"type": "float", "default": 5.0}},
)
def relational(core=None, *, s1=3.0, s2=5.0, **kwargs) -> dict:
    return build_relational_composite(s1=s1, s2=s2)


@composite_generator(
    name="ctrnn_parameter_space",
    description=(
        "A bare CTRNN (Beer 1995) exploring its own dynamics -- the substrate for "
        "the parameter-space bifurcation/equilibria studies."
    ),
    parameters={"size": {"type": "integer", "default": 5}},
)
def ctrnn_parameter_space(core=None, *, size=5, **kwargs) -> dict:
    return build_ctrnn_composite(size=size)


__all__ = [
    "categorical_perception",
    "walker",
    "forager",
    "relational",
    "ctrnn_parameter_space",
]
