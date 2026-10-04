"""Composite generators for viva-bbe-systems.

Importing this package fires the ``@composite_generator`` decorators below so the
generators register in the process-bigraph composite registry and resolve in the
workbench. Defining them here (module ``viva_bbe_systems.composites``) gives clean
ids like ``viva_bbe_systems.composites.categorical_perception``.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from ..processes.categorical_env_process import build_categorical_composite


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


__all__ = ["categorical_perception"]
