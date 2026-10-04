"""Shared animation helper: render a matplotlib FuncAnimation to an animated GIF.

GIF (Pillow writer) is used so the output embeds anywhere as a plain <img> with
no video-codec dependency in the viewer. Rendering needs only matplotlib+Pillow.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def save_gif(fig, update, frames, path, *, fps=20, dpi=80, init=None) -> Path:
    """Render `update(i)` over `frames` frames of `fig` to a GIF at `path`.

    `update(i)` mutates the figure's artists for frame i; `init` (optional) sets
    up the first frame. Returns the output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
