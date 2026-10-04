"""The parameter-space gallery renders its three figures to disk."""
import matplotlib
matplotlib.use("Agg")

from viva_bbe_systems.gallery import render


def test_render_writes_three_nonempty_figures(tmp_path):
    written = render(tmp_path, phase_res=8, bifurcation_points=6, codim2_res=6, animate=False)
    names = {p.name for p in written}
    assert names == {"phase_portrait.png", "bifurcation.png", "codim2.png"}
    for p in written:
        assert p.exists() and p.stat().st_size > 1000  # a real PNG, not a stub
