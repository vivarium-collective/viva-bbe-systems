"""The categorical-perception gallery renders its three figures from the seed."""
import matplotlib
matplotlib.use("Agg")

from viva_bbe_systems.categorical_gallery import render


def test_render_writes_three_nonempty_figures(tmp_path):
    written = render(tmp_path)
    names = {p.name for p in written}
    assert names == {"trajectories.png", "categorization_map.png", "decision_dynamics.png"}
    for p in written:
        assert p.exists() and p.stat().st_size > 1000  # real PNGs, not stubs
