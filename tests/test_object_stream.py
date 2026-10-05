from viva_bbe_systems.environments.object_stream import TwoObjectStream


def make():
    return TwoObjectStream(s1=3.0, s2=6.0, phase1_steps=10, isi_steps=4, phase2_steps=10)


def test_phase1_returns_obj1():
    s = make()
    for t in range(10):
        o = s.visible(t)
        assert o is not None and o.size == 3.0 and o.shape == "circle"
        assert s.phase(t) == "obj1"


def test_isi_nothing_visible():
    s = make()
    for t in range(10, 14):
        assert s.visible(t) is None
        assert s.phase(t) == "isi"


def test_phase2_returns_obj2_never_obj1():
    s = make()
    for t in range(14, 24):
        o = s.visible(t)
        assert o is not None and o.size == 6.0  # memory-forcing: never s1
        assert s.phase(t) == "obj2"


def test_after_done():
    s = make()
    assert s.visible(24) is None and s.phase(24) == "done"
    assert s.visible(100) is None


def test_objects_fall_and_reset():
    s = make()
    ys = [s.visible(t).center[1] for t in range(10)]
    assert ys[0] == 20.0 and all(a > b for a, b in zip(ys, ys[1:]))
    ys2 = [s.visible(t).center[1] for t in range(14, 24)]
    assert ys2[0] == 20.0 and all(a > b for a, b in zip(ys2, ys2[1:]))


def test_pure_and_offsets():
    s = TwoObjectStream(3, 6, offset1=1.5, offset2=-2.0, phase1_steps=10, isi_steps=4, phase2_steps=10)
    assert s.visible(5).center[1] == s.visible(5).center[1]
    assert s.visible(5).center[0] == 1.5 and s.visible(15).center[0] == -2.0


def test_totals_sizes_boundaries():
    s = make()
    assert s.total_steps == 24 and s.sizes == (3.0, 6.0)
    assert [s.phase(t) for t in (0, 9, 10, 13, 14, 23, 24)] == \
        ["obj1", "obj1", "isi", "isi", "obj2", "obj2", "done"]
