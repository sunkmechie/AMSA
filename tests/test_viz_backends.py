from __future__ import annotations

import importlib

import pytest


def test_backend_namespace_is_lazy() -> None:
    backends = importlib.import_module("amsa.viz.backends")
    assert backends.__all__ == ["mpl", "vispy"]


def test_mpl_backend_smoke() -> None:
    pytest.importorskip("matplotlib")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from amsa import Algebra
    from amsa.viz.adapters import to_point
    from amsa.viz.backends import mpl

    alg = Algebra.pga2d()
    mv = alg.multivector({"e01": 3.0, "e02": 4.0, "e12": 1.0})

    _fig, ax = plt.subplots()
    artist = mpl.plot(ax, to_point(mv))

    assert artist is not None


def test_vispy_backend_smoke() -> None:
    pytest.importorskip("vispy")

    from vispy import scene

    from amsa import Algebra
    from amsa.viz.adapters import to_line_segments
    from amsa.viz.backends import vispy as vback

    alg = Algebra.pga2d()
    mv = alg.multivector({"e01": [0.0, 1.0], "e02": [0.0, 1.0], "e12": 1.0}, batch_shape=(2,))

    _canvas = scene.SceneCanvas(show=False)
    view = _canvas.central_widget.add_view()
    visual = vback.plot(view.scene, to_line_segments(mv))

    assert visual is not None
