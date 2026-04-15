# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from amsa import Algebra
from amsa import viz
from amsa.viz.primitives import Point

def test_backend_resolution_logic():
    from amsa.viz import core
    # Test manual override
    viz.use_backend("mpl")
    assert core._resolve_backend() == "mpl"
    
    # Reset to auto
    viz.use_backend("auto")
    # This will depend on what's installed in the test env
    backend = core._resolve_backend()
    assert backend in ("mpl", "vispy")

def test_view_dispatch_returns_correct_types():
    alg = Algebra.pga2d()
    mv = alg.multivector({"e01": 1.0, "e12": 1.0})
    
    # Force MPL for test stability
    viz.use_backend("mpl")
    from matplotlib.axes import Axes
    ax = viz.view(mv)
    assert isinstance(ax, Axes)
    
    # If vispy is present, we could test it too
    try:
        import vispy
        viz.use_backend("vispy")
        from amsa.viz.backends.vispy import AMSAScene
        scene = viz.view(mv)
        assert isinstance(scene, AMSAScene)
    except ImportError:
        pass

def test_to_point_integration():
    alg = Algebra.pga2d()
    # Create a batch of 10 points
    coords = np.random.randn(10, 2)
    mv = alg.multivector({
        "e01": coords[:, 0],
        "e02": coords[:, 1],
        "e12": 1.0
    })
    
    primitive = viz.to_point(mv)
    assert isinstance(primitive, Point)
    assert primitive.position.shape == (10, 2)
    np.testing.assert_allclose(primitive.position, coords)

def test_plot_dispatches_to_mpl():
    viz.use_backend("mpl")
    alg = Algebra.pga2d()
    mv = alg.multivector({"e01": 1.0, "e12": 1.0})
    p = viz.to_point(mv)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # This should run without error
    viz.plot(p, ax=ax)
    plt.close(fig)

def test_vispy_plot_requires_scene():
    try:
        import vispy
        viz.use_backend("vispy")
        p = Point(position=np.array([0, 0]))
        with pytest.raises(RuntimeError, match="requires an AMSAScene"):
            viz.plot(p)
    except ImportError:
        pass
