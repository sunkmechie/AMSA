import numpy as np
import pytest

from amsa import Algebra
from amsa.viz.adapters import to_point


def test_pga2d_to_point():
    alg = Algebra.pga2d()
    
    # Standard point at (3, 4) in pga2d: x e01 + y e02 + 1 e12
    # So e01 = 3, e02 = 4, e12 = 1
    pt_mv = alg.multivector({"e01": 3.0, "e02": 4.0, "e12": 1.0})
    
    pt = to_point(pt_mv, label="TestPt", color="red")
    
    np.testing.assert_allclose(pt.position, [3.0, 4.0])
    assert pt.label == "TestPt"
    assert pt.color == "red"


def test_pga2d_to_point_with_weight():
    alg = Algebra.pga2d()
    # Point at (3, 4) but with weight 2: x * w = 6, y * w = 8, w = 2
    pt_mv = alg.multivector({"e01": 6.0, "e02": 8.0, "e12": 2.0})
    
    pt = to_point(pt_mv)
    np.testing.assert_allclose(pt.position, [3.0, 4.0])


def test_pga3d_to_point():
    alg = Algebra.pga3d()
    
    # x=-1 corresponds to e032=1, meaning e023=-1
    # y=2 corresponds to e013=2
    # z=-3 corresponds to e021=3, meaning e012=-3
    pt_mv = alg.multivector({"e023": -1.0, "e013": 2.0, "e012": -3.0, "e123": 1.0})
    
    pt = to_point(pt_mv)
    np.testing.assert_allclose(pt.position, [1.0, 2.0, 3.0])


def test_unsupported_algebra_for_point():
    alg = Algebra.vga2d()
    # In vga2d, points are just vectors (e1, e2).
    # Since I haven't implemented it in the adapter yet, it should raise NotImplementedError.
    pt_mv = alg.vector([1.0, 2.0])
    
    with pytest.raises(NotImplementedError):
        to_point(pt_mv)
