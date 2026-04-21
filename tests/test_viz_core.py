import numpy as np

from amsa import Algebra
from amsa.viz.adapters import to_point
from amsa.viz.primitives import Point


def test_to_point_integration():
    alg = Algebra.pga2d()
    coords = np.random.randn(10, 2)
    mv = alg.multivector({"e01": coords[:, 0], "e02": coords[:, 1], "e12": 1.0})

    primitive = to_point(mv)
    assert isinstance(primitive, Point)
    assert primitive.position.shape == (10, 2)
    np.testing.assert_allclose(primitive.position, coords)
