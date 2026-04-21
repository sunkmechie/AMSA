import numpy as np

from amsa import Algebra
from amsa.viz import to_line, to_plane, to_rotor

from ._utils import assert_allclose


def test_pga3d_to_plane():
    alg = Algebra.pga3d()

    # Plane: x = 5  -> Vector P = 1*e1 - 5*e0
    # Normal is (1, 0, 0), distance -5
    plane_mv = alg.multivector({"e1": 1.0, "e0": -5.0})

    p = to_plane(plane_mv)
    # Origin should be the point on plane closest to origin: (5, 0, 0)
    assert_allclose(p.origin, [5.0, 0.0, 0.0], tol=1e-7)
    assert_allclose(p.normal, [1.0, 0.0, 0.0], tol=1e-7)


def test_pga3d_to_line():
    alg = Algebra.pga3d()

    # Line: Z-axis at x=1, y=2
    line_mv = alg.multivector({"e12": 1.0, "e01": -2.0, "e02": 1.0})

    line = to_line(line_mv)
    assert_allclose(line.direction, [0.0, 0.0, 1.0], tol=1e-7)
    assert_allclose(line.origin, [1.0, 2.0, 0.0], tol=1e-7)


def test_pga3d_to_rotor():
    alg = Algebra.pga3d()

    # 1. Identity motor
    m_id = alg.multivector({"e": 1.0})
    rotor = to_rotor(m_id)
    assert_allclose(rotor.origin, [0.0, 0.0, 0.0], tol=1e-7)
    assert_allclose(rotor.matrix, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], tol=1e-7)

    # 2. Translation of 5 units in X
    # T = 1 - 0.5 * (5 * e01)
    m_trans = alg.multivector({"e": 1.0, "e01": -2.5})
    rotor_trans = to_rotor(m_trans)
    assert_allclose(rotor_trans.origin, [5.0, 0.0, 0.0], tol=1e-7)
    assert_allclose(
        rotor_trans.matrix, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], tol=1e-7
    )

    # 3. 90-degree rotation around Z
    axis = alg.multivector({"e12": 1.0})
    m_rot = (axis * (np.pi / 4.0)).exp()  # Angle/2
    rotor_rot = to_rotor(m_rot)

    # In GA, exp(0.5 * theta * e12) rotates e1 towards -e2 (clockwise)
    expected_matrix = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert_allclose(rotor_rot.matrix, expected_matrix, tol=1e-7)
