import numpy as np

import amsa
from amsa import cga


def test_cga3d_null_basis_identities() -> None:
    alg = amsa.Algebra.cga3d()
    no = cga.origin(alg)
    ninf = cga.infinity(alg)

    assert np.allclose((no * no).component(0), 0.0)
    assert np.allclose((ninf * ninf).component(0), 0.0)
    assert np.allclose((no.inner(ninf)).component(0), -1.0)


def test_cga_point_is_null_and_distance_identity() -> None:
    alg = amsa.Algebra.cga3d()
    a = cga.point(alg, [1.0, 2.0, 3.0])
    b = cga.point(alg, [2.0, 2.0, 3.0])

    assert np.allclose((a * a).component(0), 0.0)
    assert np.allclose(cga.distance_squared(a, b), 1.0)


def test_cga_translator_moves_points() -> None:
    alg = amsa.Algebra.cga2d()
    x = cga.point(alg, [1.0, 2.0])
    translated = amsa.sandwich(cga.translate(alg, [3.0, -1.0]), x)
    expected = cga.point(alg, [4.0, 1.0])

    assert np.allclose(translated.to_layout(expected.layout).values, expected.values)
