import numpy as np
import pytest

import amsa


def test_cga3d_null_basis_identities() -> None:
    alg = amsa.Algebra.cga3d()
    no = alg.origin()
    ninf = alg.infinity()

    assert np.allclose((no * no).component(0), 0.0)
    assert np.allclose((ninf * ninf).component(0), 0.0)
    assert np.allclose((no.inner(ninf)).component(0), -1.0)


def test_cga_point_is_null_and_distance_identity() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([1.0, 2.0, 3.0])
    b = alg.point([2.0, 2.0, 3.0])

    assert np.allclose((a * a).component(0), 0.0)
    assert np.allclose(alg.distance_squared(a, b), 1.0)


def test_cga_translator_moves_points() -> None:
    alg = amsa.Algebra.cga2d()
    x = alg.point([1.0, 2.0])
    translated = amsa.sandwich(alg.translate([3.0, -1.0]), x)
    expected = alg.point([4.0, 1.0])

    assert np.allclose(translated.to_layout(expected.layout).values, expected.values)


def test_algebra_origin_returns_null_vector() -> None:
    alg = amsa.Algebra.cga3d()
    no = alg.origin()
    assert np.allclose((no * no).component(0), 0.0)


def test_algebra_infinity_returns_null_vector() -> None:
    alg = amsa.Algebra.cga2d()
    ninf = alg.infinity()
    assert np.allclose((ninf * ninf).component(0), 0.0)


def test_algebra_sphere_squares_to_radius_squared() -> None:
    alg = amsa.Algebra.cga3d()
    s = alg.sphere([0.0, 0.0, 0.0], 1.0)
    result = alg.norm_squared(s)
    assert np.allclose(result.component(0), 1.0)  # dual sphere S^2 = r^2


def test_algebra_plane_squares_to_normal_squared() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.plane([0.0, 0.0, 1.0], 2.0)
    result = alg.norm_squared(p)
    assert np.allclose(result.component(0), 1.0)  # dual plane P^2 = n^2 = 1


def test_algebra_line_through_points() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([0.0, 0.0, 0.0])
    b = alg.point([1.0, 0.0, 0.0])
    L = alg.line_through_points(a, b)
    result = alg.norm_squared(L)
    assert np.allclose(np.abs(result.component(0)), 1.0)


def test_algebra_circle_through_points() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([1.0, 0.0, 0.0])
    b = alg.point([0.0, 1.0, 0.0])
    c = alg.point([-1.0, 0.0, 0.0])
    C = alg.circle_through_points(a, b, c)
    result = alg.norm_squared(C)
    assert np.allclose(np.abs(result.component(0)), 4.0)


def test_algebra_translate_returns_motor() -> None:
    alg = amsa.Algebra.cga3d()
    T = alg.translate([1.0, 2.0, 3.0])
    assert T.algebra == alg.spec


def test_algebra_euclidean_vector() -> None:
    alg = amsa.Algebra.cga3d()
    v = alg.euclidean_vector([1.0, 2.0, 3.0])
    assert np.allclose((v * v).component(0), 14.0)


def test_algebra_cga_methods_reject_non_cga() -> None:
    alg = amsa.Algebra.vga3d()
    with pytest.raises(ValueError):
        alg.origin()


def test_cga_standalone_still_works() -> None:
    from amsa.cga import distance_squared, infinity, origin, point

    alg = amsa.Algebra.cga3d()
    no = origin(alg)
    ninf = infinity(alg)
    a = point(alg, [1.0, 2.0, 3.0])
    b = point(alg, [2.0, 2.0, 3.0])

    assert np.allclose((no * no).component(0), 0.0)
    assert np.allclose((ninf * ninf).component(0), 0.0)
    assert np.allclose(distance_squared(alg, a, b), 1.0)


# -- extraction utilities -------------------------------------------------------


def test_extract_point_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.point([1.0, 2.0, 3.0])
    coords = alg.extract_point(p)
    assert np.allclose(coords, [1.0, 2.0, 3.0])


def test_extract_sphere_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    s = alg.sphere([1.0, 0.0, 0.0], 3.0)
    center, radius = alg.extract_sphere(s)
    assert np.allclose(center, [1.0, 0.0, 0.0])
    assert np.allclose(radius, 3.0)


def test_extract_sphere_origin() -> None:
    alg = amsa.Algebra.cga3d()
    s = alg.sphere([0.0, 0.0, 0.0], 2.5)
    center, radius = alg.extract_sphere(s)
    assert np.allclose(center, [0.0, 0.0, 0.0])
    assert np.allclose(radius, 2.5)


def test_extract_plane_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.plane([0.0, 0.0, 1.0], 5.0)
    normal, distance = alg.extract_plane(p)
    assert np.allclose(normal, [0.0, 0.0, 1.0])
    assert np.allclose(distance, 5.0)


def test_extract_plane_default_normal() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.plane([1.0, 0.0, 0.0], 3.0)
    normal, distance = alg.extract_plane(p)
    assert np.allclose(normal, [1.0, 0.0, 0.0])
    assert np.allclose(distance, 3.0)


def test_extract_euclidean_vector_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    v = alg.euclidean_vector([4.0, 5.0, 6.0])
    coords = alg.extract_euclidean_vector(v)
    assert np.allclose(coords, [4.0, 5.0, 6.0])


def test_extract_point_reflected() -> None:
    alg = amsa.Algebra.cga3d()
    plane = alg.plane([1.0, 0.0, 0.0], 0.0)
    p = alg.point([3.0, 2.0, 1.0])
    reflected = amsa.sandwich(plane, p)
    coords = alg.extract_point(reflected)
    assert np.allclose(coords, [-3.0, 2.0, 1.0])


def test_extract_point_translated() -> None:
    alg = amsa.Algebra.cga2d()
    T = alg.translate([5.0, -2.0])
    p = alg.point([1.0, 1.0])
    moved = amsa.sandwich(T, p)
    coords = alg.extract_point(moved)
    assert np.allclose(coords, [6.0, -1.0])


def test_extract_cga2d() -> None:
    alg = amsa.Algebra.cga2d()
    p = alg.point([3.0, 4.0])
    coords = alg.extract_point(p)
    assert np.allclose(coords, [3.0, 4.0])

    s = alg.sphere([1.0, 1.0], 2.0)
    center, radius = alg.extract_sphere(s)
    assert np.allclose(center, [1.0, 1.0])
    assert np.allclose(radius, 2.0)

    pl = alg.plane([0.0, 1.0], 3.0)
    normal, distance = alg.extract_plane(pl)
    assert np.allclose(normal, [0.0, 1.0])
    assert np.allclose(distance, 3.0)


def test_extract_standalone_functions() -> None:
    from amsa.cga import extract_plane, extract_point, extract_sphere

    alg = amsa.Algebra.cga3d()
    p = alg.point([1.0, 2.0, 3.0])
    assert np.allclose(extract_point(p), [1.0, 2.0, 3.0])

    s = alg.sphere([0.0, 0.0, 0.0], 4.0)
    center, radius = extract_sphere(s)
    assert np.allclose(center, [0.0, 0.0, 0.0])
    assert np.allclose(radius, 4.0)

    pl = alg.plane([0.0, 0.0, 1.0], 2.0)
    normal, distance = extract_plane(pl)
    assert np.allclose(normal, [0.0, 0.0, 1.0])
    assert np.allclose(distance, 2.0)


def test_extract_rejects_non_cga() -> None:
    alg = amsa.Algebra.vga3d()
    v = alg.vector([1.0, 2.0, 3.0])
    from amsa.cga import extract_point

    with pytest.raises(ValueError):
        extract_point(v)
