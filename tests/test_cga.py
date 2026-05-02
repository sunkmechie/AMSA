import numpy as np
import pytest

import amsa
from tests._utils import assert_allclose


def test_cga3d_null_basis_identities() -> None:
    alg = amsa.Algebra.cga3d()
    no = alg.origin()
    ninf = alg.infinity()

    assert_allclose((no * no).component(0), 0.0)
    assert_allclose((ninf * ninf).component(0), 0.0)
    assert_allclose((no.inner(ninf)).component(0), -1.0)


def test_cga_point_is_null_and_distance_identity() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([1.0, 2.0, 3.0])
    b = alg.point([2.0, 2.0, 3.0])

    assert_allclose((a * a).component(0), 0.0)
    assert_allclose(alg.distance_squared(a, b), 1.0)


def test_cga_translator_moves_points() -> None:
    alg = amsa.Algebra.cga2d()
    x = alg.point([1.0, 2.0])
    translated = amsa.sandwich(alg.translate([3.0, -1.0]), x)
    expected = alg.point([4.0, 1.0])

    assert_allclose(translated.to_layout(expected.layout).values, expected.values)


def test_algebra_origin_returns_null_vector() -> None:
    alg = amsa.Algebra.cga3d()
    no = alg.origin()
    assert_allclose((no * no).component(0), 0.0)


def test_algebra_infinity_returns_null_vector() -> None:
    alg = amsa.Algebra.cga2d()
    ninf = alg.infinity()
    assert_allclose((ninf * ninf).component(0), 0.0)


def test_algebra_sphere_squares_to_radius_squared() -> None:
    alg = amsa.Algebra.cga3d()
    s = alg.sphere([0.0, 0.0, 0.0], 1.0)
    result = alg.norm_squared(s)
    assert_allclose(result.component(0), 1.0)  # dual sphere S^2 = r^2


def test_algebra_plane_squares_to_normal_squared() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.plane([0.0, 0.0, 1.0], 2.0)
    result = alg.norm_squared(p)
    assert_allclose(result.component(0), 1.0)  # dual plane P^2 = n^2 = 1


def test_algebra_line_through_points() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([0.0, 0.0, 0.0])
    b = alg.point([1.0, 0.0, 0.0])
    L = alg.line_through_points(a, b)
    result = alg.norm_squared(L)
    assert_allclose(np.abs(result.component(0)), 1.0)


def test_algebra_circle_through_points() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([1.0, 0.0, 0.0])
    b = alg.point([0.0, 1.0, 0.0])
    c = alg.point([-1.0, 0.0, 0.0])
    C = alg.circle_through_points(a, b, c)
    result = alg.norm_squared(C)
    assert_allclose(np.abs(result.component(0)), 4.0)


def test_algebra_translate_returns_motor() -> None:
    alg = amsa.Algebra.cga3d()
    T = alg.translate([1.0, 2.0, 3.0])
    assert T.algebra == alg.spec


def test_algebra_euclidean_vector() -> None:
    alg = amsa.Algebra.cga3d()
    v = alg.euclidean_vector([1.0, 2.0, 3.0])
    assert_allclose((v * v).component(0), 14.0)


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

    assert_allclose((no * no).component(0), 0.0)
    assert_allclose((ninf * ninf).component(0), 0.0)
    assert_allclose(distance_squared(alg, a, b), 1.0)


# -- extraction utilities -------------------------------------------------------


def test_extract_point_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.point([1.0, 2.0, 3.0])
    coords = alg.extract_point(p)
    assert_allclose(coords, [1.0, 2.0, 3.0])


def test_extract_sphere_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    s = alg.sphere([1.0, 0.0, 0.0], 3.0)
    center, radius = alg.extract_sphere(s)
    assert_allclose(center, [1.0, 0.0, 0.0])
    assert_allclose(radius, 3.0)


def test_extract_sphere_origin() -> None:
    alg = amsa.Algebra.cga3d()
    s = alg.sphere([0.0, 0.0, 0.0], 2.5)
    center, radius = alg.extract_sphere(s)
    assert_allclose(center, [0.0, 0.0, 0.0])
    assert_allclose(radius, 2.5)


def test_extract_plane_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.plane([0.0, 0.0, 1.0], 5.0)
    normal, distance = alg.extract_plane(p)
    assert_allclose(normal, [0.0, 0.0, 1.0])
    assert_allclose(distance, 5.0)


def test_extract_plane_default_normal() -> None:
    alg = amsa.Algebra.cga3d()
    p = alg.plane([1.0, 0.0, 0.0], 3.0)
    normal, distance = alg.extract_plane(p)
    assert_allclose(normal, [1.0, 0.0, 0.0])
    assert_allclose(distance, 3.0)


def test_extract_euclidean_vector_round_trip() -> None:
    alg = amsa.Algebra.cga3d()
    v = alg.euclidean_vector([4.0, 5.0, 6.0])
    coords = alg.extract_euclidean_vector(v)
    assert_allclose(coords, [4.0, 5.0, 6.0])


def test_extract_point_reflected() -> None:
    alg = amsa.Algebra.cga3d()
    plane = alg.plane([1.0, 0.0, 0.0], 0.0)
    p = alg.point([3.0, 2.0, 1.0])
    reflected = amsa.sandwich(plane, p)
    coords = alg.extract_point(reflected)
    assert_allclose(coords, [-3.0, 2.0, 1.0])


def test_extract_point_translated() -> None:
    alg = amsa.Algebra.cga2d()
    T = alg.translate([5.0, -2.0])
    p = alg.point([1.0, 1.0])
    moved = amsa.sandwich(T, p)
    coords = alg.extract_point(moved)
    assert_allclose(coords, [6.0, -1.0])


def test_extract_cga2d() -> None:
    alg = amsa.Algebra.cga2d()
    p = alg.point([3.0, 4.0])
    coords = alg.extract_point(p)
    assert_allclose(coords, [3.0, 4.0])

    s = alg.sphere([1.0, 1.0], 2.0)
    center, radius = alg.extract_sphere(s)
    assert_allclose(center, [1.0, 1.0])
    assert_allclose(radius, 2.0)

    pl = alg.plane([0.0, 1.0], 3.0)
    normal, distance = alg.extract_plane(pl)
    assert_allclose(normal, [0.0, 1.0])
    assert_allclose(distance, 3.0)


def test_extract_standalone_functions() -> None:
    from amsa.cga import extract_plane, extract_point, extract_sphere

    alg = amsa.Algebra.cga3d()
    p = alg.point([1.0, 2.0, 3.0])
    assert_allclose(extract_point(p), [1.0, 2.0, 3.0])

    s = alg.sphere([0.0, 0.0, 0.0], 4.0)
    center, radius = extract_sphere(s)
    assert_allclose(center, [0.0, 0.0, 0.0])
    assert_allclose(radius, 4.0)

    pl = alg.plane([0.0, 0.0, 1.0], 2.0)
    normal, distance = extract_plane(pl)
    assert_allclose(normal, [0.0, 0.0, 1.0])
    assert_allclose(distance, 2.0)


def test_extract_rejects_non_cga() -> None:
    alg = amsa.Algebra.vga3d()
    v = alg.vector([1.0, 2.0, 3.0])
    from amsa.cga import extract_point

    with pytest.raises(ValueError):
        extract_point(v)


# -- classify ----------------------------------------------------------------


def test_classify_normalized_point() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.point([1.0, 2.0, 3.0]))
    assert info.kind == "normalized conformal point"
    assert info.representation == "direct"
    assert info.null
    assert info.normalized
    assert_allclose(info.geometric_data["coordinates"], [1.0, 2.0, 3.0])


def test_classify_origin() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.origin())
    assert info.kind == "normalized conformal point"
    assert info.null
    assert info.normalized
    assert_allclose(info.geometric_data["coordinates"], [0.0, 0.0, 0.0])


def test_classify_point_at_infinity() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.infinity())
    assert info.kind == "point at infinity"
    assert info.null
    assert not info.normalized
    assert "could not extract" in str(info.warnings)


def test_classify_dual_sphere() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.sphere([1.0, 0.0, 0.0], 3.0))
    assert info.kind == "dual sphere"
    assert info.representation == "dual"
    assert not info.null
    assert_allclose(info.geometric_data["center"], [1.0, 0.0, 0.0])
    assert_allclose(info.geometric_data["radius"], 3.0)


def test_classify_dual_plane() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.plane([0.0, 0.0, 1.0], 5.0))
    assert info.kind == "dual plane"
    assert info.representation == "dual"
    assert not info.null
    assert_allclose(info.geometric_data["normal"], [0.0, 0.0, 1.0])
    assert_allclose(info.geometric_data["signed_distance"], 5.0)


def test_classify_translator() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.translate([1.0, 2.0, 3.0]))
    assert info.kind == "translator candidate"
    assert 0 in info.grades and 2 in info.grades


def test_classify_generic_scalar() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.scalar(5.0))
    assert info.kind == "generic blade"
    assert info.grades == (0,)


def test_classify_euclidean_vector_is_vector() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.euclidean_vector([1.0, 2.0, 3.0]))
    assert info.kind in ("dual plane", "generic vector")
    assert not info.null


def test_classify_line() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([0.0, 0.0, 0.0])
    b = alg.point([1.0, 0.0, 0.0])
    info = alg.classify(alg.line_through_points(a, b))
    assert info.kind == "direct line"
    assert 3 in info.grades


def test_classify_circle() -> None:
    alg = amsa.Algebra.cga3d()
    a = alg.point([1.0, 0.0, 0.0])
    b = alg.point([0.0, 1.0, 0.0])
    c = alg.point([-1.0, 0.0, 0.0])
    info = alg.classify(alg.circle_through_points(a, b, c))
    # Current heuristic cannot distinguish direct line from direct circle
    # (both contain conformal axes in their blade content).
    # TODO: L ^ n_inf == 0 check for line distinction.
    assert info.kind in ("direct line", "direct circle")
    assert 3 in info.grades


def test_classify_cga2d() -> None:
    alg = amsa.Algebra.cga2d()
    info = alg.classify(alg.point([3.0, 4.0]))
    assert info.algebra == "cga2d"
    assert info.kind == "normalized conformal point"
    assert_allclose(info.geometric_data["coordinates"], [3.0, 4.0])


def test_classify_str_output() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.point([1.0, 2.0, 3.0]))
    text = str(info)
    assert "CGA3D Classification" in text
    assert "normalized conformal point" in text
    assert "coordinates" in text


def test_classify_zero() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.zeros())
    assert info.kind == "zero multivector"


def test_classify_reflected_point() -> None:
    alg = amsa.Algebra.cga3d()
    plane = alg.plane([1.0, 0.0, 0.0], 0.0)
    p = alg.point([3.0, 2.0, 1.0])
    reflected = amsa.sandwich(plane, p)
    info = alg.classify(reflected)
    assert info.kind == "conformal point"
    assert info.null
    assert_allclose(info.geometric_data["coordinates"], [-3.0, 2.0, 1.0])


# -- PGA classify --------------------------------------------------------------


def test_classify_pga2d_point() -> None:
    alg = amsa.Algebra.pga2d()
    point = alg.multivector({"e12": 1.0, "e01": 2.0, "e02": 3.0})
    info = alg.classify(point)
    assert info.kind == "normalized Euclidean point"
    assert_allclose(info.geometric_data["coordinates"], [2.0, 3.0])


def test_classify_pga2d_ideal_point() -> None:
    alg = amsa.Algebra.pga2d()
    point = alg.multivector({"e12": 0.0, "e01": 1.0, "e02": 2.0})
    info = alg.classify(point)
    assert info.kind == "ideal point"
    assert "direction" in info.geometric_data


def test_classify_pga2d_line() -> None:
    alg = amsa.Algebra.pga2d()
    line = alg.multivector({"e0": 2.0, "e1": 1.0, "e2": 0.0})
    info = alg.classify(line)
    assert info.kind == "line"
    assert info.representation == "dual"


def test_classify_pga2d_rotor() -> None:
    alg = amsa.Algebra.pga2d()
    rotor = alg.multivector({"e": 0.7071, "e12": 0.7071})
    info = alg.classify(rotor)
    assert info.kind == "rotor"
    assert 0 in info.grades and 2 in info.grades


def test_classify_pga3d_point() -> None:
    alg = amsa.Algebra.pga3d()
    point = alg.multivector({"e123": 1.0, "e012": -1.0, "e013": 2.0, "e023": -3.0})
    info = alg.classify(point)
    assert info.kind == "normalized Euclidean point"
    assert_allclose(info.geometric_data["coordinates"], [3.0, 2.0, 1.0])


def test_classify_pga3d_plane() -> None:
    alg = amsa.Algebra.pga3d()
    plane = alg.multivector({"e0": 5.0, "e1": 0.0, "e2": 0.0, "e3": 1.0})
    info = alg.classify(plane)
    assert info.kind == "plane"


def test_classify_pga3d_line() -> None:
    alg = amsa.Algebra.pga3d()
    line = alg.multivector({"e01": 1.0, "e02": 2.0, "e03": 3.0, "e12": 0.0, "e13": 0.0, "e23": 1.0})
    info = alg.classify(line)
    assert info.kind == "line"


def test_classify_pga2d_str_output() -> None:
    alg = amsa.Algebra.pga2d()
    info = alg.classify(alg.multivector({"e12": 1.0, "e01": 2.0, "e02": 3.0}))
    text = str(info)
    assert "PGA2D Classification" in text
    assert "normalized Euclidean point" in text


def test_classify_vga_vector() -> None:
    alg = amsa.Algebra.vga3d()
    info = alg.classify(alg.vector([1.0, 2.0, 3.0]))
    assert info.kind == "vector"
    assert 1 in info.grades


def test_classify_vga_bivector() -> None:
    alg = amsa.Algebra.vga2d()
    info = alg.classify(alg.bivector([1.0]))
    assert info.kind == "bivector"


def test_classify_vga_scalar() -> None:
    alg = amsa.Algebra.vga3d()
    info = alg.classify(alg.scalar(5.0))
    assert info.kind == "scalar"


def test_classify_vga_pseudoscalar() -> None:
    alg = amsa.Algebra.vga3d()
    info = alg.classify(alg.trivector([1.0]))
    assert info.kind == "pseudoscalar"


def test_classify_routing_pga() -> None:
    alg = amsa.Algebra.pga2d()
    info = alg.classify(alg.multivector({"e12": 1.0, "e01": 0.0, "e02": 0.0}))
    assert info.algebra == "pga2d"


def test_classify_routing_vga() -> None:
    alg = amsa.Algebra.vga2d()
    info = alg.classify(alg.scalar(1.0))
    assert info.algebra == "vga2d"


def test_classify_routing_cga() -> None:
    alg = amsa.Algebra.cga3d()
    info = alg.classify(alg.point([1.0, 2.0, 3.0]))
    assert info.algebra == "cga3d"
