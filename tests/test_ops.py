import numpy as np
import pytest

from amsa import Algebra, MVLayout, geometric_product, inner_product, outer_product, pga2d, vga2d, vga3d
from amsa.plans import plan_binary_product
from amsa.specs import grade_of_blade
from ._utils import assert_mv_allclose


def _keep_term(kind: str, lhs_blade: int, rhs_blade: int, out_blade: int) -> bool:
    if kind == "geometric":
        return True
    if kind == "outer":
        return grade_of_blade(out_blade) == grade_of_blade(lhs_blade) + grade_of_blade(rhs_blade)
    if kind == "inner":
        return grade_of_blade(out_blade) == abs(grade_of_blade(lhs_blade) - grade_of_blade(rhs_blade))
    raise ValueError(f"Unsupported operator kind: {kind}")


def _naive_binary_product(lhs, rhs, *, kind: str):
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    lhs_values = np.broadcast_to(lhs.values, batch_shape + (lhs.layout.size,))
    rhs_values = np.broadcast_to(rhs.values, batch_shape + (rhs.layout.size,))

    support: set[int] = set()
    accumulators: dict[int, np.ndarray] = {}
    dtype = np.result_type(lhs.dtype, rhs.dtype)
    zero = np.zeros(batch_shape, dtype=dtype)

    for lhs_index, lhs_blade in enumerate(lhs.layout.blades):
        lhs_component = lhs_values[..., lhs_index]
        for rhs_index, rhs_blade in enumerate(rhs.layout.blades):
            coefficient, out_blade = lhs.algebra.blade_product(lhs_blade, rhs_blade)
            if coefficient == 0:
                continue
            if not _keep_term(kind, lhs_blade, rhs_blade, out_blade):
                continue

            support.add(out_blade)
            contribution = coefficient * lhs_component * rhs_values[..., rhs_index]
            if out_blade in accumulators:
                accumulators[out_blade] = accumulators[out_blade] + contribution
            else:
                accumulators[out_blade] = zero + contribution

    blades = tuple(sorted(support))
    layout = MVLayout.dense(lhs.algebra) if len(blades) == lhs.algebra.blade_count else MVLayout.sparse_pattern(lhs.algebra, blades, name=kind)
    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)
    for index, blade in enumerate(layout.blades):
        result[..., index] = accumulators[blade]
    return type(lhs)(algebra=lhs.algebra, layout=layout, values=result)


@pytest.mark.parametrize(
    ("factory", "kind", "operation"),
    [
        (vga2d, "geometric", geometric_product),
        (vga2d, "outer", outer_product),
        (vga2d, "inner", inner_product),
        (vga3d, "geometric", geometric_product),
        (vga3d, "outer", outer_product),
        (vga3d, "inner", inner_product),
        (pga2d, "geometric", geometric_product),
        (pga2d, "outer", outer_product),
        (pga2d, "inner", inner_product),
    ],
)
def test_planned_products_match_naive_reference(factory, kind, operation) -> None:
    algebra = Algebra(factory())
    lhs = algebra.multivector({1: np.array([1.0, -2.0]), 2: 3.0, 3: -1.5})
    rhs = algebra.multivector({0: 2.0, 1: np.array([0.5, 1.5]), 3: 4.0})

    actual = operation(lhs, rhs)
    expected = _naive_binary_product(lhs, rhs, kind=kind)
    assert_mv_allclose(actual, expected)


def test_product_plans_are_cached_by_operator_and_layout_support() -> None:
    spec = vga3d()
    lhs = MVLayout.grade(spec, 1, 2)
    rhs = MVLayout.sparse_pattern(spec, (0, 1, 3, 7), name="rhs")

    first = plan_binary_product(lhs, rhs, "geometric")
    second = plan_binary_product(lhs, rhs, "geometric")

    assert first is second
    assert first.kind == "geometric"
    assert first.lhs_blades == lhs.blades
    assert first.rhs_blades == rhs.blades


def test_outer_and_inner_split_vector_product_in_vga2d() -> None:
    algebra = Algebra.vga2d()
    u = algebra.vector([1.0, 2.0])
    v = algebra.vector([3.0, -4.0])

    assert_mv_allclose(u * v, (u | v) + (u ^ v))
    assert (u | v).component("e") == -5.0
    assert (u ^ v).component("e12") == -10.0


def test_outer_and_inner_handle_basis_cases_in_vga3d() -> None:
    algebra = Algebra.vga3d()
    e2 = algebra.blade("e2")
    e3 = algebra.blade("e3")
    e12 = algebra.blade("e12")
    e23 = algebra.blade("e23")

    assert (e12 | e2).component("e1") == 1.0
    assert (e12 ^ e3).component("e123") == 1.0
    assert (e12 ^ e23).layout.size == 0


def test_outer_and_inner_handle_degenerate_pga2d_cases() -> None:
    algebra = Algebra.pga2d()
    e0 = algebra.blade("e0")
    e1 = algebra.blade("e1")

    assert (e0 | e0).layout.size == 0
    assert (e0 ^ e0).layout.size == 0
    assert (e0 ^ e1).component("e01") == 1.0


def test_named_presets_and_grade_helpers_cover_common_robotics_shapes() -> None:
    pga = Algebra.from_name("2DPGA")
    assert pga.signature == (0, 1, 1)

    algebra = Algebra.vga3d()
    rotor = algebra.even([1.0, 0.0, 0.5, -0.25])
    trivector = algebra.trivector([2.0])
    mixed = algebra.multivector({"e1": 1.0, "e12": 2.0, "e123": 3.0})
    projected = mixed.grade(1, 3)

    assert rotor.grades == (0, 2)
    assert rotor.layout.size == 4
    assert trivector.component("e123") == 2.0
    assert projected.layout.blades == (1, 7)
