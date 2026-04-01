import numpy as np
import pytest

from amsa import (
    Algebra,
    MVLayout,
    dual,
    geometric_product,
    inner_product,
    outer_product,
    pga2d,
    poincare_dual,
    undual,
    vga2d,
    vga3d,
)
from amsa.mv import MVArray
from amsa.plans import plan_binary_product
from amsa.specs import grade_of_blade
from amsa.storage import CSRStorage

from ._utils import assert_mv_allclose


def _keep_term(kind: str, lhs_blade: int, rhs_blade: int, out_blade: int) -> bool:
    if kind == "geometric":
        return True
    if kind == "outer":
        return grade_of_blade(out_blade) == grade_of_blade(lhs_blade) + grade_of_blade(rhs_blade)
    if kind == "inner":
        return grade_of_blade(out_blade) == abs(
            grade_of_blade(lhs_blade) - grade_of_blade(rhs_blade)
        )
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
    if len(blades) == lhs.algebra.blade_count:
        layout = MVLayout.dense(lhs.algebra)
    else:
        layout = MVLayout.sparse_pattern(lhs.algebra, blades, name=kind)
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


def test_dual_maps_vga3d_plane_bivector_to_normal_vector() -> None:
    algebra = Algebra.vga3d()
    e12 = algebra.blade("e12")

    normal = dual(e12)

    assert normal.layout.blades == (4,)
    assert normal.component("e3") == 1.0
    assert_mv_allclose(normal.undual(), e12)


def test_dual_and_undual_round_trip_dense_vga2d_multivector() -> None:
    algebra = Algebra.vga2d()
    mv = algebra.multivector({"e": 2.0, "e1": -3.0, "e12": 4.0})

    assert_mv_allclose(undual(dual(mv)), mv)
    assert_mv_allclose(dual(undual(mv)), mv)


def test_dual_preserves_csr_storage_for_sparse_support() -> None:
    algebra = Algebra.vga3d()
    mv = algebra.multivector({"e12": np.array([1.0, -2.0]), "e23": 3.0}, backend="csr")

    dual_mv = mv.dual()
    restored = dual_mv.undual()

    assert dual_mv.storage_kind == "csr"
    assert dual_mv.layout.blades == (1, 4)
    assert_mv_allclose(dual_mv, algebra.multivector({"e1": 3.0, "e3": np.array([1.0, -2.0])}))
    assert_mv_allclose(restored, mv)


def test_dual_rejects_degenerate_algebras_with_noninvertible_pseudoscalar() -> None:
    algebra = Algebra.pga2d()
    line = algebra.blade("e12")

    with pytest.raises(ValueError, match="invertible pseudoscalar"):
        line.dual()


def test_poincare_dual_uses_right_complement_signs_in_vga2d() -> None:
    algebra = Algebra.vga2d()

    assert poincare_dual(algebra.scalar(1.0)).component("e12") == 1.0
    assert poincare_dual(algebra.blade("e1")).component("e2") == 1.0
    assert poincare_dual(algebra.blade("e2")).component("e1") == -1.0
    assert poincare_dual(algebra.blade("e12")).component("e") == 1.0


def test_poincare_dual_is_distinct_from_metric_dual_in_vga2d() -> None:
    algebra = Algebra.vga2d()
    e1 = algebra.blade("e1")

    assert e1.dual().component("e2") == -1.0
    assert e1.poincare_dual().component("e2") == 1.0


def test_poincare_dual_satisfies_basis_complement_identity_in_pga3d() -> None:
    algebra = Algebra.pga3d()
    pseudoscalar = algebra.spec.blade_name(algebra.spec.pseudoscalar_blade)

    for blade in range(algebra.spec.blade_count):
        basis = algebra.blade(blade)
        complement = basis.poincare_dual()
        joined = basis ^ complement

        assert joined.layout.blades == (algebra.spec.pseudoscalar_blade,)
        assert joined.component(pseudoscalar) == 1.0


def test_poincare_dual_round_trips_and_preserves_csr_storage_in_pga2d() -> None:
    algebra = Algebra.pga2d()
    mv = algebra.multivector({"e0": np.array([1.0, -2.0]), "e12": 3.0}, backend="csr")

    dual_mv = mv.poincare_dual()
    restored = dual_mv.poincare_undual()

    assert dual_mv.storage_kind == "csr"
    assert dual_mv.layout.blades == (1, 6)
    assert_mv_allclose(dual_mv, algebra.multivector({"e0": 3.0, "e12": np.array([1.0, -2.0])}))
    assert_mv_allclose(restored, mv)


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


def test_reference_execution_consumes_csr_inputs_without_dense_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algebra = Algebra.vga2d()
    layout = MVLayout.sparse_pattern(algebra.spec, (1, 2, 3), name="support")

    lhs = MVArray(
        algebra=algebra.spec,
        layout=layout,
        storage=CSRStorage(
            np.array([1.0, 2.0, -3.0]),
            np.array([0, 2, 1]),
            np.array([0, 2, 3]),
            batch_shape=(2,),
            width=layout.size,
        ),
    )
    rhs = MVArray(
        algebra=algebra.spec,
        layout=layout,
        storage=CSRStorage(
            np.array([4.0, -5.0, 6.0]),
            np.array([1, 2, 0]),
            np.array([0, 2, 3]),
            batch_shape=(2,),
            width=layout.size,
        ),
    )

    def fail_as_dense(self: CSRStorage) -> np.ndarray:
        raise AssertionError("binary execution should not densify CSR inputs via as_dense()")

    expected = _naive_binary_product(lhs.copy(), rhs.copy(), kind="geometric")
    monkeypatch.setattr(CSRStorage, "as_dense", fail_as_dense)

    actual = geometric_product(lhs, rhs)

    assert actual.storage_kind == "dense"
    assert_mv_allclose(actual, expected)


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        ("geometric", geometric_product),
        ("outer", outer_product),
        ("inner", inner_product),
    ],
)
def test_mixed_dense_and_csr_products_match_dense_reference(kind, operation) -> None:
    algebra = Algebra.vga3d()
    lhs = algebra.multivector({"e1": np.array([1.0, -2.0]), "e23": 3.0}, backend="dense")
    rhs = algebra.multivector({"e": 2.0, "e2": np.array([0.5, 1.5]), "e123": -4.0}, backend="csr")

    actual = operation(lhs, rhs)
    expected = _naive_binary_product(
        lhs.with_storage("dense"),
        rhs.with_storage("dense"),
        kind=kind,
    )

    assert_mv_allclose(actual, expected)


def test_mixed_dense_and_csr_add_sub_match_dense_reference() -> None:
    algebra = Algebra.vga2d()
    lhs = algebra.multivector({"e1": np.array([1.0, 2.0]), "e12": -3.0}, backend="csr")
    rhs = algebra.multivector({"e": 2.0, "e2": 4.0}, backend="dense")

    added = lhs + rhs
    subtracted = lhs - rhs
    lhs_dense = lhs.with_storage("dense")
    rhs_dense = rhs.with_storage("dense")

    assert_mv_allclose(added, lhs_dense + rhs_dense)
    assert_mv_allclose(subtracted, lhs_dense - rhs_dense)
