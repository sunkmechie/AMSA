import numpy as np
import pytest

from amsa import Algebra, MVArray, MVLayout, involute, neg, pga2d, reverse, vga
from amsa.storage import CSRStorage, DenseStorage, to_csr_storage, to_dense_storage

from ._utils import assert_allclose


def test_dense_layout_covers_full_basis() -> None:
    spec = vga(2)
    layout = MVLayout.dense(spec)
    assert layout.blades == (0, 1, 2, 3)
    assert layout.kind == "dense"
    assert layout.grades == (0, 1, 2)


def test_grade_layout_is_grade_packed() -> None:
    spec = vga(3)
    layout = MVLayout.grade(spec, 1, 2)
    assert layout.blades == (1, 2, 4, 3, 5, 6)
    assert layout.grades == (1, 2)


def test_sparse_layout_requires_unique_blades() -> None:
    spec = vga(2)
    with pytest.raises(ValueError):
        MVLayout.sparse_pattern(spec, (1, 1))


def test_empty_sparse_layout_is_allowed_for_zero_support() -> None:
    spec = vga(2)
    layout = MVLayout.sparse_pattern(spec, ())
    assert layout.size == 0


def test_mvarray_validates_shape_against_layout() -> None:
    spec = vga(2)
    layout = MVLayout.grade(spec, 1)
    values = np.zeros((5, layout.size))
    mv = MVArray(algebra=spec, layout=layout, values=values)
    assert mv.batch_shape == (5,)

    with pytest.raises(ValueError):
        MVArray(algebra=spec, layout=layout, values=np.zeros((layout.size + 1,)))


def test_mvarray_accepts_dense_storage_object() -> None:
    spec = vga(2)
    layout = MVLayout.grade(spec, 1)
    storage = DenseStorage.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

    mv = MVArray(algebra=spec, layout=layout, storage=storage)

    assert mv.storage_kind == "dense"
    assert mv.batch_shape == (2,)
    np.testing.assert_array_equal(mv.values, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_csr_storage_preserves_flattened_batch_shape() -> None:
    storage = CSRStorage(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([0, 2, 1, 0, 2]),
        np.array([0, 2, 2, 3, 5]),
        batch_shape=(2, 2),
        width=3,
    )

    assert storage.kind == "csr"
    assert storage.batch_shape == (2, 2)
    assert storage.row_count == 4
    assert storage.width == 3
    assert storage.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        storage.as_dense(),
        np.array(
            [
                [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
                [[0.0, 3.0, 0.0], [4.0, 0.0, 5.0]],
            ]
        ),
    )


def test_csr_storage_zeros_supports_empty_sparse_layouts() -> None:
    storage = CSRStorage.zeros(width=0, batch_shape=(2, 3))

    assert storage.batch_shape == (2, 3)
    assert storage.row_count == 6
    assert storage.width == 0
    assert storage.data.size == 0
    assert storage.indices.size == 0
    np.testing.assert_array_equal(storage.indptr, np.zeros(7, dtype=np.intp))
    np.testing.assert_array_equal(storage.as_dense(), np.zeros((2, 3, 0)))


def test_csr_storage_copy_detaches_underlying_arrays() -> None:
    storage = CSRStorage(
        np.array([1.0, 2.0]),
        np.array([0, 2]),
        np.array([0, 1, 2]),
        batch_shape=(2,),
        width=3,
    )

    copied = storage.copy()

    np.testing.assert_array_equal(copied.as_dense(), storage.as_dense())
    assert not np.shares_memory(copied.data, storage.data)
    assert not np.shares_memory(copied.indices, storage.indices)
    assert not np.shares_memory(copied.indptr, storage.indptr)


def test_dense_and_csr_storage_round_trip_preserves_coefficients() -> None:
    dense = DenseStorage.from_array(
        np.array(
            [
                [[1.0, 0.0, -2.0], [0.0, 0.0, 0.0]],
                [[3.5, 0.0, 0.0], [0.0, 4.0, 5.0]],
            ]
        )
    )

    csr = to_csr_storage(dense)
    restored = to_dense_storage(csr)

    assert csr.kind == "csr"
    assert csr.batch_shape == (2, 2)
    np.testing.assert_array_equal(restored.array, dense.array)


def test_csr_storage_validates_flattened_row_count() -> None:
    with pytest.raises(ValueError, match="row_count"):
        CSRStorage(
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            np.array([0, 2, 2]),
            batch_shape=(2, 2),
            width=3,
        )


def test_csr_storage_requires_strictly_increasing_row_indices() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CSRStorage(
            np.array([1.0, 2.0]),
            np.array([1, 1]),
            np.array([0, 2]),
            batch_shape=(),
            width=3,
        )


def test_mvarray_accepts_csr_storage_object() -> None:
    spec = vga(2)
    layout = MVLayout.sparse_pattern(spec, (1, 3), name="support")
    storage = CSRStorage(
        np.array([2.0, -3.0]),
        np.array([0, 1]),
        np.array([0, 2]),
        batch_shape=(),
        width=layout.size,
    )

    mv = MVArray(algebra=spec, layout=layout, storage=storage)

    assert mv.storage_kind == "csr"
    np.testing.assert_array_equal(mv.values, np.array([2.0, -3.0]))


def test_csr_to_layout_preserves_csr_backend() -> None:
    spec = vga(2)
    source_layout = MVLayout.sparse_pattern(spec, (1, 3), name="support")
    storage = CSRStorage(
        np.array([2.0, -3.0, 4.0]),
        np.array([0, 1, 0]),
        np.array([0, 2, 3]),
        batch_shape=(2,),
        width=source_layout.size,
    )
    mv = MVArray(algebra=spec, layout=source_layout, storage=storage)

    projected = mv.to_layout(MVLayout.dense(spec))

    assert projected.storage_kind == "csr"
    np.testing.assert_array_equal(
        projected.values,
        np.array([[0.0, 2.0, 0.0, -3.0], [0.0, 4.0, 0.0, 0.0]]),
    )


def test_as_dense_converts_dense_layout_csr_storage_to_dense_storage() -> None:
    spec = vga(2)
    dense_layout = MVLayout.dense(spec)
    storage = CSRStorage(
        np.array([1.5, -2.0]),
        np.array([1, 3]),
        np.array([0, 2]),
        batch_shape=(),
        width=dense_layout.size,
    )
    mv = MVArray(algebra=spec, layout=dense_layout, storage=storage)

    dense = mv.as_dense()

    assert dense.storage_kind == "dense"
    np.testing.assert_array_equal(dense.values, np.array([0.0, 1.5, 0.0, -2.0]))


def test_csr_component_and_grade_selection_work_without_breakage() -> None:
    spec = vga(2)
    layout = MVLayout.sparse_pattern(spec, (1, 3), name="support")
    storage = CSRStorage(
        np.array([2.0, -3.0, 4.0]),
        np.array([0, 1, 0]),
        np.array([0, 2, 3]),
        batch_shape=(2,),
        width=layout.size,
    )
    mv = MVArray(algebra=spec, layout=layout, storage=storage)

    graded = mv.grade(2)

    assert_allclose(mv.component("e1"), np.array([2.0, 4.0]))
    assert_allclose(mv.component("e2"), np.array([0.0, 0.0]))
    assert graded.storage_kind == "csr"
    assert graded.layout.blades == (3,)
    assert_allclose(graded.values, np.array([[-3.0], [0.0]]))


def test_scalar_multiplication_preserves_csr_storage_and_canonical_zero() -> None:
    spec = vga(2)
    layout = MVLayout.sparse_pattern(spec, (1, 3), name="support")
    storage = CSRStorage(
        np.array([2.0, -3.0]),
        np.array([0, 1]),
        np.array([0, 2]),
        batch_shape=(),
        width=layout.size,
    )
    mv = MVArray(algebra=spec, layout=layout, storage=storage)

    scaled = -2 * mv
    zeroed = mv * 0

    assert scaled.storage_kind == "csr"
    np.testing.assert_array_equal(scaled.values, np.array([-4.0, 6.0]))
    assert zeroed.storage_kind == "csr"
    assert isinstance(zeroed.storage, CSRStorage)
    assert zeroed.storage.data.size == 0
    np.testing.assert_array_equal(zeroed.values, np.array([0.0, 0.0]))


def test_exact_zero_csr_multivector_handles_empty_sparse_layout() -> None:
    spec = vga(2)
    layout = MVLayout.sparse_pattern(spec, ())
    mv = MVArray(algebra=spec, layout=layout, storage=CSRStorage.zeros(width=0, batch_shape=(2,)))

    projected = mv.to_layout(MVLayout.dense(spec))
    dense = mv.as_dense()

    assert projected.storage_kind == "csr"
    np.testing.assert_array_equal(projected.values, np.zeros((2, 4)))
    assert dense.storage_kind == "dense"
    np.testing.assert_array_equal(dense.values, np.zeros((2, 4)))


def test_mvarray_requires_exactly_one_storage_source() -> None:
    spec = vga(2)
    layout = MVLayout.grade(spec, 1)
    storage = DenseStorage.from_array(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        MVArray(algebra=spec, layout=layout)

    with pytest.raises(ValueError):
        MVArray(algebra=spec, layout=layout, values=np.array([1.0, 2.0]), storage=storage)


def test_multivector_mapping_constructor_builds_sparse_layout() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": 2.0, "e12": -3.0})
    assert mv.layout.blades == (1, 3)
    assert mv.component("e1") == 2.0
    assert mv.component("e2") == 0.0


def test_to_layout_projects_into_dense_coordinates() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": 2.0, "e12": -3.0})
    dense = mv.as_dense()
    np.testing.assert_array_equal(dense.values, np.array([0.0, 2.0, 0.0, -3.0]))


def test_geometric_product_matches_vector_identity() -> None:
    algebra = Algebra(vga(2))
    e1 = algebra.blade("e1")
    e2 = algebra.blade("e2")
    product = e1 * e2
    assert product.layout.blades == (3,)
    assert product.component("e12") == 1.0

    reverse_product = e2 * e1
    assert reverse_product.component("e12") == -1.0


def test_geometric_product_handles_null_square() -> None:
    algebra = Algebra(pga2d())
    e0 = algebra.blade("e0")
    product = e0 * e0
    assert product.layout.size == 0
    assert product.component("e") == 0.0


def test_addition_unions_sparse_support() -> None:
    algebra = Algebra(vga(2))
    x = algebra.multivector({"e1": 1.0})
    y = algebra.multivector({"e2": 2.0})
    z = x + y
    assert z.layout.blades == (1, 2)
    assert z.component("e1") == 1.0
    assert z.component("e2") == 2.0


def test_addition_and_subtraction_support_scalars() -> None:
    algebra = Algebra(vga(2))
    x = algebra.multivector({"e1": 1.0})

    z = x + 2.0
    assert z.component("e") == 2.0
    assert z.component("e1") == 1.0

    z = 3.0 + x
    assert z.component("e") == 3.0
    assert z.component("e1") == 1.0

    z = x - 2.0
    assert z.component("e") == -2.0
    assert z.component("e1") == 1.0

    z = 2.0 - x
    assert z.component("e") == 2.0
    assert z.component("e1") == -1.0


def test_batch_mapping_values_are_broadcast() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": np.array([1.0, 2.0]), "e2": 3.0})
    assert mv.batch_shape == (2,)
    assert_allclose(mv.component("e2"), np.array([3.0, 3.0]))


def test_algebra_add_and_sub_helpers_delegate_to_operator_layer() -> None:
    algebra = Algebra(vga(2))
    lhs = algebra.multivector({"e1": 1.0})
    rhs = algebra.multivector({"e2": 2.0})
    added = algebra.add(lhs, rhs)
    subtracted = algebra.sub(lhs, rhs)
    assert added.component("e1") == 1.0
    assert added.component("e2") == 2.0
    assert subtracted.component("e1") == 1.0
    assert subtracted.component("e2") == -2.0


def test_algebra_multivector_rejects_scalar_input() -> None:
    algebra = Algebra(vga(2))

    with pytest.raises(ValueError, match="at least one dimension"):
        algebra.multivector(2.5, layout=algebra.grade_layout(0))


def test_step5_policy_keeps_dense_as_default_constructor_backend() -> None:
    algebra = Algebra(vga(2))

    mv = algebra.multivector({"e1": 2.0, "e12": -3.0})
    zeros = algebra.zeros(layout=algebra.sparse_layout((1, 3)))

    assert mv.storage_kind == "dense"
    assert zeros.storage_kind == "dense"


def test_explicit_csr_backend_is_available_on_constructors() -> None:
    algebra = Algebra(vga(3))

    mv = algebra.multivector({"e1": np.array([0.0, 2.0]), "e23": 3.0}, backend="csr")
    scalar = algebra.scalar(2.0, backend="csr")
    zeros = algebra.zeros(layout=algebra.grade_layout(2), batch_shape=(2,), backend="csr")

    assert mv.storage_kind == "csr"
    assert scalar.storage_kind == "csr"
    assert zeros.storage_kind == "csr"
    assert_allclose(mv.values, np.array([[0.0, 3.0], [2.0, 3.0]]))
    assert_allclose(scalar.values, np.array([2.0]))
    assert_allclose(zeros.values, np.zeros((2, 3)))


def test_imported_multivector_can_be_rewrapped_with_requested_backend() -> None:
    algebra = Algebra(vga(2))
    dense = algebra.multivector({"e1": 2.0, "e12": -3.0})

    csr = algebra.multivector(dense, backend="csr")
    dense_again = algebra.multivector(csr, backend="dense")

    assert dense.storage_kind == "dense"
    assert csr.storage_kind == "csr"
    assert dense_again.storage_kind == "dense"
    assert csr.layout == dense.layout
    assert_allclose(csr.values, dense.values)
    assert_allclose(dense_again.values, dense.values)


def test_mvarray_with_storage_round_trip_preserves_layout_and_values() -> None:
    algebra = Algebra(vga(3))
    mv = algebra.multivector({"e1": 1.0, "e23": -2.0})

    csr = mv.with_storage("csr")
    dense = csr.with_storage("dense")

    assert csr.storage_kind == "csr"
    assert dense.storage_kind == "dense"
    assert csr.layout == mv.layout
    assert dense.layout == mv.layout
    assert_allclose(csr.values, mv.values)
    assert_allclose(dense.values, mv.values)


def test_unary_ops_preserve_csr_storage() -> None:
    spec = vga(3)
    layout = MVLayout.sparse_pattern(spec, (1, 3, 7), name="support")
    storage = CSRStorage(
        np.array([2.0, -3.0, 4.0, -5.0]),
        np.array([0, 2, 1, 2]),
        np.array([0, 2, 4]),
        batch_shape=(2,),
        width=layout.size,
    )
    mv = MVArray(algebra=spec, layout=layout, storage=storage)

    negated = neg(mv)
    reversed_mv = reverse(mv)
    involuted = involute(mv)

    assert negated.storage_kind == "csr"
    assert reversed_mv.storage_kind == "csr"
    assert involuted.storage_kind == "csr"
    assert_allclose(negated.values, np.array([[-2.0, 0.0, 3.0], [0.0, -4.0, 5.0]]))
    assert_allclose(reversed_mv.values, np.array([[2.0, 0.0, 3.0], [0.0, -4.0, 5.0]]))
    assert_allclose(involuted.values, np.array([[-2.0, 0.0, 3.0], [0.0, 4.0, 5.0]]))


def test_reverse_and_involute_sign_rules() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": 1.0, "e12": 2.0})
    assert_allclose(mv.reverse().values, np.array([1.0, -2.0]))
    assert_allclose(mv.involute().values, np.array([-1.0, 2.0]))
