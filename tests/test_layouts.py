import numpy as np
import pytest

from amsa import Algebra, MVArray, MVLayout, pga2d, vga
from amsa.storage import CSRStorage, DenseStorage

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


def test_reverse_and_involute_sign_rules() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": 1.0, "e12": 2.0})
    assert_allclose(mv.reverse().values, np.array([1.0, -2.0]))
    assert_allclose(mv.involute().values, np.array([-1.0, 2.0]))
