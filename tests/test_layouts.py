import numpy as np
import pytest

from amsa import Algebra, MVArray, MVLayout, pga2d, vga
from tests._utils import assert_allclose


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


def test_batch_mapping_values_are_broadcast() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": np.array([1.0, 2.0]), "e2": 3.0})
    assert mv.batch_shape == (2,)
    assert_allclose(mv.component("e2"), np.array([3.0, 3.0]))


def test_reverse_and_involute_sign_rules() -> None:
    algebra = Algebra(vga(2))
    mv = algebra.multivector({"e1": 1.0, "e12": 2.0})
    assert_allclose(mv.reverse().values, np.array([1.0, -2.0]))
    assert_allclose(mv.involute().values, np.array([-1.0, 2.0]))
