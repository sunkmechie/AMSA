"""Regression tests for storage operations.

These tests lock down storage behavior (conversion, projection, scaling)
to provide a safety net for Pass 3 storage/backend separation refactoring.
"""

import numpy as np
import pytest

from amsa.storage import (
    CSRStorage,
    DenseStorage,
    add_storage,
    coefficient_magnitude_squared_storage,
    gather_storage_columns,
    project_storage,
    reweight_storage,
    row_scale_storage,
    scale_storage,
    storage_component,
    sub_storage,
    to_csr_storage,
    to_dense_storage,
)


def test_dense_to_csr_preserves_coefficients() -> None:
    """Converting dense to CSR must preserve all coefficient values."""
    dense = DenseStorage.from_array(
        np.array([[1.0, 0.0, -2.0], [0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [0.0, 4.0, 5.0]])
    )
    csr = to_csr_storage(dense)
    
    np.testing.assert_array_equal(csr.as_dense(), dense.as_dense())


def test_csr_to_dense_preserves_coefficients() -> None:
    """CSR to dense conversion must preserve all coefficient values."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0, 5.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    dense = to_dense_storage(csr)
    expected = np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]])
    np.testing.assert_array_equal(dense.as_dense(), expected)


def test_csr_to_csr_is_copy() -> None:
    """CSR to CSR conversion must return a deep copy."""
    csr = CSRStorage(
        data=np.array([1.0, 2.0]),
        indices=np.array([0, 1]),
        indptr=np.array([0, 2]),
        batch_shape=(),
        width=2,
    )
    csr_copy = to_csr_storage(csr)
    np.testing.assert_array_equal(csr_copy.as_dense(), csr.as_dense())
    assert csr_copy is not csr


def test_dense_to_dense_is_copy() -> None:
    """Dense to dense conversion must return a deep copy."""
    dense = DenseStorage.from_array(np.array([1.0, 2.0, 3.0]))
    dense_copy = to_dense_storage(dense)
    np.testing.assert_array_equal(dense_copy.as_dense(), dense.as_dense())
    assert dense_copy is not dense


def test_scale_dense_storage() -> None:
    """Scaling dense storage must multiply all coefficients."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    scaled = scale_storage(dense, 2.0)
    expected = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(scaled.as_dense(), expected)


def test_scale_csr_storage() -> None:
    """Scaling CSR storage must multiply all stored coefficients."""
    csr = CSRStorage(
        data=np.array([1.0, 2.0, 3.0]),
        indices=np.array([0, 1, 2]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    scaled = scale_storage(csr, 2.0)
    expected = np.array([[2.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    np.testing.assert_array_equal(scaled.as_dense(), expected)


def test_scale_by_zero_returns_zero_csr() -> None:
    """Scaling CSR by zero must return zero CSR storage."""
    csr = CSRStorage(
        data=np.array([1.0, 2.0]),
        indices=np.array([0, 1]),
        indptr=np.array([0, 2]),
        batch_shape=(),
        width=2,
    )
    scaled = scale_storage(csr, 0.0)
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(scaled.as_dense(), expected)
    assert scaled.width == 2


def test_reweight_dense_storage() -> None:
    """Reweighting dense storage must multiply columns by weights."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    weighted = reweight_storage(dense, np.array([2.0, 3.0, 4.0]))
    expected = np.array([[2.0, 6.0, 12.0], [8.0, 15.0, 24.0]])
    np.testing.assert_array_equal(weighted.as_dense(), expected)


def test_reweight_csr_storage() -> None:
    """Reweighting CSR storage must multiply stored columns by weights."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0, 5.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    weighted = reweight_storage(csr, np.array([2.0, 3.0, 4.0]))
    expected = np.array([[2.0, 0.0, 12.0], [0.0, 15.0, 0.0]])  # 1*2, 3*4, 5*3
    np.testing.assert_array_equal(weighted.as_dense(), expected)


def test_row_scale_dense_storage() -> None:
    """Row scaling dense storage must multiply each row by factor."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    scaled = row_scale_storage(dense, np.array([2.0, 3.0]))
    expected = np.array([[2.0, 4.0], [9.0, 12.0]])
    np.testing.assert_array_equal(scaled.as_dense(), expected)


def test_row_scale_csr_storage() -> None:
    """Row scaling CSR storage must multiply each row's entries by factor."""
    csr = CSRStorage(
        data=np.array([1.0, 2.0, 3.0, 4.0]),
        indices=np.array([0, 1, 0, 1]),
        indptr=np.array([0, 2, 4]),
        batch_shape=(2,),
        width=2,
    )
    scaled = row_scale_storage(csr, np.array([2.0, 3.0]))
    expected = np.array([[2.0, 4.0], [9.0, 12.0]])
    np.testing.assert_array_equal(scaled.as_dense(), expected)


def test_add_csr_storage_preserves_csr_without_densifying(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding CSR storage with matching batch shape must merge sparse rows directly."""
    lhs = CSRStorage(
        data=np.array([1.0, 3.0, 5.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    rhs = CSRStorage(
        data=np.array([2.0, -3.0, 7.0]),
        indices=np.array([0, 2, 2]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )

    def fail_as_dense(self: CSRStorage) -> np.ndarray:
        raise AssertionError("CSR add should not densify matching CSR inputs")

    monkeypatch.setattr(CSRStorage, "as_dense", fail_as_dense)
    added = add_storage(lhs, rhs)

    assert isinstance(added, CSRStorage)
    np.testing.assert_array_equal(added._payload.indices, np.array([0, 1, 2]))
    np.testing.assert_array_equal(added._payload.data, np.array([3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(added._payload.indptr, np.array([0, 1, 3]))


def test_sub_csr_storage_preserves_csr_without_densifying(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subtracting CSR storage with matching batch shape must merge sparse rows directly."""
    lhs = CSRStorage(
        data=np.array([1.0, 3.0, 5.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    rhs = CSRStorage(
        data=np.array([1.0, 4.0]),
        indices=np.array([0, 2]),
        indptr=np.array([0, 1, 2]),
        batch_shape=(2,),
        width=3,
    )

    def fail_as_dense(self: CSRStorage) -> np.ndarray:
        raise AssertionError("CSR sub should not densify matching CSR inputs")

    monkeypatch.setattr(CSRStorage, "as_dense", fail_as_dense)
    subtracted = sub_storage(lhs, rhs)

    assert isinstance(subtracted, CSRStorage)
    np.testing.assert_array_equal(subtracted._payload.indices, np.array([2, 1, 2]))
    np.testing.assert_array_equal(subtracted._payload.data, np.array([3.0, 5.0, -4.0]))
    np.testing.assert_array_equal(subtracted._payload.indptr, np.array([0, 1, 3]))


def test_coefficient_magnitude_squared_csr_storage() -> None:
    """CSR coefficient magnitude squared must reduce stored coefficients by row."""
    csr = CSRStorage(
        data=np.array([1.0, -2.0, 3.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )

    np.testing.assert_array_equal(coefficient_magnitude_squared_storage(csr), np.array([5.0, 9.0]))


def test_project_dense_storage() -> None:
    """Projecting dense storage must select and reorder columns."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    projected = project_storage(dense, (2, 0))  # Select columns 2, 0
    expected = np.array([[3.0, 1.0], [6.0, 4.0]])
    np.testing.assert_array_equal(projected.as_dense(), expected)


def test_project_dense_storage_with_none() -> None:
    """Projecting with None must insert zero columns."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0, 3.0]]))
    projected = project_storage(dense, (0, None, 2))  # Select col 0, zero, col 2
    expected = np.array([[1.0, 0.0, 3.0]])
    np.testing.assert_array_equal(projected.as_dense(), expected)


def test_project_csr_storage() -> None:
    """Projecting CSR storage must select and reorder columns."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0, 5.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    projected = project_storage(csr, (2, 0))  # Select columns 2, 0
    assert projected.width == 2
    expected = np.array([[3.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(projected.as_dense(), expected)


def test_project_csr_storage_with_none() -> None:
    """Projecting CSR with None must preserve sparsity structure."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0]),
        indices=np.array([0, 2]),
        indptr=np.array([0, 2]),
        batch_shape=(),
        width=3,
    )
    projected = project_storage(csr, (0, None, 2))
    assert projected.width == 3
    expected = np.array([1.0, 0.0, 3.0])
    np.testing.assert_array_equal(projected.as_dense(), expected)


def test_storage_component_dense() -> None:
    """Extracting a component from dense storage must return the column."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    component = storage_component(dense, 1)
    expected = np.array([2.0, 5.0])
    np.testing.assert_array_equal(component, expected)


def test_storage_component_csr() -> None:
    """Extracting a component from CSR must return zeros for missing columns."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0]),
        indices=np.array([0, 2]),
        indptr=np.array([0, 2]),
        batch_shape=(),
        width=3,
    )
    component = storage_component(csr, 1)  # Column 1 is not stored
    np.testing.assert_array_equal(component, 0.0)


def test_storage_component_csr_present() -> None:
    """Extracting a present component from CSR must return the value."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0]),
        indices=np.array([0, 2]),
        indptr=np.array([0, 2]),
        batch_shape=(),
        width=3,
    )
    component = storage_component(csr, 2)
    np.testing.assert_array_equal(component, 3.0)


def test_gather_storage_columns_dense() -> None:
    """Gathering columns from dense storage must select in order."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    gathered = gather_storage_columns(dense, (2, 0))
    expected = np.array([[3.0, 1.0], [6.0, 4.0]])
    np.testing.assert_array_equal(gathered, expected)


def test_gather_storage_columns_csr() -> None:
    """Gathering columns from CSR must handle sparsity correctly."""
    csr = CSRStorage(
        data=np.array([1.0, 3.0, 5.0]),
        indices=np.array([0, 2, 1]),
        indptr=np.array([0, 2, 3]),
        batch_shape=(2,),
        width=3,
    )
    gathered = gather_storage_columns(csr, (2, 0))
    expected = np.array([[3.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(gathered, expected)


def test_gather_empty_columns_returns_zeros() -> None:
    """Gathering no columns must return empty array."""
    dense = DenseStorage.from_array(np.array([[1.0, 2.0, 3.0]]))
    gathered = gather_storage_columns(dense, ())
    expected = np.array([[]])
    np.testing.assert_array_equal(gathered, expected)


def test_csr_validation_rejects_multidimensional_data() -> None:
    """CSR validation must reject multidimensional data arrays."""
    with pytest.raises(ValueError, match="CSR data must be a one-dimensional array"):
        CSRStorage(
            data=np.array([[1.0, 2.0]]),
            indices=np.array([0, 1]),
            indptr=np.array([0, 2]),
            batch_shape=(),
            width=2,
        )


def test_csr_validation_rejects_mismatched_shapes() -> None:
    """CSR validation must reject data/indices shape mismatch."""
    with pytest.raises(ValueError, match="CSR data and indices must have the same shape"):
        CSRStorage(
            data=np.array([1.0, 2.0, 3.0]),
            indices=np.array([0, 1]),
            indptr=np.array([0, 2]),
            batch_shape=(),
            width=2,
        )


def test_csr_validation_rejects_invalid_indptr_length() -> None:
    """CSR validation must reject indptr length mismatch."""
    with pytest.raises(ValueError, match="CSR indptr length must match"):
        CSRStorage(
            data=np.array([1.0, 2.0]),
            indices=np.array([0, 1]),
            indptr=np.array([0, 1, 2, 3]),  # Wrong length
            batch_shape=(2,),
            width=2,
        )


def test_csr_validation_rejects_nonzero_indptr_start() -> None:
    """CSR validation must reject indptr that doesn't start at 0."""
    with pytest.raises(ValueError, match="CSR indptr must start at 0"):
        CSRStorage(
            data=np.array([1.0, 2.0]),
            indices=np.array([0, 1]),
            indptr=np.array([1, 2]),
            batch_shape=(),
            width=2,
        )


def test_csr_validation_rejects_decreasing_indptr() -> None:
    """CSR validation must reject non-decreasing indptr."""
    with pytest.raises(ValueError, match="CSR indptr must be nondecreasing"):
        CSRStorage(
            data=np.array([1.0, 2.0]),
            indices=np.array([0, 1]),
            indptr=np.array([0, 2, 1]),
            batch_shape=(2,),
            width=2,
        )


def test_csr_validation_rejects_out_of_bounds_indices() -> None:
    """CSR validation must reject indices outside [0, width)."""
    with pytest.raises(ValueError, match="CSR indices must be between 0 and width - 1"):
        CSRStorage(
            data=np.array([1.0, 2.0]),
            indices=np.array([0, 5]),  # 5 >= width=2
            indptr=np.array([0, 2]),
            batch_shape=(),
            width=2,
        )


def test_csr_validation_rejects_non_increasing_row_indices() -> None:
    """CSR validation must reject non-strictly-increasing row indices."""
    with pytest.raises(ValueError, match="CSR row indices must be strictly increasing"):
        CSRStorage(
            data=np.array([1.0, 2.0, 3.0]),
            indices=np.array([0, 1, 1]),  # Row 0: 1 after 1 is not strictly increasing
            indptr=np.array([0, 3, 3]),
            batch_shape=(2,),
            width=2,
        )
