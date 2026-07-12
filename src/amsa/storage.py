# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
from typing import Any, Literal, Protocol, Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

StorageKind = Literal["dense", "csr"]
StorageRequest = Literal["auto", "dense", "csr"]


_JAX_ARRAY_TYPE: Any = None


def _get_jax_array_type() -> Any:
    global _JAX_ARRAY_TYPE
    if _JAX_ARRAY_TYPE is None:
        try:
            from jax import Array
            _JAX_ARRAY_TYPE = Array
        except ImportError:
            _JAX_ARRAY_TYPE = False
    return _JAX_ARRAY_TYPE


def is_jax_array(value: Any) -> bool:
    cls = _get_jax_array_type()
    return cls is not False and isinstance(value, cls)


def _normalize_batch_shape(batch_shape: tuple[int, ...]) -> tuple[int, ...]:
    normalized: list[int] = []
    for size in batch_shape:
        size_int = index(size)
        if size_int < 0:
            raise ValueError("batch dimensions must be non-negative.")
        normalized.append(size_int)
    return tuple(normalized)


def _validate_csr_arrays(
    data_array: NDArray[Any],
    index_array: NDArray[Any],
    indptr_array: NDArray[Any],
    width_value: int,
) -> None:
    """Validate basic CSR array structure and dimensions."""
    if data_array.ndim != 1:
        raise ValueError("CSR data must be a one-dimensional array.")
    if index_array.ndim != 1:
        raise ValueError("CSR indices must be a one-dimensional array.")
    if indptr_array.ndim != 1:
        raise ValueError("CSR indptr must be a one-dimensional array.")
    if data_array.shape != index_array.shape:
        raise ValueError("CSR data and indices must have the same shape.")
    if indptr_array.size == 0:
        raise ValueError("CSR indptr must include at least the starting offset.")
    if width_value < 0:
        raise ValueError("CSR width must be non-negative.")


def _validate_csr_indptr(
    indptr_array: NDArray[Any],
    row_count: int,
    data_size: int,
) -> None:
    """Validate CSR indptr structure and consistency with data."""
    if indptr_array.size != row_count + 1:
        raise ValueError("CSR indptr length must match flattened row_count + 1.")
    if int(indptr_array[0]) != 0:
        raise ValueError("CSR indptr must start at 0.")
    if np.any(indptr_array[1:] < indptr_array[:-1]):
        raise ValueError("CSR indptr must be nondecreasing.")
    if int(indptr_array[-1]) != data_size:
        raise ValueError("CSR indptr must end at the number of stored values.")


def _validate_csr_indices(
    index_array: NDArray[Any],
    width_value: int,
    data_size: int,
) -> None:
    """Validate CSR index bounds."""
    if data_size and (np.any(index_array < 0) or np.any(index_array >= width_value)):
        raise ValueError("CSR indices must be between 0 and width - 1.")


def _validate_csr_row_ordering(
    index_array: NDArray[Any],
    indptr_array: NDArray[Any],
    row_count: int,
) -> None:
    """Validate that indices within each row are strictly increasing."""
    for row in range(row_count):
        start = int(indptr_array[row])
        stop = int(indptr_array[row + 1])
        row_indices = index_array[start:stop]
        if np.any(row_indices[1:] <= row_indices[:-1]):
            raise ValueError("CSR row indices must be strictly increasing.")


def _broadcast_row_indices(
    source_batch_shape: tuple[int, ...],
    target_batch_shape: tuple[int, ...],
) -> NDArray[np.intp]:
    row_count = int(prod(source_batch_shape))
    rows = np.arange(row_count, dtype=np.intp).reshape(source_batch_shape)
    return np.broadcast_to(rows, target_batch_shape).reshape(-1)


class StorageDescriptor(Protocol):
    """Backend-agnostic storage metadata descriptor.

    Separates storage structure (shape, dtype, width) from backend-specific
    array payloads. This enables different backends (NumPy, JAX, etc.) to
    share the same storage contract while managing their own array types.

    Contract:
    - ``kind``: Storage backend identifier ("dense" or "csr").
    - ``batch_shape``: Shape of the multivector batch (all dimensions except the last).
    - ``dtype``: Data type of coefficient values.
    - ``width``: Number of coefficient slots (last dimension size).
    """

    @property
    def kind(self) -> StorageKind: ...

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def width(self) -> int: ...


class BackendPayload(Protocol):
    """Backend-specific array payload interface.

    Implemented by backends to provide array data for storage operations.
    This allows storage operations to work with different array types
    (NumPy arrays, JAX arrays, etc.) through a common interface.

    Contract:
    - ``as_dense()``: Materialize as a dense array with shape (batch_shape + (width,)).
    - ``copy()``: Return a deep copy of the payload.
    """

    def as_dense(self) -> NDArray[Any]: ...

    def copy(self) -> Self: ...


@dataclass(frozen=True, slots=True)
class NumPyPayload:
    """NumPy-based backend payload implementation.

    Wraps NumPy arrays to implement the BackendPayload protocol.
    This is the default payload for the NumPy backend.
    """

    array: NDArray[Any]

    def as_dense(self) -> NDArray[Any]:
        return self.array

    def copy(self) -> NumPyPayload:
        return NumPyPayload(self.array.copy())


@dataclass(frozen=True, slots=True)
class NumPyCSRPayload:
    """NumPy-based CSR backend payload implementation.

    Wraps CSR arrays (data, indices, indptr) to implement the BackendPayload protocol.
    This is the payload for CSR storage in the NumPy backend.
    """

    data: NDArray[Any]
    indices: NDArray[Any]
    indptr: NDArray[Any]
    _batch_shape: tuple[int, ...]
    _width: int

    def as_dense(self) -> NDArray[Any]:
        """Materialize CSR as a dense array."""
        row_count = int(self.indptr.size - 1)
        dense = np.zeros((row_count, self._width), dtype=self.data.dtype)
        for row in range(row_count):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            if start == stop:
                continue
            dense[row, self.indices[start:stop]] = self.data[start:stop]
        return dense.reshape(self._batch_shape + (self._width,))

    def copy(self) -> NumPyCSRPayload:
        return NumPyCSRPayload(
            data=self.data.copy(),
            indices=self.indices.copy(),
            indptr=self.indptr.copy(),
            _batch_shape=self._batch_shape,
            _width=self._width,
        )


class MVStorage(Protocol):
    """Structural interface implemented by concrete storage backends.

    Storage backends represent coefficient arrays for multivector batches.
    They combine a StorageDescriptor (metadata) with a BackendPayload (array data).

    Contract for storage implementations:
    - ``kind``: Storage backend identifier ("dense" or "csr").
    - ``batch_shape``: Shape of the multivector batch (all dimensions except the last).
    - ``dtype``: NumPy dtype of coefficient values.
    - ``width``: Number of coefficient slots (last dimension size).
    - ``as_dense()``: Materialize as a dense array with shape (batch_shape + (width,)).
    - ``copy()``: Return a deep copy of the storage.

    Storage backends must preserve:
    - Coefficient values across conversions
    - Batch broadcasting semantics
    - Dtype promotion rules
    - Zero-support behavior (empty layouts return zeros on component access)

    Storage backends are NOT responsible for:
    - Algebraic blade identity or ordering (handled by layouts)
    - Product planning or execution (handled by plans/IR/backends)
    - Geometric interpretation (handled by algebra/ops layers)
    """

    @property
    def kind(self) -> StorageKind: ...

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def width(self) -> int: ...

    def as_dense(self) -> NDArray[Any]: ...

    def copy(self) -> Self: ...


def resolve_storage_kind(
    kind: StorageRequest,
    *,
    auto_kind: StorageKind = "dense",
) -> StorageKind:
    if kind == "auto":
        return auto_kind
    if kind in ("dense", "csr"):
        return kind
    raise ValueError(f"Unsupported storage kind request: {kind!r}")


@dataclass(frozen=True, slots=True)
class DenseStorage:
    _payload: NumPyPayload
    kind: StorageKind = "dense"

    @classmethod
    def from_array(cls, array: ArrayLike) -> DenseStorage:
        """Create DenseStorage from an array-like object."""
        values: Any = array if is_jax_array(array) else np.asarray(array)
        if values.ndim == 0:
            raise ValueError("storage values must have at least one dimension.")
        return cls(_payload=NumPyPayload(array=cast(NDArray[Any], values)))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._payload.array.shape[:-1]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._payload.array.dtype

    @property
    def width(self) -> int:
        return int(self._payload.array.shape[-1])

    def as_dense(self) -> NDArray[Any]:
        return self._payload.as_dense()

    def copy(self) -> DenseStorage:
        return DenseStorage(_payload=self._payload.copy())

    @classmethod
    def zeros(
        cls,
        width: int,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[Any] | type[np.float64] = np.float64,
    ) -> DenseStorage:
        return cls.from_array(np.zeros(batch_shape + (width,), dtype=dtype))


@dataclass(frozen=True, slots=True, init=False)
class CSRStorage:
    """NumPy-backed compressed row storage for flattened multivector batches."""

    _payload: NumPyCSRPayload
    _dtype: np.dtype[Any]
    kind: StorageKind

    def __init__(
        self,
        data: ArrayLike,
        indices: ArrayLike,
        indptr: ArrayLike,
        *,
        batch_shape: tuple[int, ...],
        width: int,
        dtype: np.dtype[Any] | type[Any] | None = None,
    ) -> None:
        resolved_dtype = np.dtype(np.asarray(data).dtype if dtype is None else dtype)
        data_array = np.asarray(data, dtype=resolved_dtype)
        index_array = np.asarray(indices, dtype=np.intp)
        indptr_array = np.asarray(indptr, dtype=np.intp)
        normalized_batch_shape = _normalize_batch_shape(batch_shape)
        width_value = index(width)

        _validate_csr_arrays(data_array, index_array, indptr_array, width_value)

        row_count = int(prod(normalized_batch_shape))
        _validate_csr_indptr(indptr_array, row_count, data_array.size)
        _validate_csr_indices(index_array, width_value, data_array.size)
        _validate_csr_row_ordering(index_array, indptr_array, row_count)

        payload = NumPyCSRPayload(
            data=data_array,
            indices=index_array,
            indptr=indptr_array,
            _batch_shape=normalized_batch_shape,
            _width=width_value,
        )

        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_dtype", resolved_dtype)
        object.__setattr__(self, "kind", "csr")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._payload._batch_shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def width(self) -> int:
        return self._payload._width

    @property
    def row_count(self) -> int:
        return int(self._payload.indptr.size - 1)

    def as_dense(self) -> NDArray[Any]:
        return self._payload.as_dense()

    def copy(self) -> CSRStorage:
        return CSRStorage._from_validated_arrays(
            self._payload.data.copy(),
            self._payload.indices.copy(),
            self._payload.indptr.copy(),
            batch_shape=self.batch_shape,
            width=self.width,
            dtype=self.dtype,
        )

    @classmethod
    def _from_validated_arrays(
        cls,
        data: NDArray[Any],
        indices: NDArray[Any],
        indptr: NDArray[Any],
        *,
        batch_shape: tuple[int, ...],
        width: int,
        dtype: np.dtype[Any] | type[Any],
    ) -> CSRStorage:
        storage = object.__new__(cls)
        resolved_dtype = np.dtype(dtype)
        payload = NumPyCSRPayload(
            data=np.asarray(data, dtype=resolved_dtype),
            indices=np.asarray(indices, dtype=np.intp),
            indptr=np.asarray(indptr, dtype=np.intp),
            _batch_shape=batch_shape,
            _width=width,
        )
        object.__setattr__(storage, "_payload", payload)
        object.__setattr__(storage, "_dtype", resolved_dtype)
        object.__setattr__(storage, "kind", "csr")
        return storage

    @classmethod
    def zeros(
        cls,
        width: int,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[Any] | type[np.float64] = np.float64,
    ) -> CSRStorage:
        normalized_batch_shape = _normalize_batch_shape(batch_shape)
        row_count = int(prod(normalized_batch_shape))
        return cls._from_validated_arrays(
            np.array([], dtype=dtype),
            np.array([], dtype=np.intp),
            np.zeros(row_count + 1, dtype=np.intp),
            batch_shape=normalized_batch_shape,
            width=width,
            dtype=dtype,
        )


def to_dense_storage(storage: MVStorage) -> DenseStorage:
    if isinstance(storage, DenseStorage):
        return storage.copy()
    if isinstance(storage, CSRStorage):
        return DenseStorage.from_array(storage.as_dense())
    raise TypeError(f"Unsupported storage type: {type(storage)!r}")


def to_csr_storage(storage: MVStorage) -> CSRStorage:
    if isinstance(storage, CSRStorage):
        return storage.copy()
    if not isinstance(storage, DenseStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    flat_values = storage._payload.array.reshape((-1, storage.width))
    data_values: list[Any] = []
    index_values: list[int] = []
    indptr = np.zeros(flat_values.shape[0] + 1, dtype=np.intp)

    nnz = 0
    for row_index, row in enumerate(flat_values):
        for column_index in np.flatnonzero(row):
            data_values.append(row[column_index])
            index_values.append(int(column_index))
            nnz += 1
        indptr[row_index + 1] = nnz

    return CSRStorage._from_validated_arrays(
        np.asarray(data_values, dtype=storage.dtype),
        np.asarray(index_values, dtype=np.intp),
        indptr,
        batch_shape=storage.batch_shape,
        width=storage.width,
        dtype=storage.dtype,
    )


def build_storage_from_array(values: ArrayLike, *, kind: StorageRequest = "dense") -> MVStorage:
    dense = DenseStorage.from_array(values)
    resolved_kind = resolve_storage_kind(kind)
    if resolved_kind == "dense":
        return dense
    return to_csr_storage(dense)


def build_zero_storage(
    width: int,
    *,
    batch_shape: tuple[int, ...] = (),
    dtype: np.dtype[Any] | type[np.float64] = np.float64,
    kind: StorageRequest = "dense",
) -> MVStorage:
    resolved_kind = resolve_storage_kind(kind)
    if resolved_kind == "dense":
        return DenseStorage.zeros(width, batch_shape=batch_shape, dtype=dtype)
    return CSRStorage.zeros(width, batch_shape=batch_shape, dtype=dtype)


def convert_storage_kind(storage: MVStorage, kind: StorageKind) -> MVStorage:
    if kind == "dense":
        return to_dense_storage(storage)
    if kind == "csr":
        return to_csr_storage(storage)
    raise ValueError(f"Unsupported storage kind: {kind!r}")


def storage_component(storage: MVStorage, column: int) -> NDArray[Any]:
    if column < 0 or column >= storage.width:
        raise IndexError(f"Storage column {column} is out of bounds for width {storage.width}.")

    if isinstance(storage, DenseStorage):
        return storage._payload.array[..., column]
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    flat = np.zeros(storage.row_count, dtype=storage.dtype)
    for row in range(storage.row_count):
        start = int(storage._payload.indptr[row])
        stop = int(storage._payload.indptr[row + 1])
        row_indices = storage._payload.indices[start:stop]
        match = int(np.searchsorted(row_indices, column))
        if match < row_indices.size and int(row_indices[match]) == column:
            flat[row] = storage._payload.data[start + match]
    return flat.reshape(storage.batch_shape)


def gather_storage_columns(
    storage: MVStorage,
    columns: tuple[int, ...],
    *,
    batch_shape: tuple[int, ...] | None = None,
) -> NDArray[Any]:
    for column in columns:
        if column < 0 or column >= storage.width:
            raise IndexError(f"Storage column {column} is out of bounds for width {storage.width}.")

    target_batch_shape = storage.batch_shape if batch_shape is None else batch_shape

    if not columns:
        empty = np.zeros(storage.batch_shape + (0,), dtype=storage.dtype)
        return np.broadcast_to(empty, target_batch_shape + (0,))

    if isinstance(storage, DenseStorage):
        gathered = np.asarray(storage._payload.array[..., list(columns)], dtype=storage.dtype)
        return np.broadcast_to(gathered, target_batch_shape + (len(columns),))
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    source_to_targets: dict[int, list[int]] = {}
    for target_column, source_column in enumerate(columns):
        source_to_targets.setdefault(source_column, []).append(target_column)

    flat = np.zeros((storage.row_count, len(columns)), dtype=storage.dtype)
    for row in range(storage.row_count):
        start = int(storage._payload.indptr[row])
        stop = int(storage._payload.indptr[row + 1])
        for offset in range(start, stop):
            source_column = int(storage._payload.indices[offset])
            targets = source_to_targets.get(source_column)
            if targets is None:
                continue
            for target_column in targets:
                flat[row, target_column] = storage._payload.data[offset]

    dense = flat.reshape(storage.batch_shape + (len(columns),))
    return np.broadcast_to(dense, target_batch_shape + (len(columns),))


def project_storage(storage: MVStorage, columns: tuple[int | None, ...]) -> MVStorage:
    target_width = len(columns)
    if isinstance(storage, DenseStorage):
        if all(column is not None for column in columns):
            dense_columns = cast(tuple[int, ...], columns)
            return DenseStorage(
                _payload=NumPyPayload(array=storage._payload.array[..., list(dense_columns)])
            )
        projected = np.zeros(storage.batch_shape + (target_width,), dtype=storage.dtype)
        for out_column, in_column in enumerate(columns):
            if in_column is not None:
                projected[..., out_column] = storage._payload.array[..., in_column]
        return DenseStorage.from_array(projected)
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    source_to_target = {
        source_column: target_column
        for target_column, source_column in enumerate(columns)
        if source_column is not None
    }

    data_values: list[Any] = []
    index_values: list[int] = []
    indptr = np.zeros(storage.row_count + 1, dtype=np.intp)

    nnz = 0
    for row in range(storage.row_count):
        start = int(storage._payload.indptr[row])
        stop = int(storage._payload.indptr[row + 1])
        row_entries: list[tuple[int, Any]] = []

        for offset in range(start, stop):
            source_column = int(storage._payload.indices[offset])
            target_column = source_to_target.get(source_column)
            if target_column is None:
                continue
            row_entries.append((target_column, storage._payload.data[offset]))

        row_entries.sort(key=lambda entry: entry[0])
        for target_column, value in row_entries:
            index_values.append(target_column)
            data_values.append(value)
            nnz += 1
        indptr[row + 1] = nnz

    return CSRStorage._from_validated_arrays(
        np.asarray(data_values, dtype=storage.dtype),
        np.asarray(index_values, dtype=np.intp),
        indptr,
        batch_shape=storage.batch_shape,
        width=target_width,
        dtype=storage.dtype,
    )


def scale_storage(storage: MVStorage, scalar: Any) -> MVStorage:
    scalar_array = np.asarray(scalar)
    result_dtype = np.dtype(np.result_type(storage.dtype, scalar_array.dtype))
    is_zero = bool(np.equal(scalar_array, 0).item())

    if isinstance(storage, DenseStorage):
        values = np.asarray(storage._payload.array, dtype=result_dtype) * scalar
        return DenseStorage.from_array(values)
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")
    if is_zero:
        return CSRStorage.zeros(storage.width, batch_shape=storage.batch_shape, dtype=result_dtype)

    return CSRStorage._from_validated_arrays(
        np.asarray(storage._payload.data, dtype=result_dtype) * scalar,
        storage._payload.indices.copy(),
        storage._payload.indptr.copy(),
        batch_shape=storage.batch_shape,
        width=storage.width,
        dtype=result_dtype,
    )


def reweight_storage(storage: MVStorage, weights: ArrayLike) -> MVStorage:
    weight_array = np.asarray(weights)
    if weight_array.ndim != 1:
        raise ValueError("weights must be a one-dimensional array.")
    if weight_array.shape[0] != storage.width:
        raise ValueError(f"weights must have length {storage.width}.")

    result_dtype = np.dtype(np.result_type(storage.dtype, weight_array.dtype))
    resolved_weights = np.asarray(weight_array, dtype=result_dtype)

    if isinstance(storage, DenseStorage):
        values = np.asarray(storage._payload.array, dtype=result_dtype) * resolved_weights
        return DenseStorage.from_array(values)
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    weighted_data = np.asarray(storage._payload.data, dtype=result_dtype) * resolved_weights[
        storage._payload.indices
    ]
    return CSRStorage._from_validated_arrays(
        weighted_data,
        storage._payload.indices.copy(),
        storage._payload.indptr.copy(),
        batch_shape=storage.batch_shape,
        width=storage.width,
        dtype=result_dtype,
    )


def row_scale_storage(storage: MVStorage, factors: ArrayLike) -> MVStorage:
    factor_array = np.asarray(factors)
    if factor_array.shape != storage.batch_shape:
        raise ValueError(
            "factors must match storage batch_shape "
            f"{storage.batch_shape}, got {factor_array.shape}."
        )

    result_dtype = np.dtype(np.result_type(storage.dtype, factor_array.dtype))
    resolved_factors = np.asarray(factor_array, dtype=result_dtype)

    if isinstance(storage, DenseStorage):
        values = np.asarray(
            storage._payload.array, dtype=result_dtype
        ) * resolved_factors[..., np.newaxis]
        return DenseStorage.from_array(values)
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    flat_factors = resolved_factors.reshape(storage.row_count)
    data = np.asarray(storage._payload.data, dtype=result_dtype).copy()
    for row in range(storage.row_count):
        start = int(storage._payload.indptr[row])
        stop = int(storage._payload.indptr[row + 1])
        if start == stop:
            continue
        data[start:stop] *= flat_factors[row]

    return CSRStorage._from_validated_arrays(
        data,
        storage._payload.indices.copy(),
        storage._payload.indptr.copy(),
        batch_shape=storage.batch_shape,
        width=storage.width,
        dtype=result_dtype,
    )


def _check_binary_storage_compatible(lhs: MVStorage, rhs: MVStorage) -> tuple[int, tuple[int, ...]]:
    if lhs.width != rhs.width:
        raise ValueError(f"storage widths must match, got {lhs.width} and {rhs.width}.")
    return lhs.width, np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)


def _add_dense_storage(lhs: MVStorage, rhs: MVStorage, sign: Literal[1, -1]) -> DenseStorage:
    _, batch_shape = _check_binary_storage_compatible(lhs, rhs)
    lhs_values = np.broadcast_to(lhs.as_dense(), batch_shape + (lhs.width,))
    rhs_values = np.broadcast_to(rhs.as_dense(), batch_shape + (rhs.width,))
    values = lhs_values + rhs_values if sign == 1 else lhs_values - rhs_values
    return DenseStorage(_payload=NumPyPayload(array=values))


def _add_dense_csr_storage(
    dense: DenseStorage,
    csr: CSRStorage,
    sign: Literal[1, -1],
    *,
    dense_is_lhs: bool,
) -> DenseStorage:
    _, batch_shape = _check_binary_storage_compatible(dense, csr)
    result_dtype = np.dtype(np.result_type(dense.dtype, csr.dtype))
    values = np.array(
        np.broadcast_to(dense._payload.array, batch_shape + (dense.width,)),
        dtype=result_dtype,
        copy=True,
    )
    if not dense_is_lhs:
        values *= sign
    csr_rows = _broadcast_row_indices(csr.batch_shape, batch_shape)

    flat_values = values.reshape((-1, dense.width))
    for target_row, csr_row in enumerate(csr_rows):
        start = int(csr._payload.indptr[csr_row])
        stop = int(csr._payload.indptr[csr_row + 1])
        if start == stop:
            continue
        columns = csr._payload.indices[start:stop]
        csr_values = np.asarray(csr._payload.data[start:stop], dtype=result_dtype)
        if dense_is_lhs:
            flat_values[target_row, columns] += sign * csr_values
        else:
            flat_values[target_row, columns] += csr_values

    return DenseStorage.from_array(values)


def _add_csr_storage(lhs: CSRStorage, rhs: CSRStorage, sign: Literal[1, -1]) -> CSRStorage:
    _, batch_shape = _check_binary_storage_compatible(lhs, rhs)
    result_dtype = np.dtype(np.result_type(lhs.dtype, rhs.dtype))

    if lhs.width == 0:
        return CSRStorage.zeros(lhs.width, batch_shape=batch_shape, dtype=result_dtype)

    row_count = int(prod(batch_shape))
    lhs_rows = _broadcast_row_indices(lhs.batch_shape, batch_shape)
    rhs_rows = _broadcast_row_indices(rhs.batch_shape, batch_shape)

    def broadcast_entries(
        storage: CSRStorage,
        rows: NDArray[np.intp],
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[Any]]:
        starts = storage._payload.indptr[rows]
        lengths = storage._payload.indptr[rows + 1] - starts
        total = int(np.sum(lengths))
        if total == 0:
            return (
                np.empty(0, dtype=np.intp),
                np.empty(0, dtype=np.intp),
                np.empty(0, dtype=result_dtype),
            )

        offsets = np.empty(rows.size + 1, dtype=np.intp)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        positions = (
            np.repeat(starts, lengths)
            + np.arange(total, dtype=np.intp)
            - np.repeat(offsets[:-1], lengths)
        )
        out_rows = np.repeat(np.arange(rows.size, dtype=np.intp), lengths)
        return (
            out_rows,
            storage._payload.indices[positions],
            np.asarray(storage._payload.data[positions], dtype=result_dtype),
        )

    lhs_out_rows, lhs_columns, lhs_data = broadcast_entries(lhs, lhs_rows)
    rhs_out_rows, rhs_columns, rhs_data = broadcast_entries(rhs, rhs_rows)

    entry_rows = np.concatenate((lhs_out_rows, rhs_out_rows))
    entry_columns = np.concatenate((lhs_columns, rhs_columns))
    entry_data = np.concatenate((lhs_data, sign * rhs_data))

    if entry_data.size == 0:
        return CSRStorage.zeros(lhs.width, batch_shape=batch_shape, dtype=result_dtype)

    keys = entry_rows * lhs.width + entry_columns
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_data = entry_data[order]

    unique_starts = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]).astype(np.intp) + 1,
        )
    )
    reduced_keys = sorted_keys[unique_starts]
    reduced_data = np.add.reduceat(sorted_data, unique_starts)
    nonzero = reduced_data != 0
    reduced_keys = reduced_keys[nonzero]
    reduced_data = reduced_data[nonzero]

    index_values = reduced_keys % lhs.width
    row_values = reduced_keys // lhs.width
    indptr = np.zeros(row_count + 1, dtype=np.intp)
    if row_values.size:
        counts = np.bincount(row_values, minlength=row_count)
        np.cumsum(counts, out=indptr[1:])

    return CSRStorage._from_validated_arrays(
        reduced_data,
        np.asarray(index_values, dtype=np.intp),
        indptr,
        batch_shape=batch_shape,
        width=lhs.width,
        dtype=result_dtype,
    )


def add_storage(lhs: MVStorage, rhs: MVStorage) -> MVStorage:
    """Add storage payloads with backend preservation where storage-local."""
    _, batch_shape = _check_binary_storage_compatible(lhs, rhs)
    if isinstance(lhs, CSRStorage) and isinstance(rhs, CSRStorage):
        return _add_csr_storage(lhs, rhs, 1)
    if isinstance(lhs, DenseStorage) and isinstance(rhs, DenseStorage):
        if batch_shape != lhs.batch_shape or batch_shape != rhs.batch_shape:
            return _add_dense_storage(lhs, rhs, 1)
        return DenseStorage(_payload=NumPyPayload(array=lhs._payload.array + rhs._payload.array))
    if isinstance(lhs, DenseStorage) and isinstance(rhs, CSRStorage):
        return _add_dense_csr_storage(lhs, rhs, 1, dense_is_lhs=True)
    if isinstance(lhs, CSRStorage) and isinstance(rhs, DenseStorage):
        return _add_dense_csr_storage(rhs, lhs, 1, dense_is_lhs=False)
    return _add_dense_storage(lhs, rhs, 1)


def sub_storage(lhs: MVStorage, rhs: MVStorage) -> MVStorage:
    """Subtract storage payloads with backend preservation where storage-local."""
    _, batch_shape = _check_binary_storage_compatible(lhs, rhs)
    if isinstance(lhs, CSRStorage) and isinstance(rhs, CSRStorage):
        return _add_csr_storage(lhs, rhs, -1)
    if isinstance(lhs, DenseStorage) and isinstance(rhs, DenseStorage):
        if batch_shape != lhs.batch_shape or batch_shape != rhs.batch_shape:
            return _add_dense_storage(lhs, rhs, -1)
        return DenseStorage(_payload=NumPyPayload(array=lhs._payload.array - rhs._payload.array))
    if isinstance(lhs, DenseStorage) and isinstance(rhs, CSRStorage):
        return _add_dense_csr_storage(lhs, rhs, -1, dense_is_lhs=True)
    if isinstance(lhs, CSRStorage) and isinstance(rhs, DenseStorage):
        return _add_dense_csr_storage(rhs, lhs, -1, dense_is_lhs=False)
    return _add_dense_storage(lhs, rhs, -1)

def coefficient_magnitude_squared_storage(storage: MVStorage) -> NDArray[Any]:
    """Return sum of squared stored coefficients over the coefficient axis."""
    dtype = np.result_type(storage.dtype, np.float64)
    if storage.width == 0:
        return np.zeros(storage.batch_shape, dtype=dtype)
    if isinstance(storage, DenseStorage):
        values = storage._payload.array
        return cast(NDArray[Any], np.sum(values * values, axis=-1, dtype=dtype))
    if not isinstance(storage, CSRStorage):
        raise TypeError(f"Unsupported storage type: {type(storage)!r}")

    flat = np.zeros(storage.row_count, dtype=dtype)
    for row in range(storage.row_count):
        start = int(storage._payload.indptr[row])
        stop = int(storage._payload.indptr[row + 1])
        row_values = storage._payload.data[start:stop]
        flat[row] = np.sum(row_values * row_values, dtype=dtype)
    return flat.reshape(storage.batch_shape)


def index_dense_storage(storage: MVStorage, key: Any) -> MVStorage:
    """Index or slice dense storage for batch indexing.

    This is the storage-aware counterpart to MVArray.__getitem__ for dense storage.
    """
    if not isinstance(storage, DenseStorage):
        raise TypeError("index_dense_storage only supports DenseStorage.")

    new_array = storage._payload.array[key]
    if new_array.ndim == 0:
        raise IndexError("Too many indices for multivector batch.")
    return DenseStorage.from_array(new_array)


def index_csr_storage(storage: MVStorage, key: Any) -> MVStorage:
    """Index or slice CSR storage by flattened batch rows without densifying."""
    if not isinstance(storage, CSRStorage):
        raise TypeError("index_csr_storage only supports CSRStorage.")

    if isinstance(key, slice) and len(storage.batch_shape) == 1:
        row_indices = np.arange(storage.row_count, dtype=np.intp)[key]
        if row_indices.ndim == 0:
            row_indices = row_indices.reshape(1)
        if row_indices.size == 0:
            return CSRStorage.zeros(storage.width, batch_shape=(0,), dtype=storage.dtype)
        if row_indices.size == 1 or np.all(row_indices[1:] == row_indices[:-1] + 1):
            first_row = int(row_indices[0])
            last_row = int(row_indices[-1])
            data_start = int(storage._payload.indptr[first_row])
            data_stop = int(storage._payload.indptr[last_row + 1])
            indptr = storage._payload.indptr[first_row : last_row + 2] - data_start
            return CSRStorage._from_validated_arrays(
                storage._payload.data[data_start:data_stop].copy(),
                storage._payload.indices[data_start:data_stop].copy(),
                indptr.copy(),
                batch_shape=(int(row_indices.size),),
                width=storage.width,
                dtype=storage.dtype,
            )

    row_grid = np.arange(storage.row_count, dtype=np.intp).reshape(storage.batch_shape)
    selected_rows = row_grid[key]
    selected_array = np.asarray(selected_rows)
    if selected_array.ndim == 0:
        result_batch_shape = ()
        flat_selected_rows = selected_array.reshape(1)
    else:
        result_batch_shape = selected_array.shape
        flat_selected_rows = selected_array.reshape(-1)

    lengths = (
        storage._payload.indptr[flat_selected_rows + 1]
        - storage._payload.indptr[flat_selected_rows]
    )
    indptr = np.empty(flat_selected_rows.size + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])

    nnz = int(indptr[-1])
    if nnz:
        starts = storage._payload.indptr[flat_selected_rows]
        source_offsets = np.repeat(starts, lengths)
        output_offsets = np.repeat(indptr[:-1], lengths)
        positions = source_offsets + np.arange(nnz, dtype=np.intp) - output_offsets
        data = storage._payload.data[positions]
        indices = storage._payload.indices[positions]
    else:
        data = np.empty(0, dtype=storage.dtype)
        indices = np.empty(0, dtype=np.intp)

    return CSRStorage._from_validated_arrays(
        data,
        indices,
        indptr,
        batch_shape=result_batch_shape,
        width=storage.width,
        dtype=storage.dtype,
    )
