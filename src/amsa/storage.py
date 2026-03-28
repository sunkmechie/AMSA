from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
from typing import Any, Literal, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

StorageKind = Literal["dense", "csr"]


def _normalize_batch_shape(batch_shape: tuple[int, ...]) -> tuple[int, ...]:
    normalized: list[int] = []
    for size in batch_shape:
        size_int = index(size)
        if size_int < 0:
            raise ValueError("batch dimensions must be non-negative.")
        normalized.append(size_int)
    return tuple(normalized)


class MVStorage(Protocol):
    @property
    def kind(self) -> StorageKind:
        ...

    @property
    def batch_shape(self) -> tuple[int, ...]:
        ...

    @property
    def dtype(self) -> np.dtype[Any]:
        ...

    @property
    def width(self) -> int:
        ...

    def as_dense(self) -> NDArray[Any]:
        ...

    def copy(self) -> Self:
        ...


@dataclass(frozen=True, slots=True)
class DenseStorage:
    array: NDArray[Any]
    kind: StorageKind = "dense"

    def __post_init__(self) -> None:
        values = np.asarray(self.array)
        if values.ndim == 0:
            raise ValueError("storage values must have at least one dimension.")
        object.__setattr__(self, "array", values)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.array.shape[:-1]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.array.dtype

    @property
    def width(self) -> int:
        return int(self.array.shape[-1])

    def as_dense(self) -> NDArray[Any]:
        return self.array

    def copy(self) -> DenseStorage:
        return DenseStorage(self.array.copy())

    @classmethod
    def zeros(
        cls,
        width: int,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[Any] | type[np.float64] = np.float64,
    ) -> DenseStorage:
        return cls(np.zeros(batch_shape + (width,), dtype=dtype))

    @classmethod
    def from_array(cls, values: ArrayLike) -> DenseStorage:
        return cls(np.asarray(values))


@dataclass(frozen=True, slots=True, init=False)
class CSRStorage:
    """NumPy-backed compressed row storage for flattened multivector batches."""

    data: NDArray[Any]
    indices: NDArray[Any]
    indptr: NDArray[Any]
    _batch_shape: tuple[int, ...]
    _width: int
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

        if width_value < 0:
            raise ValueError("CSR width must be non-negative.")
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

        row_count = int(prod(normalized_batch_shape))
        if indptr_array.size != row_count + 1:
            raise ValueError("CSR indptr length must match flattened row_count + 1.")
        if int(indptr_array[0]) != 0:
            raise ValueError("CSR indptr must start at 0.")
        if np.any(indptr_array[1:] < indptr_array[:-1]):
            raise ValueError("CSR indptr must be nondecreasing.")
        if int(indptr_array[-1]) != data_array.size:
            raise ValueError("CSR indptr must end at the number of stored values.")
        if data_array.size and (np.any(index_array < 0) or np.any(index_array >= width_value)):
            raise ValueError("CSR indices must be between 0 and width - 1.")

        for row in range(row_count):
            start = int(indptr_array[row])
            stop = int(indptr_array[row + 1])
            row_indices = index_array[start:stop]
            if np.any(row_indices[1:] <= row_indices[:-1]):
                raise ValueError("CSR row indices must be strictly increasing.")

        object.__setattr__(self, "data", data_array)
        object.__setattr__(self, "indices", index_array)
        object.__setattr__(self, "indptr", indptr_array)
        object.__setattr__(self, "_batch_shape", normalized_batch_shape)
        object.__setattr__(self, "_width", width_value)
        object.__setattr__(self, "_dtype", resolved_dtype)
        object.__setattr__(self, "kind", "csr")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def width(self) -> int:
        return self._width

    @property
    def row_count(self) -> int:
        return int(self.indptr.size - 1)

    def as_dense(self) -> NDArray[Any]:
        dense = np.zeros((self.row_count, self.width), dtype=self.dtype)
        for row in range(self.row_count):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            if start == stop:
                continue
            dense[row, self.indices[start:stop]] = self.data[start:stop]
        return dense.reshape(self.batch_shape + (self.width,))

    def copy(self) -> CSRStorage:
        return CSRStorage(
            self.data.copy(),
            self.indices.copy(),
            self.indptr.copy(),
            batch_shape=self.batch_shape,
            width=self.width,
            dtype=self.dtype,
        )

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
        return cls(
            np.array([], dtype=dtype),
            np.array([], dtype=np.intp),
            np.zeros(row_count + 1, dtype=np.intp),
            batch_shape=normalized_batch_shape,
            width=width,
            dtype=dtype,
        )
