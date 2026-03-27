from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

StorageKind = Literal["dense", "csr"]


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
