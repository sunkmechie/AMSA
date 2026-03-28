from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amsa.layouts import MVLayout
from amsa.specs import AlgebraSpec
from amsa.storage import (
    DenseStorage,
    MVStorage,
    project_storage,
    scale_storage,
    storage_component,
    to_dense_storage,
)


@dataclass(frozen=True, slots=True, init=False)
class MVArray:
    """Storage-backed multivector values paired with an algebra and layout."""

    algebra: AlgebraSpec
    layout: MVLayout
    storage: MVStorage

    def __init__(
        self,
        algebra: AlgebraSpec,
        layout: MVLayout,
        values: ArrayLike | None = None,
        *,
        storage: MVStorage | None = None,
    ) -> None:
        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "layout", layout)
        if self.layout.algebra != self.algebra:
            raise ValueError("layout.algebra must match algebra.")

        if (values is None) == (storage is None):
            raise ValueError("Provide exactly one of values or storage.")

        if storage is None:
            assert values is not None
            active_storage: MVStorage = DenseStorage.from_array(values)
        else:
            active_storage = storage

        object.__setattr__(self, "storage", active_storage)
        expected = self.layout.size
        if self.storage.width != expected:
            raise ValueError(
                f"Last axis of values must match layout size {expected}, got {self.storage.width}."
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.storage.batch_shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.storage.dtype

    @property
    def values(self) -> NDArray[Any]:
        return self.storage.as_dense()

    @property
    def storage_kind(self) -> str:
        return self.storage.kind

    @property
    def grades(self) -> tuple[int, ...]:
        return self.layout.grades

    @classmethod
    def zeros(
        cls,
        algebra: AlgebraSpec,
        layout: MVLayout,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[Any] | type[np.float64] = np.float64,
    ) -> MVArray:
        return cls(
            algebra=algebra,
            layout=layout,
            storage=DenseStorage.zeros(layout.size, batch_shape=batch_shape, dtype=dtype),
        )

    @classmethod
    def from_array(
        cls,
        algebra: AlgebraSpec,
        layout: MVLayout,
        values: ArrayLike,
    ) -> MVArray:
        return cls(algebra=algebra, layout=layout, storage=DenseStorage.from_array(values))

    def copy(self) -> MVArray:
        return MVArray(algebra=self.algebra, layout=self.layout, storage=self.storage.copy())

    def to_layout(self, layout: MVLayout) -> MVArray:
        if layout.algebra != self.algebra:
            raise ValueError("Target layout must belong to the same algebra.")
        if layout == self.layout:
            return self.copy()

        source_index = {blade: idx for idx, blade in enumerate(self.layout.blades)}
        columns = tuple(source_index.get(blade) for blade in layout.blades)
        return MVArray(
            algebra=self.algebra,
            layout=layout,
            storage=project_storage(self.storage, columns),
        )

    def as_dense(self) -> MVArray:
        dense_layout = MVLayout.dense(self.algebra)
        projected = self if self.layout == dense_layout else self.to_layout(dense_layout)
        return MVArray(
            algebra=self.algebra,
            layout=dense_layout,
            storage=to_dense_storage(projected.storage),
        )

    def component(self, key: int | str) -> Any:
        blade = self.algebra.blade_from_key(key)
        for index, candidate in enumerate(self.layout.blades):
            if candidate == blade:
                component = storage_component(self.storage, index)
                if self.batch_shape:
                    return component
                return component[()]
        if self.batch_shape:
            return np.zeros(self.batch_shape, dtype=self.dtype)
        return self.dtype.type(0)

    def grade(self, *grades: int) -> MVArray:
        from amsa.ops import project_grades

        return project_grades(self, *grades)

    def reverse(self) -> MVArray:
        from amsa.ops import reverse

        return reverse(self)

    def involute(self) -> MVArray:
        from amsa.ops import involute

        return involute(self)

    def conjugate(self) -> MVArray:
        from amsa.ops import conjugate

        return conjugate(self)

    def outer(self, other: MVArray) -> MVArray:
        from amsa.ops import outer_product

        return outer_product(self, other)

    def inner(self, other: MVArray) -> MVArray:
        from amsa.ops import inner_product

        return inner_product(self, other)

    def __neg__(self) -> MVArray:
        from amsa.ops import neg

        return neg(self)

    def __add__(self, other: MVArray) -> MVArray:
        from amsa.ops import add

        try:
            return add(self, other)
        except TypeError:
            return NotImplemented

    def __radd__(self, other: Number) -> MVArray:
        from amsa.ops import add

        try:
            return add(self, other)
        except TypeError:
            return NotImplemented

    def __sub__(self, other: MVArray) -> MVArray:
        from amsa.ops import sub

        try:
            return sub(self, other)
        except TypeError:
            return NotImplemented

    def __rsub__(self, other: Number) -> MVArray:
        from amsa.ops import sub

        if isinstance(other, Number):
            other_array = np.asarray(other)
            scalar_layout = MVLayout.grade(self.algebra, 0)
            scalar = MVArray(
                algebra=self.algebra,
                layout=scalar_layout,
                values=np.asarray([other], dtype=np.result_type(self.dtype, other_array.dtype)),
            )
            return sub(scalar, self)
        return NotImplemented

    def __mul__(self, other: MVArray | Number) -> MVArray:
        if isinstance(other, MVArray):
            from amsa.ops import geometric_product

            return geometric_product(self, other)
        if isinstance(other, Number):
            return MVArray(
                algebra=self.algebra,
                layout=self.layout,
                storage=scale_storage(self.storage, other),
            )
        return NotImplemented

    def __rmul__(self, other: Number) -> MVArray:
        if isinstance(other, Number):
            return MVArray(
                algebra=self.algebra,
                layout=self.layout,
                storage=scale_storage(self.storage, other),
            )
        return NotImplemented

    def __xor__(self, other: MVArray) -> MVArray:
        from amsa.ops import outer_product

        try:
            return outer_product(self, other)
        except TypeError:
            return NotImplemented

    def __or__(self, other: MVArray) -> MVArray:
        from amsa.ops import inner_product

        try:
            return inner_product(self, other)
        except TypeError:
            return NotImplemented
