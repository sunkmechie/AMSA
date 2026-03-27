from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amsa.layouts import MVLayout
from amsa.specs import AlgebraSpec


@dataclass(frozen=True, slots=True)
class MVArray:
    """Array-backed multivector values paired with an algebra and layout."""

    algebra: AlgebraSpec
    layout: MVLayout
    values: NDArray[Any]

    def __post_init__(self) -> None:
        if self.layout.algebra != self.algebra:
            raise ValueError("layout.algebra must match algebra.")

        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        if values.ndim == 0:
            raise ValueError("values must have at least one dimension.")

        expected = self.layout.size
        if values.shape[-1] != expected:
            raise ValueError(
                f"Last axis of values must match layout size {expected}, got {values.shape[-1]}."
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.values.shape[:-1]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @classmethod
    def zeros(
        cls,
        algebra: AlgebraSpec,
        layout: MVLayout,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[Any] | type[np.float64] = np.float64,
    ) -> MVArray:
        values = np.zeros(batch_shape + (layout.size,), dtype=dtype)
        return cls(algebra=algebra, layout=layout, values=values)

    @classmethod
    def from_array(
        cls,
        algebra: AlgebraSpec,
        layout: MVLayout,
        values: ArrayLike,
    ) -> MVArray:
        return cls(algebra=algebra, layout=layout, values=np.asarray(values))

    def copy(self) -> MVArray:
        return MVArray(algebra=self.algebra, layout=self.layout, values=self.values.copy())

    def to_layout(self, layout: MVLayout) -> MVArray:
        if layout.algebra != self.algebra:
            raise ValueError("Target layout must belong to the same algebra.")
        if layout == self.layout:
            return self.copy()

        result = np.zeros(self.batch_shape + (layout.size,), dtype=self.dtype)
        source_index = {blade: idx for idx, blade in enumerate(self.layout.blades)}
        for out_idx, blade in enumerate(layout.blades):
            in_idx = source_index.get(blade)
            if in_idx is not None:
                result[..., out_idx] = self.values[..., in_idx]
        return MVArray(algebra=self.algebra, layout=layout, values=result)

    def as_dense(self) -> MVArray:
        return self.to_layout(MVLayout.dense(self.algebra))

    def component(self, key: int | str) -> Any:
        blade = self.algebra.blade_from_key(key)
        for index, candidate in enumerate(self.layout.blades):
            if candidate == blade:
                return self.values[..., index]
        if self.batch_shape:
            return np.zeros(self.batch_shape, dtype=self.dtype)
        return self.values.dtype.type(0)

    def grade(self, *grades: int) -> MVArray:
        if len(grades) == 1 and isinstance(grades[0], tuple):
            grades = grades[0]
        selected = tuple(
            blade for blade in self.layout.blades if blade.bit_count() in set(grades)
        )
        layout = MVLayout.sparse_pattern(self.algebra, selected, name="grade-select")
        return self.to_layout(layout)

    def reverse(self) -> MVArray:
        from amsa.ops import reverse

        return reverse(self)

    def involute(self) -> MVArray:
        from amsa.ops import involute

        return involute(self)

    def conjugate(self) -> MVArray:
        from amsa.ops import conjugate

        return conjugate(self)

    def __neg__(self) -> MVArray:
        from amsa.ops import neg

        return neg(self)

    def __add__(self, other: MVArray) -> MVArray:
        if not isinstance(other, MVArray):
            return NotImplemented
        from amsa.ops import add

        return add(self, other)

    def __sub__(self, other: MVArray) -> MVArray:
        if not isinstance(other, MVArray):
            return NotImplemented
        from amsa.ops import sub

        return sub(self, other)

    def __mul__(self, other: MVArray | Number) -> MVArray:
        if isinstance(other, MVArray):
            from amsa.ops import geometric_product

            return geometric_product(self, other)
        if isinstance(other, Number):
            return MVArray(algebra=self.algebra, layout=self.layout, values=self.values * other)
        return NotImplemented

    def __rmul__(self, other: Number) -> MVArray:
        if isinstance(other, Number):
            return MVArray(algebra=self.algebra, layout=self.layout, values=other * self.values)
        return NotImplemented
