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
            scalar_layout = MVLayout.grade(self.algebra, 0)
            scalar = MVArray(
                algebra=self.algebra,
                layout=scalar_layout,
                values=np.asarray([other], dtype=np.result_type(self.dtype, other)),
            )
            return sub(scalar, self)
        return NotImplemented

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
