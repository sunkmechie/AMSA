from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from amsa.layouts import MVLayout
from amsa.specs import AlgebraSpec


@dataclass(frozen=True, slots=True)
class MVArray:
    """Array-backed multivector values paired with an algebra and layout."""

    algebra: AlgebraSpec
    layout: MVLayout
    values: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.layout.algebra != self.algebra:
            raise ValueError("layout.algebra must match algebra.")

        if self.values.ndim == 0:
            raise ValueError("values must have at least one dimension.")

        expected = self.layout.size
        if self.values.shape[-1] != expected:
            raise ValueError(
                f"Last axis of values must match layout size {expected}, got {self.values.shape[-1]}."
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.values.shape[:-1]

    @property
    def dtype(self) -> np.dtype[np.floating]:
        return self.values.dtype

    @classmethod
    def zeros(
        cls,
        algebra: AlgebraSpec,
        layout: MVLayout,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[np.floating] | type[np.float64] = np.float64,
    ) -> MVArray:
        values = np.zeros(batch_shape + (layout.size,), dtype=dtype)
        return cls(algebra=algebra, layout=layout, values=values)

    def copy(self) -> MVArray:
        return MVArray(algebra=self.algebra, layout=self.layout, values=self.values.copy())
