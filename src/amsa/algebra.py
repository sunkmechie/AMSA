from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.specs import AlgebraSpec


@dataclass(frozen=True, slots=True)
class Algebra:
    """User-facing algebra handle for the initial scaffold."""

    spec: AlgebraSpec

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    @property
    def signature(self) -> tuple[int, ...]:
        return self.spec.signature

    def dense_layout(self) -> MVLayout:
        return MVLayout.dense(self.spec)

    def grade_layout(self, *grades: int) -> MVLayout:
        return MVLayout.grade(self.spec, *grades)

    def sparse_layout(self, blades: tuple[int, ...], *, name: str = "sparse") -> MVLayout:
        return MVLayout.sparse_pattern(self.spec, blades, name=name)

    def zeros(
        self,
        layout: MVLayout | None = None,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: np.dtype[Any] | type[np.float64] = np.float64,
    ) -> MVArray:
        active_layout = layout if layout is not None else self.dense_layout()
        return MVArray.zeros(self.spec, active_layout, batch_shape=batch_shape, dtype=dtype)

    def blade(self, key: int | str, value: Any = 1.0) -> MVArray:
        blade = self.spec.blade_from_key(key)
        layout = self.sparse_layout((blade,), name=self.spec.blade_name(blade))
        return self.multivector({blade: value}, layout=layout)

    def multivector(
        self,
        data: MVArray | Mapping[int | str, Any] | Any,
        *,
        layout: MVLayout | None = None,
    ) -> MVArray:
        if isinstance(data, MVArray):
            if data.algebra != self.spec:
                raise ValueError("Cannot import a multivector from a different algebra.")
            if layout is None:
                return data.copy()
            return data.to_layout(layout)

        if isinstance(data, Mapping):
            normalized = {self.spec.blade_from_key(key): np.asarray(value) for key, value in data.items()}
            if layout is None:
                blades = tuple(sorted(normalized))
                layout = self.sparse_layout(blades, name="sparse")
            else:
                if layout.algebra != self.spec:
                    raise ValueError("layout must belong to this algebra.")

            values_list = list(normalized.values())
            if values_list:
                batch_shape = np.broadcast_shapes(*(value.shape for value in values_list))
                dtype = np.result_type(*(value.dtype for value in values_list))
            else:
                batch_shape = ()
                dtype = np.float64
            result = np.zeros(batch_shape + (layout.size,), dtype=dtype)
            blade_to_index = {blade: idx for idx, blade in enumerate(layout.blades)}

            for blade, value in normalized.items():
                try:
                    index = blade_to_index[blade]
                except KeyError as exc:
                    raise ValueError(f"Blade {self.spec.blade_name(blade)} is not present in layout {layout.name}.") from exc
                result[..., index] = np.broadcast_to(value, batch_shape)
            return MVArray(algebra=self.spec, layout=layout, values=result)

        array = np.asarray(data)
        if layout is None:
            layout = self.dense_layout()
        return MVArray.from_array(self.spec, layout, array)

    def scalar(self, value: Any = 0.0) -> MVArray:
        return self.multivector([value], layout=self.grade_layout(0))

    def vector(self, values: Any) -> MVArray:
        return self.multivector(values, layout=self.grade_layout(1))

    def bivector(self, values: Any) -> MVArray:
        return self.multivector(values, layout=self.grade_layout(2))

    def pseudoscalar(self, value: Any = 0.0) -> MVArray:
        return self.multivector([value], layout=self.grade_layout(self.dimension))

    def gp(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return lhs * rhs
