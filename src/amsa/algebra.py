from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.ops import add as add_op
from amsa.ops import inner_product as inner_op
from amsa.ops import outer_product as outer_op
from amsa.ops import sub as sub_op
from amsa.specs import AlgebraSpec
from amsa.specs import pga2d as pga2d_spec
from amsa.specs import pga3d as pga3d_spec
from amsa.specs import vga2d as vga2d_spec
from amsa.specs import vga3d as vga3d_spec


@dataclass(frozen=True, slots=True)
class Algebra:
    """User-facing algebra handle for the initial scaffold."""

    spec: AlgebraSpec

    @classmethod
    def vga2d(cls) -> Algebra:
        return cls(vga2d_spec())

    @classmethod
    def vga3d(cls) -> Algebra:
        return cls(vga3d_spec())

    @classmethod
    def pga2d(cls) -> Algebra:
        return cls(pga2d_spec())

    @classmethod
    def pga3d(cls) -> Algebra:
        return cls(pga3d_spec())

    @classmethod
    def from_name(cls, name: str) -> Algebra:
        normalized = "".join(char for char in name.casefold() if char.isalnum())
        presets = {
            "vga2d": vga2d_spec,
            "vga3d": vga3d_spec,
            "pga2d": pga2d_spec,
            "2dpga": pga2d_spec,
            "pga3d": pga3d_spec,
            "3dpga": pga3d_spec,
        }
        try:
            return cls(presets[normalized]())
        except KeyError as exc:
            supported = ", ".join(sorted(presets))
            message = f"Unknown algebra preset {name!r}. Supported presets: {supported}."
            raise ValueError(message) from exc

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

    def even_layout(self) -> MVLayout:
        return self.grade_layout(*range(0, self.dimension + 1, 2))

    def odd_layout(self) -> MVLayout:
        return self.grade_layout(*range(1, self.dimension + 1, 2))

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
            normalized = {
                self.spec.blade_from_key(key): np.asarray(value)
                for key, value in data.items()
            }
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
                dtype = np.dtype(np.float64)
            result = np.zeros(batch_shape + (layout.size,), dtype=dtype)
            blade_to_index = {blade: idx for idx, blade in enumerate(layout.blades)}

            for blade, value in normalized.items():
                try:
                    index = blade_to_index[blade]
                except KeyError as exc:
                    message = (
                        f"Blade {self.spec.blade_name(blade)} is not present "
                        f"in layout {layout.name}."
                    )
                    raise ValueError(message) from exc
                result[..., index] = np.broadcast_to(value, batch_shape)
            return MVArray(algebra=self.spec, layout=layout, values=result)

        array = np.asarray(data)
        if layout is None:
            layout = self.dense_layout()
        return MVArray.from_array(self.spec, layout, array)

    def scalar(self, value: Any = 0.0) -> MVArray:
        return self.multivector([value], layout=self.grade_layout(0))

    def kvector(self, grade: int, values: Any) -> MVArray:
        return self.multivector(values, layout=self.grade_layout(grade))

    def vector(self, values: Any) -> MVArray:
        return self.kvector(1, values)

    def bivector(self, values: Any) -> MVArray:
        return self.kvector(2, values)

    def trivector(self, values: Any) -> MVArray:
        return self.kvector(3, values)

    def even(self, values: Any) -> MVArray:
        return self.multivector(values, layout=self.even_layout())

    def odd(self, values: Any) -> MVArray:
        return self.multivector(values, layout=self.odd_layout())

    def pseudoscalar(self, value: Any = 0.0) -> MVArray:
        return self.multivector([value], layout=self.grade_layout(self.dimension))

    def gp(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return lhs * rhs

    def outer(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return outer_op(lhs, rhs)

    def inner(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return inner_op(lhs, rhs)

    def add(self, lhs: MVArray | Any, rhs: MVArray | Any) -> MVArray:
        left = self.scalar(lhs) if np.isscalar(lhs) else self.multivector(lhs)
        return add_op(left, rhs)

    def sub(self, lhs: MVArray | Any, rhs: MVArray | Any) -> MVArray:
        left = self.scalar(lhs) if np.isscalar(lhs) else self.multivector(lhs)
        return sub_op(left, rhs)
