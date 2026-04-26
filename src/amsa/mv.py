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
from numbers import Number
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amsa.layouts import MVLayout
from amsa.specs import AlgebraSpec
from amsa.storage import (
    MVStorage,
    StorageKind,
    StorageRequest,
    build_storage_from_array,
    build_zero_storage,
    convert_storage_kind,
    project_storage,
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
            if values is None:
                raise ValueError("values must be provided when storage is omitted.")
            active_storage = build_storage_from_array(values, kind="dense")
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
    def storage_kind(self) -> StorageKind:
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
        backend: StorageRequest = "dense",
    ) -> MVArray:
        return cls(
            algebra=algebra,
            layout=layout,
            storage=build_zero_storage(
                layout.size,
                batch_shape=batch_shape,
                dtype=dtype,
                kind=backend,
            ),
        )

    @classmethod
    def from_array(
        cls,
        algebra: AlgebraSpec,
        layout: MVLayout,
        values: ArrayLike,
        *,
        backend: StorageRequest = "dense",
    ) -> MVArray:
        return cls(
            algebra=algebra,
            layout=layout,
            storage=build_storage_from_array(values, kind=backend),
        )

    def __getitem__(self, key: Any) -> MVArray:
        """Index or slice the multivector batch."""
        from amsa.storage import DenseStorage, index_dense_storage
        
        if isinstance(self.storage, DenseStorage):
            new_storage = index_dense_storage(self.storage, key)
            return MVArray(self.algebra, self.layout, storage=new_storage)
        
        # Fallback for CSR or other storage: convert to dense for now
        # TODO: Implement sparse-aware indexing in storage.py
        dense_storage = self.with_storage("dense").storage.as_dense()
        new_array = dense_storage[key]
        return MVArray(self.algebra, self.layout, storage=DenseStorage.from_array(new_array))

    def copy(self) -> MVArray:
        return MVArray(algebra=self.algebra, layout=self.layout, storage=self.storage.copy())

    def with_storage(self, kind: StorageKind) -> MVArray:
        return MVArray(
            algebra=self.algebra,
            layout=self.layout,
            storage=convert_storage_kind(self.storage, kind),
        )

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

    def dual(self) -> MVArray:
        from amsa.ops import dual

        return dual(self)

    def undual(self) -> MVArray:
        from amsa.ops import undual

        return undual(self)

    def poincare_dual(self) -> MVArray:
        from amsa.ops import poincare_dual

        return poincare_dual(self)

    def poincare_undual(self) -> MVArray:
        from amsa.ops import poincare_undual

        return poincare_undual(self)

    def outer(self, other: MVArray) -> MVArray:
        from amsa.ops import outer_product

        return outer_product(self, other)

    def inner(self, other: MVArray) -> MVArray:
        from amsa.ops import inner_product

        return inner_product(self, other)

    def scalar_product(self, other: MVArray) -> MVArray:
        from amsa.ops import scalar_product

        return scalar_product(self, other)

    def commutator(self, other: MVArray) -> MVArray:
        from amsa.ops import commutator_product

        return commutator_product(self, other)

    def anticommutator(self, other: MVArray) -> MVArray:
        from amsa.ops import anticommutator_product

        return anticommutator_product(self, other)

    def bulk(self) -> MVArray:
        from amsa.ops import bulk

        return bulk(self)

    def weight(self) -> MVArray:
        from amsa.ops import weight

        return weight(self)

    def bulk_dual(self) -> MVArray:
        from amsa.ops import bulk_dual

        return bulk_dual(self)

    def weight_dual(self) -> MVArray:
        from amsa.ops import weight_dual

        return weight_dual(self)

    def norm_squared(self) -> MVArray:
        from amsa.ops import norm_squared

        return norm_squared(self)

    def norm(self) -> MVArray:
        from amsa.ops import norm

        return norm(self)

    def normalized(self) -> MVArray:
        from amsa.ops import normalize

        return normalize(self)

    def exp(self) -> MVArray:
        from amsa.ops import exp

        return exp(self)

    def motor_exp(self) -> MVArray:
        from amsa.ops import motor_exp

        return motor_exp(self)

    def motor_log(self) -> MVArray:
        from amsa.ops import motor_log

        return motor_log(self)

    def bulk_norm_squared(self) -> MVArray:
        from amsa.ops import bulk_norm_squared

        return bulk_norm_squared(self)

    def bulk_norm(self) -> MVArray:
        from amsa.ops import bulk_norm

        return bulk_norm(self)

    def weight_norm_squared(self) -> MVArray:
        from amsa.ops import weight_norm_squared

        return weight_norm_squared(self)

    def weight_norm(self) -> MVArray:
        from amsa.ops import weight_norm

        return weight_norm(self)

    def bulk_normalized(self) -> MVArray:
        from amsa.ops import bulk_normalize

        return bulk_normalize(self)

    def unitized(self) -> MVArray:
        from amsa.ops import unitize

        return unitize(self)

    def rigid_body_normalized(self) -> MVArray:
        from amsa.ops import rigid_body_normalize

        return rigid_body_normalize(self)

    def left_contract(self, other: MVArray) -> MVArray:
        from amsa.ops import left_contraction

        return left_contraction(self, other)

    def right_contract(self, other: MVArray) -> MVArray:
        from amsa.ops import right_contraction

        return right_contraction(self, other)

    def regress(self, other: MVArray) -> MVArray:
        from amsa.ops import regressive_product

        return regressive_product(self, other)

    def sandwich(self, other: MVArray) -> MVArray:
        from amsa.ops import sandwich

        return sandwich(self, other)

    def inverse(self) -> MVArray:
        from amsa.ops import inverse

        return inverse(self)

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
            from amsa.ops import scale

            return scale(self, other)
        return NotImplemented

    def __rmul__(self, other: Number) -> MVArray:
        if isinstance(other, Number):
            from amsa.ops import scale

            return scale(self, other)
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

    def __truediv__(self, other: MVArray | Number) -> MVArray:
        from amsa.ops import divide

        try:
            return divide(self, other)
        except TypeError:
            return NotImplemented

    def __rtruediv__(self, other: Number) -> MVArray:
        from amsa.ops import divide

        if isinstance(other, Number):
            return divide(other, self)
        return NotImplemented
