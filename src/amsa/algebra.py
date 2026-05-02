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

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from amsa.inspection import EntityInfo

from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.ops import (
    add as add_op,
)
from amsa.ops import (
    anticommutator_product as anticommutator_product_op,
)
from amsa.ops import (
    bulk as bulk_op,
)
from amsa.ops import (
    bulk_dual as bulk_dual_op,
)
from amsa.ops import (
    bulk_norm as bulk_norm_op,
)
from amsa.ops import (
    bulk_norm_squared as bulk_norm_squared_op,
)
from amsa.ops import (
    bulk_normalize as bulk_normalize_op,
)
from amsa.ops import (
    commutator_product as commutator_product_op,
)
from amsa.ops import (
    divide as divide_op,
)
from amsa.ops import (
    exp as exp_op,
)
from amsa.ops import (
    inner_product as inner_op,
)
from amsa.ops import (
    inverse as inverse_op,
)
from amsa.ops import (
    left_contraction as left_contraction_op,
)
from amsa.ops import (
    motor_exp as motor_exp_op,
)
from amsa.ops import (
    motor_log as motor_log_op,
)
from amsa.ops import (
    norm as norm_op,
)
from amsa.ops import (
    norm_squared as norm_squared_op,
)
from amsa.ops import (
    normalize as normalize_op,
)
from amsa.ops import (
    outer_product as outer_op,
)
from amsa.ops import (
    regressive_product as regressive_product_op,
)
from amsa.ops import (
    right_contraction as right_contraction_op,
)
from amsa.ops import (
    rigid_body_normalize as rigid_body_normalize_op,
)
from amsa.ops import (
    sandwich as sandwich_op,
)
from amsa.ops import (
    scalar_product as scalar_product_op,
)
from amsa.ops import (
    sub as sub_op,
)
from amsa.ops import (
    unitize as unitize_op,
)
from amsa.ops import (
    weight as weight_op,
)
from amsa.ops import (
    weight_dual as weight_dual_op,
)
from amsa.ops import (
    weight_norm as weight_norm_op,
)
from amsa.ops import (
    weight_norm_squared as weight_norm_squared_op,
)
from amsa.specs import AlgebraSpec
from amsa.specs import cga2d as cga2d_spec
from amsa.specs import cga3d as cga3d_spec
from amsa.specs import pga2d as pga2d_spec
from amsa.specs import pga3d as pga3d_spec
from amsa.specs import vga2d as vga2d_spec
from amsa.specs import vga3d as vga3d_spec
from amsa.storage import StorageRequest, resolve_storage_kind


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
    def cga2d(cls) -> Algebra:
        return cls(cga2d_spec())

    @classmethod
    def cga3d(cls) -> Algebra:
        return cls(cga3d_spec())

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
            "cga2d": cga2d_spec,
            "2dcga": cga2d_spec,
            "cga3d": cga3d_spec,
            "3dcga": cga3d_spec,
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

    def show_cayley(self, blades: tuple[int, ...] | None = None) -> str:
        """Display Cayley table subset in human-readable format.
        
        Args:
            blades: Optional tuple of blades to show. If None, shows first 8 blades
                    or all blades if algebra is small.
        """
        if blades is None:
            all_blades = list(range(self.spec.blade_count))
            blades = tuple(all_blades[:min(8, len(all_blades))])

        lines = [f"Cayley table for {self.spec.signature} ({len(blades)} blades)"]
        lines.append("  " + "  ".join(f"{self.spec.blade_name(b):>6}" for b in blades))
        
        for row_blade in blades:
            row_parts = [f"{self.spec.blade_name(row_blade):>6}"]
            for col_blade in blades:
                coeff, out_blade = self.spec.blade_product(row_blade, col_blade)
                if coeff == 0:
                    row_parts.append("     0")
                else:
                    coeff_str = f"{coeff:+d}" if coeff != 1 else "+"
                    out_name = self.spec.blade_name(out_blade)
                    row_parts.append(f"{coeff_str}{out_name:>5}")
            lines.append("  ".join(row_parts))
        
        return "\n".join(lines)

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
        backend: StorageRequest = "auto",
    ) -> MVArray:
        active_layout = layout if layout is not None else self.dense_layout()
        return MVArray.zeros(
            self.spec,
            active_layout,
            batch_shape=batch_shape,
            dtype=dtype,
            backend=backend,
        )

    def blade(
        self,
        key: int | str,
        value: Any = 1.0,
        *,
        backend: StorageRequest = "auto",
    ) -> MVArray:
        blade = self.spec.blade_from_key(key)
        layout = self.sparse_layout((blade,), name=self.spec.blade_name(blade))
        return self.multivector({blade: value}, layout=layout, backend=backend)

    def multivector(
        self,
        data: MVArray | Mapping[int | str, Any] | Any,
        *,
        layout: MVLayout | None = None,
        batch_shape: tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | type[np.float64] | None = None,
        backend: StorageRequest = "auto",
    ) -> MVArray:
        if isinstance(data, MVArray):
            if data.algebra != self.spec:
                raise ValueError("Cannot import a multivector from a different algebra.")
            requested_kind = resolve_storage_kind(backend, auto_kind=data.storage_kind)
            res = data
            if layout is not None:
                res = res.to_layout(layout)
            return res.with_storage(requested_kind)

        if isinstance(data, Mapping):
            normalized = {
                self.spec.blade_from_key(key): np.asarray(value) for key, value in data.items()
            }
            if layout is None:
                blades = tuple(sorted(normalized))
                layout = self.sparse_layout(blades, name="sparse")
            else:
                if layout.algebra != self.spec:
                    raise ValueError("layout must belong to this algebra.")

            values_list = list(normalized.values())
            if values_list:
                auto_batch_shape = np.broadcast_shapes(*(value.shape for value in values_list))
                auto_dtype = np.result_type(*(value.dtype for value in values_list))
            else:
                auto_batch_shape = ()
                auto_dtype = np.dtype(np.float64)
            
            final_batch_shape = batch_shape if batch_shape is not None else auto_batch_shape
            final_dtype = dtype if dtype is not None else auto_dtype
            
            result = np.zeros(final_batch_shape + (layout.size,), dtype=final_dtype)
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
                
                # Robust broadcasting: if value is (N,) and batch_shape is (N, 1),
                # manually expand it to align.
                v_arr = value
                if v_arr.ndim < len(final_batch_shape):
                    # Try to see if it's a simple prefix match
                    if v_arr.shape == final_batch_shape[:v_arr.ndim]:
                         # Add trailing dimensions
                         for _ in range(len(final_batch_shape) - v_arr.ndim):
                             v_arr = v_arr[..., np.newaxis]
                
                result[..., index] = np.broadcast_to(v_arr, final_batch_shape)
            return MVArray.from_array(self.spec, layout, result, backend=backend)

        array = np.asarray(data)
        if layout is None:
            layout = self.dense_layout()
        # Arrays carry their own batch shape and dtype.
        return MVArray.from_array(self.spec, layout, array, backend=backend)

    def scalar(self, value: Any = 0.0, *, backend: StorageRequest = "auto") -> MVArray:
        return self.multivector([value], layout=self.grade_layout(0), backend=backend)

    def kvector(self, grade: int, values: Any, *, backend: StorageRequest = "auto") -> MVArray:
        return self.multivector(values, layout=self.grade_layout(grade), backend=backend)

    def vector(self, values: Any, *, backend: StorageRequest = "auto") -> MVArray:
        return self.kvector(1, values, backend=backend)

    def bivector(self, values: Any, *, backend: StorageRequest = "auto") -> MVArray:
        return self.kvector(2, values, backend=backend)

    def trivector(self, values: Any, *, backend: StorageRequest = "auto") -> MVArray:
        return self.kvector(3, values, backend=backend)

    def even(self, values: Any, *, backend: StorageRequest = "auto") -> MVArray:
        return self.multivector(values, layout=self.even_layout(), backend=backend)

    def odd(self, values: Any, *, backend: StorageRequest = "auto") -> MVArray:
        return self.multivector(values, layout=self.odd_layout(), backend=backend)

    def pseudoscalar(self, value: Any = 0.0, *, backend: StorageRequest = "auto") -> MVArray:
        return self.multivector([value], layout=self.grade_layout(self.dimension), backend=backend)

    def _validate_cga(self) -> int:
        if self.dimension < 3 or self.signature[-2:] != (1, -1):
            raise ValueError("CGA helpers require an AMSA CGA algebra with signature (1^n, 1, -1).")
        return self.dimension - 2

    def origin(self, *, backend: StorageRequest = "auto") -> MVArray:
        """Return the conformal null origin vector ``n_o``."""
        from amsa.cga import origin as _cga_origin
        return _cga_origin(self, backend=backend)

    def infinity(self, *, backend: StorageRequest = "auto") -> MVArray:
        """Return the conformal null infinity vector ``n_inf``."""
        from amsa.cga import infinity as _cga_infinity
        return _cga_infinity(self, backend=backend)

    def euclidean_vector(
        self, coordinates: Any, *, backend: StorageRequest = "auto"
    ) -> MVArray:
        """Embed Euclidean coordinates in the Euclidean vector subspace of a CGA algebra."""
        from amsa.cga import euclidean_vector as _cga_euclidean_vector
        return _cga_euclidean_vector(self, coordinates, backend=backend)

    def point(
        self, coordinates: Any, *, backend: StorageRequest = "auto"
    ) -> MVArray:
        """Return the conformal point ``X = n_o + x + 0.5 * (x·x) n_inf``."""
        from amsa.cga import point as _cga_point
        return _cga_point(self, coordinates, backend=backend)

    def sphere(
        self, center: Any, radius: Any, *, backend: StorageRequest = "auto"
    ) -> MVArray:
        """Return a dual sphere ``S = C - 0.5 r^2 n_inf``."""
        from amsa.cga import sphere as _cga_sphere
        return _cga_sphere(self, center, radius, backend=backend)

    def plane(
        self, normal: Any, distance: Any, *, backend: StorageRequest = "auto"
    ) -> MVArray:
        """Return a dual plane ``P = n + d n_inf`` with Euclidean unit normal ``n``."""
        from amsa.cga import plane as _cga_plane
        return _cga_plane(self, normal, distance, backend=backend)

    def translate(
        self, displacement: Any, *, backend: StorageRequest = "auto"
    ) -> MVArray:
        """Return the CGA translator ``T = 1 - 0.5 t n_inf``."""
        from amsa.cga import translate as _cga_translate
        return _cga_translate(self, displacement, backend=backend)

    def line_through_points(self, a: MVArray, b: MVArray) -> MVArray:
        """Return the direct line through two conformal points."""
        from amsa.cga import line_through_points as _cga_line_through_points
        return _cga_line_through_points(self, a, b)

    def circle_through_points(self, a: MVArray, b: MVArray, c: MVArray) -> MVArray:
        """Return the direct circle through three conformal points."""
        from amsa.cga import circle_through_points as _cga_circle_through_points
        return _cga_circle_through_points(self, a, b, c)

    def distance_squared(self, a: MVArray, b: MVArray) -> Any:
        """Return Euclidean squared distance from normalized conformal points."""
        from amsa.cga import distance_squared as _cga_distance_squared
        return _cga_distance_squared(self, a, b)

    def extract_point(self, mv: MVArray) -> np.ndarray:
        """Return Euclidean point coordinates from a conformal point MV."""
        from amsa.cga import extract_point as _cga_extract_point
        return _cga_extract_point(mv)

    def extract_sphere(self, mv: MVArray) -> tuple[np.ndarray, np.ndarray]:
        """Return (center, radius) from a dual-sphere MV."""
        from amsa.cga import extract_sphere as _cga_extract_sphere
        return _cga_extract_sphere(mv)

    def extract_plane(self, mv: MVArray) -> tuple[np.ndarray, np.ndarray]:
        """Return (normal, signed_distance) from a dual-plane MV."""
        from amsa.cga import extract_plane as _cga_extract_plane
        return _cga_extract_plane(mv)

    def extract_euclidean_vector(self, mv: MVArray) -> np.ndarray:
        """Return Euclidean coordinates from a Euclidean vector subspace MV."""
        from amsa.cga import extract_euclidean_vector as _cga_extract_euclidean_vector
        return _cga_extract_euclidean_vector(mv)

    def classify(self, mv: MVArray) -> EntityInfo:
        """Return a geometric interpretation of *mv* for this algebra.

        Routes to the appropriate model-specific classifier (CGA, PGA, VGA)
        based on the algebra signature.
        """
        signature = self.signature

        if len(signature) >= 2 and signature[-2:] == (1, -1):
            from amsa.inspection import classify_cga
            return classify_cga(self, mv)

        if signature[0] == 0:
            from amsa.inspection import classify_pga
            return classify_pga(self, mv)

        from amsa.inspection import classify_vga

        return classify_vga(self, mv)

    def gp(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return lhs * rhs

    def outer(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return outer_op(lhs, rhs)

    def inner(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return inner_op(lhs, rhs)

    def scalar_product(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return scalar_product_op(lhs, rhs)

    def commutator(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return commutator_product_op(lhs, rhs)

    def anticommutator(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return anticommutator_product_op(lhs, rhs)

    def bulk(self, mv: MVArray) -> MVArray:
        return bulk_op(mv)

    def weight(self, mv: MVArray) -> MVArray:
        return weight_op(mv)

    def bulk_dual(self, mv: MVArray) -> MVArray:
        return bulk_dual_op(mv)

    def weight_dual(self, mv: MVArray) -> MVArray:
        return weight_dual_op(mv)

    def norm_squared(self, mv: MVArray) -> MVArray:
        return norm_squared_op(mv)

    def norm(self, mv: MVArray) -> MVArray:
        return norm_op(mv)

    def normalize(self, mv: MVArray) -> MVArray:
        return normalize_op(mv)

    def exp(self, mv: MVArray) -> MVArray:
        return exp_op(mv)

    def motor_exp(self, mv: MVArray) -> MVArray:
        return motor_exp_op(mv)

    def motor_log(self, mv: MVArray) -> MVArray:
        return motor_log_op(mv)

    def log(self, mv: MVArray) -> MVArray:
        from amsa.ops import log as log_op

        return log_op(mv)

    def bulk_norm_squared(self, mv: MVArray) -> MVArray:
        return bulk_norm_squared_op(mv)

    def bulk_norm(self, mv: MVArray) -> MVArray:
        return bulk_norm_op(mv)

    def weight_norm_squared(self, mv: MVArray) -> MVArray:
        return weight_norm_squared_op(mv)

    def weight_norm(self, mv: MVArray) -> MVArray:
        return weight_norm_op(mv)

    def bulk_normalize(self, mv: MVArray) -> MVArray:
        return bulk_normalize_op(mv)

    def unitize(self, mv: MVArray) -> MVArray:
        return unitize_op(mv)

    def rigid_body_normalize(self, mv: MVArray) -> MVArray:
        return rigid_body_normalize_op(mv)

    def left_contract(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return left_contraction_op(lhs, rhs)

    def right_contract(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return right_contraction_op(lhs, rhs)

    def regress(self, lhs: MVArray, rhs: MVArray) -> MVArray:
        return regressive_product_op(lhs, rhs)

    def sandwich(self, actor: MVArray, target: MVArray) -> MVArray:
        return sandwich_op(actor, target)

    def inverse(self, mv: MVArray) -> MVArray:
        return inverse_op(mv)

    def add(self, lhs: MVArray | Any, rhs: MVArray | Any) -> MVArray:
        left = self.scalar(lhs) if np.isscalar(lhs) else self.multivector(lhs)
        return add_op(left, rhs)

    def sub(self, lhs: MVArray | Any, rhs: MVArray | Any) -> MVArray:
        left = self.scalar(lhs) if np.isscalar(lhs) else self.multivector(lhs)
        return sub_op(left, rhs)

    def div(self, lhs: MVArray | Any, rhs: MVArray | Any) -> MVArray:
        if np.isscalar(lhs):
            lhs = self.scalar(lhs)
        else:
            lhs = self.multivector(lhs)

        if isinstance(rhs, Number):
            return divide_op(lhs, rhs)
        if isinstance(rhs, MVArray):
            return divide_op(lhs, rhs)
        return divide_op(lhs, self.multivector(rhs))
