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
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

_PRECOMPUTE_BASIS_PRODUCT_MAX_BLADE_COUNT = 512


def _blade_product_raw(
    signature: tuple[int, ...],
    lhs: int,
    rhs: int,
) -> tuple[int, int]:
    blade_count = 1 << len(signature)
    if lhs < 0 or lhs >= blade_count or rhs < 0 or rhs >= blade_count:
        raise ValueError("Blade is outside the algebra basis.")

    coefficient = 1
    remaining = lhs
    while remaining:
        bit = remaining & -remaining
        if (rhs & (bit - 1)).bit_count() % 2:
            coefficient = -coefficient
        remaining ^= bit

    overlap = lhs & rhs
    while overlap:
        bit = overlap & -overlap
        axis = bit.bit_length() - 1
        metric = signature[axis]
        if metric == 0:
            return 0, 0
        coefficient *= metric
        overlap ^= bit

    return coefficient, lhs ^ rhs


def _blade_index_dtype(
    blade_count: int,
) -> type[np.uint8] | type[np.uint16] | type[np.uint32]:
    if blade_count <= np.iinfo(np.uint8).max + 1:
        return np.uint8
    if blade_count <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    return np.uint32


@dataclass(frozen=True, slots=True)
class BasisProductTable:
    """Compact numeric basis-blade multiplication table for one algebra signature."""

    coefficients: NDArray[np.int8]
    output_blades: NDArray[Any]
    grades: NDArray[np.uint8]

    def __post_init__(self) -> None:
        blade_count = self.coefficients.shape[0]
        if self.coefficients.shape != (blade_count, blade_count):
            raise ValueError("coefficients must be a square blade_count x blade_count array.")
        if self.output_blades.shape != (blade_count, blade_count):
            raise ValueError("output_blades must match the coefficients shape.")
        if self.grades.shape != (blade_count,):
            raise ValueError("grades must have length blade_count.")

    @property
    def blade_count(self) -> int:
        return int(self.coefficients.shape[0])

    def blade_product(self, lhs: int, rhs: int) -> tuple[int, int]:
        return int(self.coefficients[lhs, rhs]), int(self.output_blades[lhs, rhs])


@lru_cache(maxsize=4096)
def _blade_product_cached(
    signature: tuple[int, ...],
    lhs: int,
    rhs: int,
) -> tuple[int, int]:
    return _blade_product_raw(signature, lhs, rhs)


@lru_cache(maxsize=256)
def _blade_name_lookup(dimension: int, start_index: int) -> dict[str, int]:
    return {
        canonical_blade_name(blade, dimension=dimension, start_index=start_index): blade
        for blade in range(1 << dimension)
    }


def grade_of_blade(blade: int) -> int:
    """Return the grade of a blade encoded as a bit pattern."""
    if blade < 0:
        raise ValueError("Blade bit patterns must be non-negative.")
    return blade.bit_count()


def canonical_blade_name(blade: int, *, dimension: int, start_index: int = 1) -> str:
    """Return the canonical basis-blade name for a blade bit pattern."""
    if blade < 0:
        raise ValueError("Blade bit patterns must be non-negative.")
    if blade >= (1 << dimension):
        raise ValueError("Blade bit pattern exceeds the algebra dimension.")
    if blade == 0:
        return "e"

    parts: list[str] = []
    for axis in range(dimension):
        if blade & (1 << axis):
            parts.append(str(axis + start_index))
    return "e" + "".join(parts)


@lru_cache(maxsize=32)
def _basis_product_table_cached(signature: tuple[int, ...]) -> BasisProductTable | None:
    blade_count = 1 << len(signature)
    if blade_count > _PRECOMPUTE_BASIS_PRODUCT_MAX_BLADE_COUNT:
        return None

    coefficients = np.zeros((blade_count, blade_count), dtype=np.int8)
    output_dtype = _blade_index_dtype(blade_count)
    output_blades = np.zeros((blade_count, blade_count), dtype=output_dtype)
    grades = np.fromiter((grade_of_blade(blade) for blade in range(blade_count)), dtype=np.uint8)

    for lhs in range(blade_count):
        for rhs in range(blade_count):
            coefficient, out_blade = _blade_product_raw(signature, lhs, rhs)
            coefficients[lhs, rhs] = coefficient
            output_blades[lhs, rhs] = out_blade

    return BasisProductTable(
        coefficients=coefficients,
        output_blades=output_blades,
        grades=grades,
    )


def _build_cayley_entries(
    signature: tuple[int, ...],
    start_index: int,
) -> tuple[tuple[tuple[str, str], str], ...]:
    dimension = len(signature)
    blade_count = 1 << dimension
    blade_names = tuple(
        canonical_blade_name(blade, dimension=dimension, start_index=start_index)
        for blade in range(blade_count)
    )
    table = _basis_product_table_cached(signature)

    entries: list[tuple[tuple[str, str], str]] = []
    for lhs in range(blade_count):
        lhs_name = blade_names[lhs]
        for rhs in range(blade_count):
            rhs_name = blade_names[rhs]
            if table is not None:
                coefficient, out_blade = table.blade_product(lhs, rhs)
            else:
                coefficient, out_blade = _blade_product_raw(signature, lhs, rhs)

            if coefficient == 0:
                value = "0"
            else:
                sign = "-" if coefficient < 0 else ""
                value = sign + blade_names[out_blade]
            entries.append(((lhs_name, rhs_name), value))

    return tuple(entries)


@lru_cache(maxsize=16)
def _cayley_entries_cached(
    signature: tuple[int, ...],
    start_index: int,
) -> tuple[tuple[tuple[str, str], str], ...]:
    return _build_cayley_entries(signature, start_index)


@dataclass(frozen=True, slots=True)
class AlgebraSpec:
    """Mathematical description of a Clifford algebra."""

    signature: tuple[int, ...]
    start_index: int = 1
    basis_prefix: str = "e"

    def __post_init__(self) -> None:
        if not self.signature:
            raise ValueError("An algebra must have at least one basis vector.")
        if any(value not in (-1, 0, 1) for value in self.signature):
            raise ValueError("Signature entries must be -1, 0, or 1.")
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative.")
        if self.basis_prefix != "e":
            raise ValueError("The initial scaffold supports only the canonical 'e' basis prefix.")

    @property
    def dimension(self) -> int:
        return len(self.signature)

    @property
    def blade_count(self) -> int:
        return 1 << self.dimension

    @property
    def p(self) -> int:
        return sum(1 for value in self.signature if value == 1)

    @property
    def q(self) -> int:
        return sum(1 for value in self.signature if value == -1)

    @property
    def r(self) -> int:
        return sum(1 for value in self.signature if value == 0)

    def grades(self) -> tuple[int, ...]:
        return tuple(range(self.dimension + 1))

    def validate_blade(self, blade: int) -> int:
        if blade < 0 or blade >= self.blade_count:
            raise ValueError(f"Blade {blade} is outside the algebra basis.")
        return blade

    def blade_name(self, blade: int) -> str:
        self.validate_blade(blade)
        return canonical_blade_name(
            blade,
            dimension=self.dimension,
            start_index=self.start_index,
        )

    def blade_names(self) -> tuple[str, ...]:
        return tuple(self.blade_name(blade) for blade in range(self.blade_count))

    def blade_from_key(self, key: int | str) -> int:
        if isinstance(key, int):
            return self.validate_blade(key)
        if not isinstance(key, str):
            raise TypeError(f"Unsupported blade key type: {type(key)!r}")

        try:
            return _blade_name_lookup(self.dimension, self.start_index)[key]
        except KeyError as exc:
            raise KeyError(f"Unknown basis blade: {key}") from exc

    def blades_of_grade(self, grade: int) -> tuple[int, ...]:
        if grade < 0 or grade > self.dimension:
            raise ValueError(f"Grade must be between 0 and {self.dimension}.")
        return tuple(blade for blade in range(self.blade_count) if grade_of_blade(blade) == grade)

    def grades_of_blades(self) -> tuple[int, ...]:
        return tuple(grade_of_blade(blade) for blade in range(self.blade_count))

    @property
    def pseudoscalar_blade(self) -> int:
        return self.blade_count - 1

    @property
    def basis_product_table(self) -> BasisProductTable | None:
        return _basis_product_table_cached(self.signature)

    def blade_product(self, lhs: int, rhs: int) -> tuple[int, int]:
        lhs = self.validate_blade(lhs)
        rhs = self.validate_blade(rhs)
        table = self.basis_product_table
        if table is not None:
            return table.blade_product(lhs, rhs)
        return _blade_product_cached(self.signature, lhs, rhs)

    def cayley_table(self) -> dict[tuple[str, str], str]:
        if self.basis_product_table is not None:
            return dict(_cayley_entries_cached(self.signature, self.start_index))
        return dict(_build_cayley_entries(self.signature, self.start_index))

    @classmethod
    def from_pqr(
        cls,
        p: int,
        q: int = 0,
        r: int = 0,
        *,
        start_index: int | None = None,
    ) -> AlgebraSpec:
        if min(p, q, r) < 0:
            raise ValueError("p, q, and r must be non-negative.")

        if start_index is None:
            start_index = 0 if r == 1 else 1

        if r == 1:
            signature = (0,) * r + (1,) * p + (-1,) * q
        else:
            signature = (1,) * p + (-1,) * q + (0,) * r
        return cls(signature=signature, start_index=start_index)


def vga(dimension: int) -> AlgebraSpec:
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    return AlgebraSpec.from_pqr(dimension, 0, 0)


def vga2d() -> AlgebraSpec:
    return vga(2)


def vga3d() -> AlgebraSpec:
    return vga(3)


def pga2d() -> AlgebraSpec:
    return AlgebraSpec(signature=(0, 1, 1), start_index=0)


def pga3d() -> AlgebraSpec:
    return AlgebraSpec(signature=(0, 1, 1, 1), start_index=0)


def cga(dimension: int) -> AlgebraSpec:
    """Return the conformal model for Euclidean ``dimension``-space.

    The basis convention is Euclidean axes followed by two orthogonal conformal
    axes with squares ``+1`` and ``-1``. ``Algebra`` methods and the standalone
    :mod:`amsa.cga` helpers expose the null origin/infinity combinations, so the
    core algebra can remain a plain diagonal Clifford signature.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    return AlgebraSpec(signature=(1,) * dimension + (1, -1), start_index=1)


def cga2d() -> AlgebraSpec:
    return cga(2)


def cga3d() -> AlgebraSpec:
    return cga(3)
