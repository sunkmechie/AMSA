from __future__ import annotations

from dataclasses import dataclass
from functools import cache


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

        for blade in range(self.blade_count):
            if self.blade_name(blade) == key:
                return blade
        raise KeyError(f"Unknown basis blade: {key}")

    def blades_of_grade(self, grade: int) -> tuple[int, ...]:
        if grade < 0 or grade > self.dimension:
            raise ValueError(f"Grade must be between 0 and {self.dimension}.")
        return tuple(blade for blade in range(self.blade_count) if grade_of_blade(blade) == grade)

    def grades_of_blades(self) -> tuple[int, ...]:
        return tuple(grade_of_blade(blade) for blade in range(self.blade_count))

    @property
    def pseudoscalar_blade(self) -> int:
        return self.blade_count - 1

    @cache
    def blade_product(self, lhs: int, rhs: int) -> tuple[int, int]:
        lhs = self.validate_blade(lhs)
        rhs = self.validate_blade(rhs)

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
            metric = self.signature[axis]
            if metric == 0:
                return 0, 0
            coefficient *= metric
            overlap ^= bit

        return coefficient, lhs ^ rhs

    @classmethod
    def from_pqr(cls, p: int, q: int = 0, r: int = 0, *, start_index: int | None = None) -> AlgebraSpec:
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


def pga2d() -> AlgebraSpec:
    return AlgebraSpec(signature=(0, 1, 1), start_index=0)


def pga3d() -> AlgebraSpec:
    return AlgebraSpec(signature=(0, 1, 1, 1), start_index=0)
