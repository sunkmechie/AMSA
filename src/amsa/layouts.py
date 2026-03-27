from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from amsa.specs import AlgebraSpec, grade_of_blade


LayoutKind = Literal["dense", "grade", "sparse"]


@dataclass(frozen=True, slots=True)
class MVLayout:
    """Descriptor for how multivector coefficients are stored."""

    algebra: AlgebraSpec
    blades: tuple[int, ...]
    kind: LayoutKind
    name: str = "custom"

    def __post_init__(self) -> None:
        max_blade = self.algebra.blade_count
        if len(set(self.blades)) != len(self.blades):
            raise ValueError("Layout blades must be unique.")
        if any(blade < 0 or blade >= max_blade for blade in self.blades):
            raise ValueError("Layout blades must belong to the algebra.")
        if self.kind not in ("dense", "grade", "sparse"):
            raise ValueError("Unsupported layout kind.")
        if not self.blades and self.kind != "sparse":
            raise ValueError("Only sparse layouts may be empty.")

    @property
    def size(self) -> int:
        return len(self.blades)

    @property
    def grades(self) -> tuple[int, ...]:
        return tuple(sorted({grade_of_blade(blade) for blade in self.blades}))

    def blade_names(self) -> tuple[str, ...]:
        return tuple(self.algebra.blade_name(blade) for blade in self.blades)

    def index_of(self, blade: int) -> int:
        return self.blades.index(blade)

    def contains(self, blade: int) -> bool:
        return blade in self.blades

    @classmethod
    def dense(cls, algebra: AlgebraSpec) -> MVLayout:
        return cls(
            algebra=algebra,
            blades=tuple(range(algebra.blade_count)),
            kind="dense",
            name="dense",
        )

    @classmethod
    def grade(cls, algebra: AlgebraSpec, *grades: int) -> MVLayout:
        if not grades:
            raise ValueError("At least one grade must be selected.")

        blades: list[int] = []
        for grade in sorted(set(grades)):
            blades.extend(algebra.blades_of_grade(grade))
        return cls(
            algebra=algebra,
            blades=tuple(blades),
            kind="grade",
            name="grade[" + ",".join(str(grade) for grade in sorted(set(grades))) + "]",
        )

    @classmethod
    def sparse_pattern(
        cls,
        algebra: AlgebraSpec,
        blades: tuple[int, ...],
        *,
        name: str = "sparse",
    ) -> MVLayout:
        return cls(algebra=algebra, blades=blades, kind="sparse", name=name)
