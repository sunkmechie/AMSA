from __future__ import annotations

from dataclasses import dataclass

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

    def zeros(self, layout: MVLayout | None = None) -> MVArray:
        active_layout = layout if layout is not None else self.dense_layout()
        return MVArray.zeros(self.spec, active_layout)
