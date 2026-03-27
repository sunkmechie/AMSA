from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

from amsa.layouts import MVLayout
from amsa.specs import AlgebraSpec, grade_of_blade


OpKind = Literal["geometric", "outer", "inner"]

_LAYOUT_NAMES: dict[OpKind, str] = {
    "geometric": "gp",
    "outer": "op",
    "inner": "ip",
}


@dataclass(frozen=True, slots=True)
class ProductTerm:
    lhs_index: int
    rhs_index: int
    out_blade: int
    coefficient: int


@dataclass(frozen=True, slots=True)
class OpPlan:
    kind: OpKind
    algebra: AlgebraSpec
    lhs_blades: tuple[int, ...]
    rhs_blades: tuple[int, ...]
    output_blades: tuple[int, ...]
    terms: tuple[ProductTerm, ...]

    def output_layout(self) -> MVLayout:
        if len(self.output_blades) == self.algebra.blade_count:
            return MVLayout.dense(self.algebra)
        return MVLayout.sparse_pattern(
            self.algebra,
            self.output_blades,
            name=_LAYOUT_NAMES[self.kind],
        )


def _include_term(kind: OpKind, lhs_blade: int, rhs_blade: int, out_blade: int) -> bool:
    if kind == "geometric":
        return True

    lhs_grade = grade_of_blade(lhs_blade)
    rhs_grade = grade_of_blade(rhs_blade)
    out_grade = grade_of_blade(out_blade)

    if kind == "outer":
        return out_grade == lhs_grade + rhs_grade
    if kind == "inner":
        return out_grade == abs(lhs_grade - rhs_grade)
    raise ValueError(f"Unsupported operator kind: {kind}")


@cache
def build_op_plan(
    algebra: AlgebraSpec,
    lhs_blades: tuple[int, ...],
    rhs_blades: tuple[int, ...],
    kind: OpKind,
) -> OpPlan:
    support: set[int] = set()
    terms: list[ProductTerm] = []

    for lhs_index, lhs_blade in enumerate(lhs_blades):
        for rhs_index, rhs_blade in enumerate(rhs_blades):
            coefficient, out_blade = algebra.blade_product(lhs_blade, rhs_blade)
            if coefficient == 0:
                continue
            if not _include_term(kind, lhs_blade, rhs_blade, out_blade):
                continue
            support.add(out_blade)
            terms.append(
                ProductTerm(
                    lhs_index=lhs_index,
                    rhs_index=rhs_index,
                    out_blade=out_blade,
                    coefficient=coefficient,
                )
            )

    return OpPlan(
        kind=kind,
        algebra=algebra,
        lhs_blades=lhs_blades,
        rhs_blades=rhs_blades,
        output_blades=tuple(sorted(support)),
        terms=tuple(terms),
    )


def plan_binary_product(lhs: MVLayout, rhs: MVLayout, kind: OpKind) -> OpPlan:
    if lhs.algebra != rhs.algebra:
        raise ValueError("Layouts belong to different algebras.")
    return build_op_plan(lhs.algebra, lhs.blades, rhs.blades, kind)
