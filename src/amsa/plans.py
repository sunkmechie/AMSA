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
from functools import cache
from typing import Literal

from amsa.layouts import MVLayout
from amsa.specs import AlgebraSpec, grade_of_blade

OpKind = Literal[
    "geometric",
    "outer",
    "inner",
    "scalar",
    "left_contraction",
    "right_contraction",
    "regressive",
]

_LAYOUT_NAMES: dict[OpKind, str] = {
    "geometric": "gp",
    "outer": "op",
    "inner": "ip",
    "scalar": "sp",
    "left_contraction": "lc",
    "right_contraction": "rc",
    "regressive": "rp",
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

    def show(self) -> str:
        """Display product plan in human-readable algebra notation."""
        lines = [f"OpPlan({self.kind})"]
        lhs_names = ", ".join(self.algebra.blade_name(b) for b in self.lhs_blades)
        lines.append(f"  LHS blades: {lhs_names}")
        rhs_names = ", ".join(self.algebra.blade_name(b) for b in self.rhs_blades)
        lines.append(f"  RHS blades: {rhs_names}")
        out_names = ", ".join(self.algebra.blade_name(b) for b in self.output_blades)
        lines.append(f"  Output blades: {out_names}")
        lines.append(f"  Terms ({len(self.terms)}):")
        for term in self.terms:
            lhs_name = self.algebra.blade_name(self.lhs_blades[term.lhs_index])
            rhs_name = self.algebra.blade_name(self.rhs_blades[term.rhs_index])
            out_name = self.algebra.blade_name(term.out_blade)
            coeff_str = f"{term.coefficient:+d}" if term.coefficient != 1 else "+"
            lines.append(f"    {coeff_str} {lhs_name} * {rhs_name} -> {out_name}")
        return "\n".join(lines)


def _include_term(kind: OpKind, lhs_blade: int, rhs_blade: int, out_blade: int) -> bool:
    if kind == "geometric":
        return True
    return _include_term_grades(
        kind,
        grade_of_blade(lhs_blade),
        grade_of_blade(rhs_blade),
        grade_of_blade(out_blade),
    )


def _include_term_grades(
    kind: OpKind,
    lhs_grade: int,
    rhs_grade: int,
    out_grade: int,
) -> bool:
    if kind == "outer":
        return out_grade == lhs_grade + rhs_grade
    if kind == "inner":
        return out_grade == abs(lhs_grade - rhs_grade)
    if kind == "scalar":
        return out_grade == 0
    if kind == "left_contraction":
        return lhs_grade <= rhs_grade and out_grade == rhs_grade - lhs_grade
    if kind == "right_contraction":
        return lhs_grade >= rhs_grade and out_grade == lhs_grade - rhs_grade
    raise ValueError(f"Unsupported operator kind: {kind}")


def _build_regressive_plan(
    algebra: AlgebraSpec,
    lhs_blades: tuple[int, ...],
    rhs_blades: tuple[int, ...],
) -> OpPlan:
    pseudoscalar = algebra.pseudoscalar_blade
    table = algebra.basis_product_table
    support: set[int] = set()
    terms: list[ProductTerm] = []

    for lhs_index, lhs_blade in enumerate(lhs_blades):
        lhs_dual_blade = lhs_blade ^ pseudoscalar
        if table is not None:
            lhs_dual_coefficient = int(table.coefficients[lhs_blade, lhs_dual_blade])
        else:
            lhs_dual_coefficient, _ = algebra.blade_product(lhs_blade, lhs_dual_blade)

        for rhs_index, rhs_blade in enumerate(rhs_blades):
            rhs_dual_blade = rhs_blade ^ pseudoscalar
            if table is not None:
                rhs_dual_coefficient = int(table.coefficients[rhs_blade, rhs_dual_blade])
                dual_outer_coefficient = int(table.coefficients[lhs_dual_blade, rhs_dual_blade])
                dual_output_blade = int(table.output_blades[lhs_dual_blade, rhs_dual_blade])
            else:
                rhs_dual_coefficient, _ = algebra.blade_product(rhs_blade, rhs_dual_blade)
                dual_outer_coefficient, dual_output_blade = algebra.blade_product(
                    lhs_dual_blade,
                    rhs_dual_blade,
                )
            if dual_outer_coefficient == 0:
                continue
            if table is not None:
                dual_output_grade = int(table.grades[dual_output_blade])
                lhs_dual_grade = int(table.grades[lhs_dual_blade])
                rhs_dual_grade = int(table.grades[rhs_dual_blade])
            else:
                dual_output_grade = grade_of_blade(dual_output_blade)
                lhs_dual_grade = grade_of_blade(lhs_dual_blade)
                rhs_dual_grade = grade_of_blade(rhs_dual_blade)
            if dual_output_grade != (lhs_dual_grade + rhs_dual_grade):
                continue

            out_blade = dual_output_blade ^ pseudoscalar
            if table is not None:
                undual_coefficient = int(table.coefficients[out_blade, dual_output_blade])
            else:
                undual_coefficient, _ = algebra.blade_product(out_blade, dual_output_blade)
            coefficient = (
                lhs_dual_coefficient
                * rhs_dual_coefficient
                * dual_outer_coefficient
                * undual_coefficient
            )
            if coefficient == 0:
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
        kind="regressive",
        algebra=algebra,
        lhs_blades=lhs_blades,
        rhs_blades=rhs_blades,
        output_blades=tuple(sorted(support)),
        terms=tuple(terms),
    )


@cache
def build_op_plan(
    algebra: AlgebraSpec,
    lhs_blades: tuple[int, ...],
    rhs_blades: tuple[int, ...],
    kind: OpKind,
) -> OpPlan:
    if kind == "regressive":
        return _build_regressive_plan(algebra, lhs_blades, rhs_blades)

    table = algebra.basis_product_table
    support: set[int] = set()
    terms: list[ProductTerm] = []

    if table is not None:
        for lhs_index, lhs_blade in enumerate(lhs_blades):
            lhs_grade = int(table.grades[lhs_blade])
            for rhs_index, rhs_blade in enumerate(rhs_blades):
                coefficient = int(table.coefficients[lhs_blade, rhs_blade])
                if coefficient == 0:
                    continue

                out_blade = int(table.output_blades[lhs_blade, rhs_blade])
                if kind != "geometric":
                    rhs_grade = int(table.grades[rhs_blade])
                    out_grade = int(table.grades[out_blade])
                    if not _include_term_grades(kind, lhs_grade, rhs_grade, out_grade):
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
    else:
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
