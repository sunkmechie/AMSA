from __future__ import annotations

from numbers import Number

import numpy as np

from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.plans import OpKind, plan_binary_product
from amsa.reference import execute_binary_plan
from amsa.specs import grade_of_blade
from amsa.storage import reweight_storage, scale_storage


def ensure_compatible(lhs: MVArray, rhs: MVArray) -> None:
    """Validate that two multivectors share algebra and layout metadata."""
    if lhs.algebra != rhs.algebra:
        raise ValueError("Multivectors belong to different algebras.")


def _coerce_operand(reference: MVArray, operand: MVArray | Number) -> MVArray:
    if isinstance(operand, MVArray):
        ensure_compatible(reference, operand)
        return operand
    if isinstance(operand, Number):
        operand_array = np.asarray(operand)
        scalar_layout = MVLayout.grade(reference.algebra, 0)
        dtype = np.result_type(reference.dtype, operand_array.dtype)
        values = np.asarray([operand], dtype=dtype)
        return MVArray(algebra=reference.algebra, layout=scalar_layout, values=values)
    raise TypeError(f"Unsupported operand type: {type(operand)!r}")


def neg(mv: MVArray) -> MVArray:
    return MVArray(algebra=mv.algebra, layout=mv.layout, storage=scale_storage(mv.storage, -1))


def _union_layout(lhs: MVArray, rhs: MVArray | Number) -> tuple[MVArray, MVLayout]:
    rhs_mv = _coerce_operand(lhs, rhs)
    if lhs.layout == rhs_mv.layout:
        return rhs_mv, lhs.layout

    blades = tuple(sorted(set(lhs.layout.blades) | set(rhs_mv.layout.blades)))
    if len(blades) == lhs.algebra.blade_count:
        return rhs_mv, MVLayout.dense(lhs.algebra)
    return rhs_mv, MVLayout.sparse_pattern(lhs.algebra, blades, name="union")


def add(lhs: MVArray, rhs: MVArray | Number) -> MVArray:
    rhs_mv, layout = _union_layout(lhs, rhs)
    lhs_projected = lhs.to_layout(layout)
    rhs_projected = rhs_mv.to_layout(layout)
    values = lhs_projected.values + rhs_projected.values
    return MVArray(algebra=lhs.algebra, layout=layout, values=values)


def sub(lhs: MVArray, rhs: MVArray | Number) -> MVArray:
    rhs_mv, layout = _union_layout(lhs, rhs)
    lhs_projected = lhs.to_layout(layout)
    rhs_projected = rhs_mv.to_layout(layout)
    values = lhs_projected.values - rhs_projected.values
    return MVArray(algebra=lhs.algebra, layout=layout, values=values)


def reverse(mv: MVArray) -> MVArray:
    signs = np.asarray(
        [
            (-1) ** ((blade.bit_count() * (blade.bit_count() - 1)) // 2)
            for blade in mv.layout.blades
        ],
        dtype=mv.dtype,
    )
    return MVArray(
        algebra=mv.algebra,
        layout=mv.layout,
        storage=reweight_storage(mv.storage, signs),
    )


def involute(mv: MVArray) -> MVArray:
    signs = np.asarray([(-1) ** blade.bit_count() for blade in mv.layout.blades], dtype=mv.dtype)
    return MVArray(
        algebra=mv.algebra,
        layout=mv.layout,
        storage=reweight_storage(mv.storage, signs),
    )


def conjugate(mv: MVArray) -> MVArray:
    return reverse(involute(mv))


def _execute_binary_product(lhs: MVArray, rhs: MVArray, kind: OpKind) -> MVArray:
    ensure_compatible(lhs, rhs)
    plan = plan_binary_product(lhs.layout, rhs.layout, kind)
    return execute_binary_plan(lhs, rhs, plan)


def geometric_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "geometric")


def outer_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "outer")


def inner_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "inner")


def project_grades(mv: MVArray, *grades: int) -> MVArray:
    normalized = grades[0] if len(grades) == 1 and isinstance(grades[0], tuple) else grades
    if not normalized:
        raise ValueError("At least one grade must be selected.")

    grade_set = set(normalized)
    for grade in grade_set:
        if grade < 0 or grade > mv.algebra.dimension:
            raise ValueError(f"Grade must be between 0 and {mv.algebra.dimension}.")

    blades = tuple(blade for blade in mv.layout.blades if grade_of_blade(blade) in grade_set)
    if blades == tuple(range(mv.algebra.blade_count)):
        layout = MVLayout.dense(mv.algebra)
    else:
        name = "grade[" + ",".join(str(grade) for grade in sorted(grade_set)) + "]"
        layout = MVLayout.sparse_pattern(mv.algebra, blades, name=name)
    return mv.to_layout(layout)
