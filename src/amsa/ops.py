from __future__ import annotations

from numbers import Number

import numpy as np

from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.plans import OpKind, plan_binary_product
from amsa.reference import execute_binary_plan
from amsa.specs import grade_of_blade
from amsa.storage import project_storage, reweight_storage, row_scale_storage, scale_storage


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


def _pseudoscalar_inverse_scale(mv: MVArray) -> float:
    pseudoscalar = mv.algebra.pseudoscalar_blade
    coefficient, _ = mv.algebra.blade_product(pseudoscalar, pseudoscalar)
    if coefficient == 0:
        raise ValueError(
            "dual/undual require an invertible pseudoscalar; this algebra is degenerate."
        )
    return 1.0 / float(coefficient)


def _complement_layout(blades: tuple[int, ...], *, pseudoscalar: int) -> tuple[int, ...]:
    return tuple(sorted(blade ^ pseudoscalar for blade in blades))


def _pseudoscalar_transform(
    mv: MVArray,
    *,
    inverse: bool,
) -> MVArray:
    pseudoscalar = mv.algebra.pseudoscalar_blade
    inverse_scale = _pseudoscalar_inverse_scale(mv) if inverse else 1.0

    target_blades = _complement_layout(mv.layout.blades, pseudoscalar=pseudoscalar)
    if len(target_blades) == mv.algebra.blade_count:
        layout = MVLayout.dense(mv.algebra)
    else:
        name = "dual" if inverse else "undual"
        layout = MVLayout.sparse_pattern(mv.algebra, target_blades, name=name)

    source_index = {blade: index for index, blade in enumerate(mv.layout.blades)}
    projection_columns: list[int] = []
    weights: list[float] = []
    for target_blade in layout.blades:
        source_blade = target_blade ^ pseudoscalar
        source_column = source_index[source_blade]
        coefficient, _ = mv.algebra.blade_product(source_blade, pseudoscalar)
        projection_columns.append(source_column)
        weights.append(inverse_scale * coefficient)

    projected = project_storage(mv.storage, tuple(projection_columns))
    transformed = reweight_storage(projected, np.asarray(weights, dtype=mv.dtype))
    return MVArray(
        algebra=mv.algebra,
        layout=layout,
        storage=transformed,
    )


def dual(mv: MVArray) -> MVArray:
    return _pseudoscalar_transform(mv, inverse=True)


def undual(mv: MVArray) -> MVArray:
    return _pseudoscalar_transform(mv, inverse=False)


def _poincare_transform(
    mv: MVArray,
    *,
    inverse: bool,
) -> MVArray:
    pseudoscalar = mv.algebra.pseudoscalar_blade
    target_blades = _complement_layout(mv.layout.blades, pseudoscalar=pseudoscalar)
    if len(target_blades) == mv.algebra.blade_count:
        layout = MVLayout.dense(mv.algebra)
    else:
        name = "poincare_undual" if inverse else "poincare_dual"
        layout = MVLayout.sparse_pattern(mv.algebra, target_blades, name=name)

    source_index = {blade: index for index, blade in enumerate(mv.layout.blades)}
    projection_columns: list[int] = []
    weights: list[int] = []
    for target_blade in layout.blades:
        source_blade = target_blade ^ pseudoscalar
        source_column = source_index[source_blade]
        lhs_blade, rhs_blade = (
            (target_blade, source_blade) if inverse else (source_blade, target_blade)
        )
        coefficient, _ = mv.algebra.blade_product(lhs_blade, rhs_blade)
        projection_columns.append(source_column)
        weights.append(coefficient)

    projected = project_storage(mv.storage, tuple(projection_columns))
    transformed = reweight_storage(projected, np.asarray(weights, dtype=mv.dtype))
    return MVArray(
        algebra=mv.algebra,
        layout=layout,
        storage=transformed,
    )


def poincare_dual(mv: MVArray) -> MVArray:
    return _poincare_transform(mv, inverse=False)


def poincare_undual(mv: MVArray) -> MVArray:
    return _poincare_transform(mv, inverse=True)


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


def scalar_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "scalar")


def left_contraction(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "left_contraction")


def right_contraction(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "right_contraction")


def regressive_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "regressive")


def _require_scalar_output(mv: MVArray, *, name: str) -> np.ndarray:
    scalar_blade = 0
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    scalar_value = np.asarray(mv.component(scalar_blade), dtype=resolved_dtype)

    if mv.layout.size == 0:
        raise ValueError(f"{name} is zero and therefore non-invertible.")

    for index, blade in enumerate(mv.layout.blades):
        if blade == scalar_blade:
            continue
        component = np.asarray(mv.values[..., index], dtype=resolved_dtype)
        if np.any(~np.isclose(component, 0.0)):
            raise ValueError(f"{name} must be scalar-valued for this operation.")

    return scalar_value


def inverse(mv: MVArray) -> MVArray:
    reversed_mv = reverse(mv)
    left_norm_mv = geometric_product(reversed_mv, mv)
    right_norm_mv = geometric_product(mv, reversed_mv)
    left_norm = _require_scalar_output(left_norm_mv, name="reverse(mv) * mv")
    right_norm = _require_scalar_output(right_norm_mv, name="mv * reverse(mv)")

    if not np.allclose(left_norm, right_norm):
        raise ValueError("inverse() currently requires matching scalar left/right reverse norms.")
    if np.any(np.isclose(left_norm, 0.0)):
        raise ValueError("inverse() is undefined for zero-norm or non-invertible multivectors.")

    reciprocals = np.reciprocal(left_norm)
    return MVArray(
        algebra=mv.algebra,
        layout=reversed_mv.layout,
        storage=row_scale_storage(reversed_mv.storage, reciprocals),
    )


def divide(lhs: MVArray | Number, rhs: MVArray | Number) -> MVArray:
    if isinstance(lhs, MVArray):
        if isinstance(rhs, MVArray):
            return geometric_product(lhs, inverse(rhs))
        if isinstance(rhs, Number):
            rhs_array = np.asarray(rhs)
            if bool(np.equal(rhs_array, 0).item()):
                raise ZeroDivisionError("Division by zero scalar.")
            return MVArray(
                algebra=lhs.algebra,
                layout=lhs.layout,
                storage=scale_storage(lhs.storage, np.reciprocal(rhs_array)),
            )
        raise TypeError(f"Unsupported operand type: {type(rhs)!r}")

    if isinstance(lhs, Number) and isinstance(rhs, MVArray):
        scalar_lhs = _coerce_operand(rhs, lhs)
        return geometric_product(scalar_lhs, inverse(rhs))

    raise TypeError(f"Unsupported operand types: {type(lhs)!r} and {type(rhs)!r}")


def project_grades(mv: MVArray, *grades: int) -> MVArray:
    if not grades:
        raise ValueError("At least one grade must be selected.")

    grade_set = set(grades)
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
