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


def commutator_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    ensure_compatible(lhs, rhs)
    result = geometric_product(lhs, rhs) - geometric_product(rhs, lhs)
    return MVArray(
        algebra=result.algebra,
        layout=result.layout,
        storage=scale_storage(result.storage, 0.5),
    )


def anticommutator_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    ensure_compatible(lhs, rhs)
    result = geometric_product(lhs, rhs) + geometric_product(rhs, lhs)
    return MVArray(
        algebra=result.algebra,
        layout=result.layout,
        storage=scale_storage(result.storage, 0.5),
    )


def _require_degenerate_algebra(mv: MVArray, *, name: str) -> int:
    null_mask = 0
    for axis, metric in enumerate(mv.algebra.signature):
        if metric == 0:
            null_mask |= 1 << axis
    if null_mask == 0:
        raise ValueError(f"{name} requires an algebra with at least one null basis vector.")
    return null_mask


def _select_null_factor(mv: MVArray, *, include_null_factor: bool, name: str) -> MVArray:
    null_mask = _require_degenerate_algebra(mv, name=name)
    blades = tuple(
        blade for blade in mv.layout.blades if bool(blade & null_mask) == include_null_factor
    )
    layout_name = name
    if len(blades) == mv.algebra.blade_count:
        layout = MVLayout.dense(mv.algebra)
    else:
        layout = MVLayout.sparse_pattern(mv.algebra, blades, name=layout_name)
    return mv.to_layout(layout)


def bulk(mv: MVArray) -> MVArray:
    return _select_null_factor(mv, include_null_factor=False, name="bulk")


def weight(mv: MVArray) -> MVArray:
    return _select_null_factor(mv, include_null_factor=True, name="weight")


def bulk_dual(mv: MVArray) -> MVArray:
    return poincare_dual(bulk(mv))


def weight_dual(mv: MVArray) -> MVArray:
    return poincare_dual(weight(mv))


def norm_squared(mv: MVArray) -> MVArray:
    return scalar_product(mv, reverse(mv))


def _scalar_mv(mv: MVArray, values: np.ndarray) -> MVArray:
    dtype = np.result_type(mv.dtype, values.dtype)
    scalar_layout = MVLayout.grade(mv.algebra, 0)
    payload = np.asarray(values, dtype=dtype)
    if payload.shape == ():
        payload = np.asarray([payload.item()], dtype=dtype)
    else:
        payload = payload[..., np.newaxis]
    return MVArray(algebra=mv.algebra, layout=scalar_layout, values=payload)


def _single_blade_mv(mv: MVArray, blade: int, values: np.ndarray) -> MVArray:
    dtype = np.result_type(mv.dtype, values.dtype)
    layout = MVLayout.sparse_pattern(mv.algebra, (blade,), name=mv.algebra.blade_name(blade))
    payload = np.asarray(values, dtype=dtype)
    if payload.shape == ():
        payload = np.asarray([payload.item()], dtype=dtype)
    else:
        payload = payload[..., np.newaxis]
    return MVArray(algebra=mv.algebra, layout=layout, values=payload)


def _row_scale_mv(mv: MVArray, scales: np.ndarray) -> MVArray:
    return MVArray(
        algebra=mv.algebra,
        layout=mv.layout,
        storage=row_scale_storage(mv.storage, scales),
    )


def _require_study_output(mv: MVArray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    pseudoscalar_blade = mv.algebra.pseudoscalar_blade
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    scalar_value = np.asarray(mv.component(0), dtype=resolved_dtype)
    pseudoscalar_value = np.asarray(mv.component(pseudoscalar_blade), dtype=resolved_dtype)

    if mv.layout.size == 0:
        zeros = np.zeros(mv.batch_shape, dtype=resolved_dtype)
        return zeros, zeros

    for index, blade in enumerate(mv.layout.blades):
        if blade in (0, pseudoscalar_blade):
            continue
        component = np.asarray(mv.values[..., index], dtype=resolved_dtype)
        if np.any(~np.isclose(component, 0.0)):
            raise ValueError(f"{name} must be scalar + pseudoscalar valued for this operation.")

    return scalar_value, pseudoscalar_value


def _study_value_mv(
    mv: MVArray,
    scalar_values: np.ndarray,
    pseudoscalar_values: np.ndarray,
) -> MVArray:
    result = _scalar_mv(mv, scalar_values)
    if np.any(~np.isclose(pseudoscalar_values, 0.0)):
        result = result + _single_blade_mv(mv, mv.algebra.pseudoscalar_blade, pseudoscalar_values)
    return result


def _study_times_mv(
    mv: MVArray,
    scalar_values: np.ndarray,
    pseudoscalar_values: np.ndarray,
) -> MVArray:
    result = _row_scale_mv(mv, scalar_values)
    if np.any(~np.isclose(pseudoscalar_values, 0.0)):
        pseudoscalar = _single_blade_mv(mv, mv.algebra.pseudoscalar_blade, pseudoscalar_values)
        result = result + geometric_product(pseudoscalar, mv)
    return result


def norm(mv: MVArray) -> MVArray:
    normsq = norm_squared(mv)
    normsq_values = _require_scalar_output(normsq, name="norm_squared(mv)")
    magnitudes = np.sqrt(np.abs(normsq_values))
    return _scalar_mv(mv, magnitudes)


def normalize(mv: MVArray) -> MVArray:
    magnitudes = _require_scalar_output(norm(mv), name="norm(mv)")
    if np.any(np.isclose(magnitudes, 0.0)):
        raise ValueError("normalize() is undefined for zero-magnitude multivectors.")
    reciprocals = np.reciprocal(magnitudes)
    return _row_scale_mv(mv, reciprocals)


def _motor_exp_from_bivector(mv: MVArray) -> MVArray:
    if mv.algebra.signature != (0, 1, 1, 1):
        raise ValueError("motor_exp() currently supports PGA3d bivector generators.")
    if set(mv.grades) != {2}:
        raise ValueError("motor_exp() currently requires a pure bivector generator.")

    square = geometric_product(mv, mv)
    scalar_part, pseudoscalar_part = _require_study_output(square, name="mv * mv")
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    scalar_part = np.asarray(scalar_part, dtype=resolved_dtype)
    pseudoscalar_part = np.asarray(pseudoscalar_part, dtype=resolved_dtype)

    scalar_coeff = np.zeros(mv.batch_shape, dtype=resolved_dtype)
    pseudo_coeff = np.zeros(mv.batch_shape, dtype=resolved_dtype)
    linear_coeff = np.zeros(mv.batch_shape, dtype=resolved_dtype)
    dual_linear_coeff = np.zeros(mv.batch_shape, dtype=resolved_dtype)

    zero_mask = np.isclose(scalar_part, 0.0)
    circular_mask = scalar_part < 0.0
    hyperbolic_mask = scalar_part > 0.0

    if np.any(zero_mask):
        scalar_coeff[zero_mask] = 1.0
        linear_coeff[zero_mask] = 1.0
        pseudo_coeff[zero_mask] = 0.5 * pseudoscalar_part[zero_mask]
        dual_linear_coeff[zero_mask] = pseudoscalar_part[zero_mask] / 6.0

    if np.any(circular_mask):
        roots = np.sqrt(-scalar_part[circular_mask])
        delta = -pseudoscalar_part[circular_mask] / (2.0 * roots)
        sinc = np.sin(roots) / roots
        dsinc = (roots * np.cos(roots) - np.sin(roots)) / (roots * roots)

        scalar_coeff[circular_mask] = np.cos(roots)
        pseudo_coeff[circular_mask] = -delta * np.sin(roots)
        linear_coeff[circular_mask] = sinc
        dual_linear_coeff[circular_mask] = delta * dsinc

    if np.any(hyperbolic_mask):
        roots = np.sqrt(scalar_part[hyperbolic_mask])
        delta = pseudoscalar_part[hyperbolic_mask] / (2.0 * roots)
        sinhc = np.sinh(roots) / roots
        dsinhc = (roots * np.cosh(roots) - np.sinh(roots)) / (roots * roots)

        scalar_coeff[hyperbolic_mask] = np.cosh(roots)
        pseudo_coeff[hyperbolic_mask] = delta * np.sinh(roots)
        linear_coeff[hyperbolic_mask] = sinhc
        dual_linear_coeff[hyperbolic_mask] = delta * dsinhc

    return _study_value_mv(mv, scalar_coeff, pseudo_coeff) + _study_times_mv(
        mv,
        linear_coeff,
        dual_linear_coeff,
    )


def exp(mv: MVArray) -> MVArray:
    square = geometric_product(mv, mv)
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    if square.layout.size == 0:
        scalar_values = np.zeros(mv.batch_shape, dtype=resolved_dtype)
    else:
        try:
            square_values = _require_scalar_output(square, name="mv * mv")
        except ValueError as exc:
            if set(mv.grades) == {2} and mv.algebra.signature == (0, 1, 1, 1):
                return _motor_exp_from_bivector(mv)
            raise exc
        scalar_values = np.asarray(square_values, dtype=resolved_dtype)

    positive_mask = scalar_values > 0.0
    negative_mask = scalar_values < 0.0
    zero_mask = np.isclose(scalar_values, 0.0)

    roots = np.sqrt(np.abs(scalar_values))
    scalar_coefficients = np.empty_like(roots, dtype=resolved_dtype)
    linear_coefficients = np.empty_like(roots, dtype=resolved_dtype)

    scalar_coefficients[positive_mask] = np.cosh(roots[positive_mask])
    linear_coefficients[positive_mask] = np.sinh(roots[positive_mask]) / roots[positive_mask]

    scalar_coefficients[negative_mask] = np.cos(roots[negative_mask])
    linear_coefficients[negative_mask] = np.sin(roots[negative_mask]) / roots[negative_mask]

    scalar_coefficients[zero_mask] = 1.0
    linear_coefficients[zero_mask] = 1.0

    return _scalar_mv(mv, scalar_coefficients) + _row_scale_mv(mv, linear_coefficients)


def motor_exp(mv: MVArray) -> MVArray:
    return _motor_exp_from_bivector(mv)


def _simple_bivector_log(mv: MVArray) -> MVArray:
    scalar_values = _require_scalar_output(project_grades(mv, 0), name="grade_0(mv)")
    bivector = project_grades(mv, 2)
    if bivector.layout.size == 0 or np.allclose(bivector.values, 0.0):
        if np.any(scalar_values < 0.0):
            raise ValueError(
                "motor_log() is undefined on the principal branch for negative scalars."
            )
        return MVArray.zeros(mv.algebra, MVLayout.grade(mv.algebra, 2), batch_shape=mv.batch_shape)

    square = geometric_product(bivector, bivector)
    square_values = _require_scalar_output(square, name="grade_2(mv) * grade_2(mv)")
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    scalar_values = np.asarray(scalar_values, dtype=resolved_dtype)
    square_values = np.asarray(square_values, dtype=resolved_dtype)
    roots = np.sqrt(np.abs(square_values))
    coefficients = np.zeros_like(roots, dtype=resolved_dtype)

    circular_mask = square_values < 0.0
    hyperbolic_mask = square_values > 0.0
    null_mask = np.isclose(square_values, 0.0)

    if np.any(circular_mask):
        coefficients[circular_mask] = (
            np.arctan2(roots[circular_mask], scalar_values[circular_mask]) / roots[circular_mask]
        )
    if np.any(hyperbolic_mask):
        coefficients[hyperbolic_mask] = (
            np.arctanh(roots[hyperbolic_mask] / scalar_values[hyperbolic_mask])
            / roots[hyperbolic_mask]
        )
    if np.any(null_mask):
        coefficients[null_mask] = np.reciprocal(scalar_values[null_mask])

    return _row_scale_mv(bivector, coefficients)


def _motor_log_pga3d(mv: MVArray) -> MVArray:
    motor = rigid_body_normalize(mv)
    if not set(motor.grades).issubset({0, 2, 4}):
        raise ValueError(
            "motor_log() currently requires a PGA3d motor-like multivector with grades 0, 2, and 4."
        )

    scalar_values = np.asarray(motor.component(0), dtype=np.result_type(motor.dtype, np.float64))
    pseudoscalar_blade = motor.algebra.pseudoscalar_blade
    pseudoscalar_values = np.asarray(
        motor.component(pseudoscalar_blade),
        dtype=np.result_type(motor.dtype, np.float64),
    )
    bivector = project_grades(motor, 2)
    moment_part = bulk(bivector)
    sine_values = _require_scalar_output(bulk_norm(moment_part), name="bulk_norm(grade_2(mv))")
    sine_values = np.asarray(sine_values, dtype=np.result_type(motor.dtype, np.float64))

    if bivector.layout.size == 0 or np.allclose(bivector.values, 0.0):
        return MVArray.zeros(
            motor.algebra,
            MVLayout.grade(motor.algebra, 2),
            batch_shape=motor.batch_shape,
        )

    zero_mask = np.isclose(sine_values, 0.0)
    nonzero_mask = ~zero_mask

    if np.any(zero_mask & ~np.isclose(scalar_values, 1.0)):
        raise ValueError("motor_log() does not support the pi-rotation singular branch yet.")

    if np.all(zero_mask):
        return bivector

    phi_values = np.zeros_like(sine_values)
    phi_values[nonzero_mask] = np.arctan2(sine_values[nonzero_mask], scalar_values[nonzero_mask])

    distance_values = np.zeros_like(sine_values)
    distance_values[nonzero_mask] = -pseudoscalar_values[nonzero_mask] / sine_values[nonzero_mask]

    alpha_values = np.zeros_like(sine_values)
    beta_values = np.zeros_like(sine_values)
    alpha_values[nonzero_mask] = phi_values[nonzero_mask] / sine_values[nonzero_mask]
    beta_values[nonzero_mask] = (
        distance_values[nonzero_mask]
        * (
            1.0
            - (
                phi_values[nonzero_mask]
                * scalar_values[nonzero_mask]
                / sine_values[nonzero_mask]
            )
        )
        / sine_values[nonzero_mask]
    )

    pseudoscalar_values = np.ones(
        motor.batch_shape if motor.batch_shape else (),
        dtype=np.result_type(motor.dtype, np.float64),
    )
    pseudoscalar = _single_blade_mv(
        motor,
        pseudoscalar_blade,
        pseudoscalar_values,
    )
    pseudoscalar_times_bivector = project_grades(geometric_product(pseudoscalar, bivector), 2)

    return _row_scale_mv(bivector, alpha_values) + _row_scale_mv(
        pseudoscalar_times_bivector,
        beta_values,
    )


def motor_log(mv: MVArray) -> MVArray:
    if mv.algebra.signature == (0, 1, 1):
        return _simple_bivector_log(rigid_body_normalize(mv))
    if mv.algebra.signature == (0, 1, 1, 1):
        return _motor_log_pga3d(mv)
    raise ValueError("motor_log() currently supports PGA2d and PGA3d motor-like multivectors.")


def bulk_norm_squared(mv: MVArray) -> MVArray:
    bulk_mv = bulk(mv)
    if bulk_mv.layout.size == 0:
        zeros = np.zeros(mv.batch_shape, dtype=np.result_type(mv.dtype, np.float64))
        return _scalar_mv(mv, zeros)
    return norm_squared(bulk_mv)


def bulk_norm(mv: MVArray) -> MVArray:
    bulksq = _require_scalar_output(bulk_norm_squared(mv), name="bulk_norm_squared(mv)")
    return _scalar_mv(mv, np.sqrt(np.abs(bulksq)))


def _coefficient_magnitude_squared(mv: MVArray) -> np.ndarray:
    values = np.asarray(mv.values, dtype=np.result_type(mv.dtype, np.float64))
    if values.shape[-1] == 0:
        return np.zeros(mv.batch_shape, dtype=values.dtype)
    return np.asarray(np.sum(values * values, axis=-1), dtype=values.dtype)


def weight_norm_squared(mv: MVArray) -> MVArray:
    weighted = weight(mv)
    return _scalar_mv(mv, _coefficient_magnitude_squared(weighted))


def weight_norm(mv: MVArray) -> MVArray:
    weightsq = _require_scalar_output(weight_norm_squared(mv), name="weight_norm_squared(mv)")
    return _scalar_mv(mv, np.sqrt(weightsq))


def bulk_normalize(mv: MVArray) -> MVArray:
    magnitudes = _require_scalar_output(bulk_norm(mv), name="bulk_norm(mv)")
    if np.any(np.isclose(magnitudes, 0.0)):
        raise ValueError("bulk_normalize() is undefined for zero bulk magnitude.")
    reciprocals = np.reciprocal(magnitudes)
    return _row_scale_mv(mv, reciprocals)


def unitize(mv: MVArray) -> MVArray:
    magnitudes = _require_scalar_output(weight_norm(mv), name="weight_norm(mv)")
    if np.any(np.isclose(magnitudes, 0.0)):
        raise ValueError("unitize() is undefined for zero weight magnitude.")
    reciprocals = np.reciprocal(magnitudes)
    return _row_scale_mv(mv, reciprocals)


def rigid_body_normalize(mv: MVArray) -> MVArray:
    _require_degenerate_algebra(mv, name="rigid_body_normalize")
    grade_set = set(mv.grades)
    if not grade_set.issubset({0, 2, 4}):
        raise ValueError(
            "rigid_body_normalize() currently requires an even PGA motor-like multivector "
            "with only grades 0, 2, and optional pseudoscalar grade 4 terms."
        )
    return bulk_normalize(mv)


def left_contraction(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "left_contraction")


def right_contraction(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "right_contraction")


def regressive_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    return _execute_binary_product(lhs, rhs, "regressive")


def sandwich(actor: MVArray, target: MVArray) -> MVArray:
    ensure_compatible(actor, target)
    return geometric_product(geometric_product(actor, target), inverse(actor))


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
