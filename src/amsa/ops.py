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
from typing import Any, cast

import numpy as np

from amsa.ir import IRStep, SequenceIR, UnaryKind, build_product_ir, build_unary_ir, get_backend
from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.plans import OpKind, plan_binary_product
from amsa.specs import grade_of_blade


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
    return scale(mv, -1)


def scale(mv: MVArray, factor: Any) -> MVArray:
    backend = get_backend()
    ir = SequenceIR(
        name="scale",
        inputs=("input",),
        steps=(
            IRStep(
                kind="scale",
                operands=("input",),
                ir=None,
                output="output",
                metadata={"factor": factor},
            ),
        ),
        result="output",
    )
    return cast(MVArray, backend.execute_sequence({"input": mv}, ir))


def row_scale(mv: MVArray, factors: Any) -> MVArray:
    backend = get_backend()
    ir = SequenceIR(
        name="row_scale",
        inputs=("input", "scales"),
        steps=(
            IRStep(
                kind="row_scale",
                operands=("input", "scales"),
                ir=None,
                output="output",
            ),
        ),
        result="output",
    )
    return cast(MVArray, backend.execute_sequence({"input": mv, "scales": factors}, ir))


def _execute_sequence_value(inputs: dict[str, Any], ir: SequenceIR) -> Any:
    return get_backend().execute_sequence(inputs, ir)


def _component_values(mv: MVArray, blade: int) -> np.ndarray:
    ir = SequenceIR(
        name="component",
        inputs=("input",),
        steps=(
            IRStep(
                kind="component",
                operands=("input",),
                ir=None,
                output="values",
                metadata={"blade": blade},
            ),
        ),
        result="values",
    )
    return cast(np.ndarray, _execute_sequence_value({"input": mv}, ir))


def _elementwise_values(function: str, values: Any) -> np.ndarray:
    ir = SequenceIR(
        name=function,
        inputs=("values",),
        steps=(
            IRStep(
                kind="elementwise",
                operands=("values",),
                ir=None,
                output="result",
                metadata={"function": function},
            ),
        ),
        result="result",
    )
    return cast(np.ndarray, _execute_sequence_value({"values": values}, ir))


def _predicate(function: str, *values: Any) -> bool:
    inputs = tuple(f"value_{index}" for index in range(len(values)))
    ir = SequenceIR(
        name=function,
        inputs=inputs,
        steps=(
            IRStep(
                kind="predicate",
                operands=inputs,
                ir=None,
                output="result",
                metadata={"function": function},
            ),
        ),
        result="result",
    )
    return cast(bool, _execute_sequence_value(dict(zip(inputs, values, strict=True)), ir))


def _is_jax_tracer(value: Any) -> bool:
    value_type = type(value)
    return "jax" in value_type.__module__ and "Tracer" in value_type.__name__


def _exp_coefficients(scalar_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ir = SequenceIR(
        name="exp_coefficients",
        inputs=("scalar_values",),
        steps=(
            IRStep(
                kind="exp_coefficients",
                operands=("scalar_values",),
                ir=None,
                output="coefficients",
            ),
        ),
        result="coefficients",
    )
    return cast(
        tuple[np.ndarray, np.ndarray],
        _execute_sequence_value({"scalar_values": scalar_values}, ir),
    )


def _motor_exp_coefficients(
    scalar_part: np.ndarray,
    pseudoscalar_part: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ir = SequenceIR(
        name="motor_exp_coefficients",
        inputs=("scalar_part", "pseudoscalar_part"),
        steps=(
            IRStep(
                kind="motor_exp_coefficients",
                operands=("scalar_part", "pseudoscalar_part"),
                ir=None,
                output="coefficients",
            ),
        ),
        result="coefficients",
    )
    return cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        _execute_sequence_value(
            {
                "scalar_part": scalar_part,
                "pseudoscalar_part": pseudoscalar_part,
            },
            ir,
        ),
    )


def _simple_bivector_log_coefficients(
    scalar_values: np.ndarray,
    square_values: np.ndarray,
) -> np.ndarray:
    ir = SequenceIR(
        name="simple_bivector_log_coefficients",
        inputs=("scalar_values", "square_values"),
        steps=(
            IRStep(
                kind="simple_bivector_log_coefficients",
                operands=("scalar_values", "square_values"),
                ir=None,
                output="coefficients",
            ),
        ),
        result="coefficients",
    )
    return cast(
        np.ndarray,
        _execute_sequence_value(
            {"scalar_values": scalar_values, "square_values": square_values},
            ir,
        ),
    )


def _pga3d_motor_log_coefficients(
    scalar_values: np.ndarray,
    pseudoscalar_values: np.ndarray,
    sine_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ir = SequenceIR(
        name="pga3d_motor_log_coefficients",
        inputs=("scalar_values", "pseudoscalar_values", "sine_values"),
        steps=(
            IRStep(
                kind="pga3d_motor_log_coefficients",
                operands=("scalar_values", "pseudoscalar_values", "sine_values"),
                ir=None,
                output="coefficients",
            ),
        ),
        result="coefficients",
    )
    return cast(
        tuple[np.ndarray, np.ndarray],
        _execute_sequence_value(
            {
                "scalar_values": scalar_values,
                "pseudoscalar_values": pseudoscalar_values,
                "sine_values": sine_values,
            },
            ir,
        ),
    )


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
    backend = get_backend()
    ir = SequenceIR(
        name="add",
        inputs=("lhs", "rhs"),
        steps=(
            IRStep(
                kind="add",
                operands=("lhs", "rhs"),
                ir=None,
                output="output",
            ),
        ),
        result="output",
    )
    return cast(
        MVArray,
        backend.execute_sequence({"lhs": lhs_projected, "rhs": rhs_projected}, ir),
    )


def sub(lhs: MVArray, rhs: MVArray | Number) -> MVArray:
    rhs_mv, layout = _union_layout(lhs, rhs)
    lhs_projected = lhs.to_layout(layout)
    rhs_projected = rhs_mv.to_layout(layout)
    backend = get_backend()
    ir = SequenceIR(
        name="sub",
        inputs=("lhs", "rhs"),
        steps=(
            IRStep(
                kind="sub",
                operands=("lhs", "rhs"),
                ir=None,
                output="output",
            ),
        ),
        result="output",
    )
    return cast(
        MVArray,
        backend.execute_sequence({"lhs": lhs_projected, "rhs": rhs_projected}, ir),
    )


def reverse(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "reverse")


def involute(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "involute")


def conjugate(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "conjugate")


def dual(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "dual")


def undual(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "undual")


def poincare_dual(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "poincare_dual")


def poincare_undual(mv: MVArray) -> MVArray:
    return _execute_unary(mv, "poincare_undual")


def _execute_unary(mv: MVArray, kind: UnaryKind) -> MVArray:
    ir = build_unary_ir(mv.layout.blades, mv.algebra, kind)
    backend = get_backend()
    return backend.execute_unary(mv, ir)  # type: ignore[no-any-return]


def _execute_binary_product(lhs: MVArray, rhs: MVArray, kind: OpKind) -> MVArray:
    ensure_compatible(lhs, rhs)
    plan = plan_binary_product(lhs.layout, rhs.layout, kind)
    ir = build_product_ir(plan, lhs.storage_kind, rhs.storage_kind)
    backend = get_backend()
    return backend.execute_product(lhs, rhs, ir)  # type: ignore[no-any-return]


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
    return scale(result, 0.5)


def anticommutator_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    ensure_compatible(lhs, rhs)
    result = geometric_product(lhs, rhs) + geometric_product(rhs, lhs)
    return scale(result, 0.5)


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
    ir = SequenceIR(
        name="scalar_mv_from_array",
        inputs=("reference", "values"),
        steps=(
            IRStep(
                kind="scalar_mv_from_array",
                operands=("reference", "values"),
                ir=None,
                output="result",
            ),
        ),
        result="result",
    )
    return cast(MVArray, _execute_sequence_value({"reference": mv, "values": values}, ir))


def _single_blade_mv(mv: MVArray, blade: int, values: np.ndarray) -> MVArray:
    ir = SequenceIR(
        name="single_blade_mv_from_array",
        inputs=("reference", "values"),
        steps=(
            IRStep(
                kind="single_blade_mv_from_array",
                operands=("reference", "values"),
                ir=None,
                output="result",
                metadata={"blade": blade},
            ),
        ),
        result="result",
    )
    return cast(MVArray, _execute_sequence_value({"reference": mv, "values": values}, ir))


def _unit_blade_mv(mv: MVArray, blade: int) -> MVArray:
    ir = SequenceIR(
        name="single_blade_mv",
        inputs=("reference",),
        steps=(
            IRStep(
                kind="single_blade_mv",
                operands=("reference",),
                ir=None,
                output="result",
                metadata={"blade": blade},
            ),
        ),
        result="result",
    )
    return cast(MVArray, _execute_sequence_value({"reference": mv}, ir))


def _row_scale_mv(mv: MVArray, scales: np.ndarray) -> MVArray:
    return row_scale(mv, scales)


def _require_study_output(mv: MVArray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    pseudoscalar_blade = mv.algebra.pseudoscalar_blade
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    scalar_value = np.asarray(_component_values(mv, 0), dtype=resolved_dtype)
    pseudoscalar_value = np.asarray(
        _component_values(mv, pseudoscalar_blade), dtype=resolved_dtype
    )

    for blade in mv.layout.blades:
        if blade in (0, pseudoscalar_blade):
            continue
        component = np.asarray(_component_values(mv, blade), dtype=resolved_dtype)
        if not _predicate("allclose_zero", component):
            raise ValueError(f"{name} must be scalar + pseudoscalar valued for this operation.")

    return scalar_value, pseudoscalar_value


def _study_value_mv(
    mv: MVArray,
    scalar_values: np.ndarray,
    pseudoscalar_values: np.ndarray,
) -> MVArray:
    result = _scalar_mv(mv, scalar_values)
    if not _predicate("allclose_zero", pseudoscalar_values):
        result = result + _single_blade_mv(mv, mv.algebra.pseudoscalar_blade, pseudoscalar_values)
    return result


def _study_times_mv(
    mv: MVArray,
    scalar_values: np.ndarray,
    pseudoscalar_values: np.ndarray,
) -> MVArray:
    result = _row_scale_mv(mv, scalar_values)
    if not _predicate("allclose_zero", pseudoscalar_values):
        pseudoscalar = _single_blade_mv(mv, mv.algebra.pseudoscalar_blade, pseudoscalar_values)
        result = result + geometric_product(pseudoscalar, mv)
    return result


def norm(mv: MVArray) -> MVArray:
    normsq = norm_squared(mv)
    normsq_values = _require_scalar_output(normsq, name="norm_squared(mv)")
    magnitudes = _elementwise_values("sqrt_abs", normsq_values)
    return _scalar_mv(mv, magnitudes)


def normalize(mv: MVArray) -> MVArray:
    magnitudes = _require_scalar_output(norm(mv), name="norm(mv)")
    if _predicate("any_close_zero", magnitudes):
        raise ValueError("normalize() is undefined for zero-magnitude multivectors.")
    reciprocals = _elementwise_values("reciprocal", magnitudes)
    return _row_scale_mv(mv, reciprocals)


def _motor_exp_from_bivector(mv: MVArray) -> MVArray:
    if len(mv.algebra.signature) >= 2 and mv.algebra.signature[-2:] == (1, -1):
        if set(mv.grades) != {2}:
            raise ValueError("motor_exp() requires a pure bivector generator.")
        square = geometric_product(mv, mv)
        scalar_values = _scalar_output_or_zero(square, name="mv * mv")
        scalar_coefficients, linear_coefficients = _exp_coefficients(scalar_values)
        return _scalar_mv(mv, scalar_coefficients) + _row_scale_mv(mv, linear_coefficients)

    if mv.algebra.signature != (0, 1, 1, 1):
        raise ValueError("motor_exp() currently supports PGA3d bivector generators.")
    if set(mv.grades) != {2}:
        raise ValueError("motor_exp() currently requires a pure bivector generator.")

    square = geometric_product(mv, mv)
    scalar_part, pseudoscalar_part = _require_study_output(square, name="mv * mv")
    scalar_coeff, pseudo_coeff, linear_coeff, dual_linear_coeff = _motor_exp_coefficients(
        scalar_part,
        pseudoscalar_part,
    )

    return _study_value_mv(mv, scalar_coeff, pseudo_coeff) + _study_times_mv(
        mv,
        linear_coeff,
        dual_linear_coeff,
    )


def _scalar_output_or_zero(mv: MVArray, *, name: str) -> np.ndarray:
    if mv.layout.size == 0:
        return np.asarray(
            _component_values(mv, 0),
            dtype=np.result_type(mv.dtype, np.float64),
        )
    return _require_scalar_output(mv, name=name)


def exp(mv: MVArray) -> MVArray:
    square = geometric_product(mv, mv)
    try:
        scalar_values = _scalar_output_or_zero(square, name="mv * mv")
    except ValueError as exc:
        if set(mv.grades) == {2} and mv.algebra.signature == (0, 1, 1, 1):
            return _motor_exp_from_bivector(mv)
        raise exc

    scalar_coefficients, linear_coefficients = _exp_coefficients(scalar_values)

    return _scalar_mv(mv, scalar_coefficients) + _row_scale_mv(mv, linear_coefficients)


def motor_exp(mv: MVArray) -> MVArray:
    return _motor_exp_from_bivector(mv)


def _simple_bivector_log(mv: MVArray) -> MVArray:
    scalar_values = _require_scalar_output(project_grades(mv, 0), name="grade_0(mv)")
    bivector = project_grades(mv, 2)
    if bivector.layout.size == 0 or _predicate(
        "allclose_zero", _coefficient_magnitude_squared(bivector)
    ):
        if _predicate("any_negative", scalar_values):
            raise ValueError(
                "motor_log() is undefined on the principal branch for negative scalars."
            )
        return MVArray.zeros(mv.algebra, MVLayout.grade(mv.algebra, 2), batch_shape=mv.batch_shape)

    square = geometric_product(bivector, bivector)
    square_values = _require_scalar_output(square, name="grade_2(mv) * grade_2(mv)")
    coefficients = _simple_bivector_log_coefficients(scalar_values, square_values)

    return _row_scale_mv(bivector, coefficients)


def _motor_log_pga3d(mv: MVArray) -> MVArray:
    motor = rigid_body_normalize(mv)
    if not set(motor.grades).issubset({0, 2, 4}):
        raise ValueError(
            "motor_log() currently requires a PGA3d motor-like multivector with grades 0, 2, and 4."
        )

    scalar_values = np.asarray(
        _component_values(motor, 0),
        dtype=np.result_type(motor.dtype, np.float64),
    )
    pseudoscalar_blade = motor.algebra.pseudoscalar_blade
    pseudoscalar_values = np.asarray(
        _component_values(motor, pseudoscalar_blade),
        dtype=np.result_type(motor.dtype, np.float64),
    )
    bivector = project_grades(motor, 2)
    moment_part = bulk(bivector)
    sine_values = _require_scalar_output(bulk_norm(moment_part), name="bulk_norm(grade_2(mv))")
    sine_values = np.asarray(sine_values, dtype=np.result_type(motor.dtype, np.float64))

    if bivector.layout.size == 0 or _predicate(
        "allclose_zero", _coefficient_magnitude_squared(bivector)
    ):
        return MVArray.zeros(
            motor.algebra,
            MVLayout.grade(motor.algebra, 2),
            batch_shape=motor.batch_shape,
        )

    if _predicate("pga3d_motor_log_pi_singular", sine_values, scalar_values):
        raise ValueError("motor_log() does not support the pi-rotation singular branch yet.")

    if _predicate("allclose_zero", sine_values):
        return bivector

    alpha_values, beta_values = _pga3d_motor_log_coefficients(
        scalar_values,
        pseudoscalar_values,
        sine_values,
    )

    pseudoscalar = _unit_blade_mv(motor, pseudoscalar_blade)
    pseudoscalar_times_bivector = project_grades(geometric_product(pseudoscalar, bivector), 2)

    return _row_scale_mv(bivector, alpha_values) + _row_scale_mv(
        pseudoscalar_times_bivector,
        beta_values,
    )


def motor_log(mv: MVArray) -> MVArray:
    if len(mv.algebra.signature) >= 2 and mv.algebra.signature[-2:] == (1, -1):
        if not set(mv.grades).issubset({0, 2}):
            raise ValueError(
                "motor_log() currently supports CGA scalar+bivector Euclidean motors."
            )
        return _simple_bivector_log(mv)
    if mv.algebra.signature == (0, 1, 1):
        return _simple_bivector_log(rigid_body_normalize(mv))
    if mv.algebra.signature == (0, 1, 1, 1):
        return _motor_log_pga3d(mv)
    raise ValueError(
        "motor_log() currently supports CGA scalar+bivector Euclidean motors, "
        "PGA2d, and PGA3d motor-like multivectors."
    )


def log(mv: MVArray) -> MVArray:
    """Principal logarithm for simple scalar+bivector rotors.

    This is the algebra-generic path: the input must contain only grade-0 and
    grade-2 terms, and the bivector part must square to a scalar. PGA motor
    special cases stay in :func:`motor_log`.
    """
    if not set(mv.grades).issubset({0, 2}):
        raise NotImplementedError("log() currently supports scalar + simple bivector rotors.")
    return _simple_bivector_log(mv)


def bulk_norm_squared(mv: MVArray) -> MVArray:
    bulk_mv = bulk(mv)
    if bulk_mv.layout.size == 0:
        return _scalar_mv(mv, _component_values(bulk_mv, 0))
    return norm_squared(bulk_mv)


def bulk_norm(mv: MVArray) -> MVArray:
    bulksq = _require_scalar_output(bulk_norm_squared(mv), name="bulk_norm_squared(mv)")
    return _scalar_mv(mv, _elementwise_values("sqrt_abs", bulksq))


def _coefficient_magnitude_squared(mv: MVArray) -> np.ndarray:
    ir = SequenceIR(
        name="coefficient_norm_squared",
        inputs=("input",),
        steps=(
            IRStep(
                kind="coefficient_norm_squared",
                operands=("input",),
                ir=None,
                output="result",
            ),
        ),
        result="result",
    )
    return cast(np.ndarray, _execute_sequence_value({"input": mv}, ir))


def weight_norm_squared(mv: MVArray) -> MVArray:
    weighted = weight(mv)
    return _scalar_mv(mv, _coefficient_magnitude_squared(weighted))


def weight_norm(mv: MVArray) -> MVArray:
    weightsq = _require_scalar_output(weight_norm_squared(mv), name="weight_norm_squared(mv)")
    return _scalar_mv(mv, _elementwise_values("sqrt", weightsq))


def bulk_normalize(mv: MVArray) -> MVArray:
    magnitudes = _require_scalar_output(bulk_norm(mv), name="bulk_norm(mv)")
    if _predicate("any_close_zero", magnitudes):
        raise ValueError("bulk_normalize() is undefined for zero bulk magnitude.")
    reciprocals = _elementwise_values("reciprocal", magnitudes)
    return _row_scale_mv(mv, reciprocals)


def unitize(mv: MVArray) -> MVArray:
    magnitudes = _require_scalar_output(weight_norm(mv), name="weight_norm(mv)")
    if _predicate("any_close_zero", magnitudes):
        raise ValueError("unitize() is undefined for zero weight magnitude.")
    reciprocals = _elementwise_values("reciprocal", magnitudes)
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
    """Return the regressive (meet) product of two multivectors.

    Defined as ``poincare_undual(poincare_dual(A) * poincare_dual(B))``
    where ``*`` is the full geometric product of the duals.  Using the
    geometric product (rather than just the outer product) preserves
    interior terms when Poincaré-dual grades exceed the algebra dimension,
    e.g. for CGA meet of dual spheres in 5D.
    See ``docs/references.rst#operations``.
    """
    return _execute_binary_product(lhs, rhs, "regressive")


def sandwich(actor: MVArray, target: MVArray) -> MVArray:
    ensure_compatible(actor, target)
    return geometric_product(geometric_product(actor, target), inverse(actor))


def _require_scalar_output(mv: MVArray, *, name: str) -> np.ndarray:
    scalar_blade = 0
    resolved_dtype = np.result_type(mv.dtype, np.float64)
    scalar_value_raw = _component_values(mv, scalar_blade)
    if _is_jax_tracer(scalar_value_raw):
        return scalar_value_raw
    scalar_value = np.asarray(scalar_value_raw, dtype=resolved_dtype)

    if mv.layout.size == 0:
        raise ValueError(f"{name} is zero and therefore non-invertible.")

    for blade in mv.layout.blades:
        if blade == scalar_blade:
            continue
        component = np.asarray(_component_values(mv, blade), dtype=resolved_dtype)
        if not _predicate("allclose_zero", component):
            raise ValueError(f"{name} must be scalar-valued for this operation.")

    return scalar_value


def inverse(mv: MVArray) -> MVArray:
    reversed_mv = reverse(mv)
    left_norm_mv = geometric_product(reversed_mv, mv)
    right_norm_mv = geometric_product(mv, reversed_mv)
    left_norm = _require_scalar_output(left_norm_mv, name="reverse(mv) * mv")
    right_norm = _require_scalar_output(right_norm_mv, name="mv * reverse(mv)")

    traced_norm = _is_jax_tracer(left_norm) or _is_jax_tracer(right_norm)
    if not traced_norm and not _predicate("allclose", left_norm, right_norm):
        raise ValueError("inverse() currently requires matching scalar left/right reverse norms.")
    if not traced_norm and _predicate("any_close_zero", left_norm):
        raise ValueError("inverse() is undefined for zero-norm or non-invertible multivectors.")

    reciprocals = _elementwise_values("reciprocal", left_norm)
    return row_scale(reversed_mv, reciprocals)


def divide(lhs: MVArray | Number, rhs: MVArray | Number) -> MVArray:
    if isinstance(lhs, MVArray):
        if isinstance(rhs, MVArray):
            return geometric_product(lhs, inverse(rhs))
        if isinstance(rhs, Number):
            rhs_array = np.asarray(rhs)
            if _predicate("any_close_zero", rhs_array):
                raise ZeroDivisionError("Division by zero scalar.")
            return scale(lhs, _elementwise_values("reciprocal", rhs_array))
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
