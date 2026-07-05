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

from typing import Any, cast

import numpy as np

from amsa.fusion import optimize_sequence_ir
from amsa.ir import (
    ProductIR,
    SequenceIR,
    UnaryIR,
    output_layout_from_product_ir,
    output_layout_from_unary_ir,
)
from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.storage import (
    CSRStorage,
    _broadcast_row_indices,
    add_storage,
    coefficient_magnitude_squared_storage,
    gather_storage_columns,
    project_storage,
    reweight_storage,
    row_scale_storage,
    scale_storage,
    sub_storage,
)


def _execute_csr_product_ir(
    lhs: MVArray,
    rhs: MVArray,
    ir: ProductIR,
) -> MVArray:
    assert isinstance(lhs.storage, CSRStorage)
    assert isinstance(rhs.storage, CSRStorage)
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    layout = output_layout_from_product_ir(ir, lhs.algebra)
    dtype = np.dtype(np.result_type(lhs.dtype, rhs.dtype))
    row_count = int(np.prod(batch_shape, dtype=np.intp))
    lhs_rows = _broadcast_row_indices(lhs.storage.batch_shape, batch_shape)
    rhs_rows = _broadcast_row_indices(rhs.storage.batch_shape, batch_shape)

    terms_by_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for term in ir.terms:
        terms_by_pair.setdefault((term.lhs_col, term.rhs_col), []).append(
            (term.out_col, term.coefficient)
        )

    data_values: list[Any] = []
    index_values: list[int] = []
    indptr = np.zeros(row_count + 1, dtype=np.intp)

    nnz = 0
    for out_row, (lhs_row, rhs_row) in enumerate(zip(lhs_rows, rhs_rows, strict=True)):
        lhs_start = int(lhs.storage._payload.indptr[lhs_row])
        lhs_stop = int(lhs.storage._payload.indptr[lhs_row + 1])
        rhs_start = int(rhs.storage._payload.indptr[rhs_row])
        rhs_stop = int(rhs.storage._payload.indptr[rhs_row + 1])
        out_values: dict[int, Any] = {}

        for lhs_offset in range(lhs_start, lhs_stop):
            lhs_col = int(lhs.storage._payload.indices[lhs_offset])
            lhs_value = lhs.storage._payload.data[lhs_offset]
            for rhs_offset in range(rhs_start, rhs_stop):
                pair_terms = terms_by_pair.get(
                    (lhs_col, int(rhs.storage._payload.indices[rhs_offset]))
                )
                if pair_terms is None:
                    continue
                product = lhs_value * rhs.storage._payload.data[rhs_offset]
                for out_col, coefficient in pair_terms:
                    out_values[out_col] = out_values.get(
                        out_col, dtype.type(0)
                    ) + coefficient * product

        for column in sorted(out_values):
            value = out_values[column]
            if value == 0:
                continue
            index_values.append(column)
            data_values.append(value)
            nnz += 1
        indptr[out_row + 1] = nnz

    storage = CSRStorage._from_validated_arrays(
        np.asarray(data_values, dtype=dtype),
        np.asarray(index_values, dtype=np.intp),
        indptr,
        batch_shape=batch_shape,
        width=layout.size,
        dtype=dtype,
    )
    return MVArray(algebra=lhs.algebra, layout=layout, storage=storage)


def execute_product_ir(
    lhs: MVArray,
    rhs: MVArray,
    ir: ProductIR,
) -> MVArray:
    """Execute a ``ProductIR`` using NumPy broadcasting."""
    if isinstance(lhs.storage, CSRStorage) and isinstance(rhs.storage, CSRStorage):
        return _execute_csr_product_ir(lhs, rhs, ir)

    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)

    lhs_columns = tuple(dict.fromkeys(term.lhs_col for term in ir.terms))
    rhs_columns = tuple(dict.fromkeys(term.rhs_col for term in ir.terms))
    lhs_values = gather_storage_columns(lhs.storage, lhs_columns, batch_shape=batch_shape)
    rhs_values = gather_storage_columns(rhs.storage, rhs_columns, batch_shape=batch_shape)
    lhs_col_index = {col: i for i, col in enumerate(lhs_columns)}
    rhs_col_index = {col: i for i, col in enumerate(rhs_columns)}

    layout = output_layout_from_product_ir(ir, lhs.algebra)
    dtype = np.result_type(lhs.dtype, rhs.dtype)
    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)

    for term in ir.terms:
        result[..., term.out_col] += (
            term.coefficient
            * lhs_values[..., lhs_col_index[term.lhs_col]]
            * rhs_values[..., rhs_col_index[term.rhs_col]]
        )

    return MVArray(algebra=lhs.algebra, layout=layout, values=result)


def execute_unary_ir(
    mv: MVArray,
    ir: UnaryIR,
) -> MVArray:
    """Execute a ``UnaryIR`` using NumPy storage operations.

    Two code paths:
    - Pure weight application (reverse, involute, conjugate): broadcast
      weights onto storage columns in-place.
    - Weight + permutation (dual, undual, Poincare variants): project
      storage through the permutation map, then reweight.
    """
    layout = output_layout_from_unary_ir(ir, mv.algebra)

    if ir.is_permutation:
        assert ir.permutation is not None
        # Project each output column from its permuted source column.
        columns = tuple(ir.permutation)
        projected = project_storage(mv.storage, columns)
        # Apply per-column weights.
        transformed = reweight_storage(
            projected, np.asarray(ir.weights, dtype=mv.dtype)
        )
        return MVArray(algebra=mv.algebra, layout=layout, storage=transformed)

    # Pure weight case: input and output layouts are identical.
    transformed = reweight_storage(
        mv.storage, np.asarray(ir.weights, dtype=mv.dtype)
    )
    return MVArray(algebra=mv.algebra, layout=layout, storage=transformed)


def execute_sequence_ir(
    inputs: dict[str, Any],
    ir: SequenceIR,
) -> Any:
    """Execute a ``SequenceIR`` step-by-step using NumPy operations.

    Each step resolves its operand references from the environment dict,
    performs the computation, and stores the result under its ``output``
    name for downstream steps.

    This is the IR-native counterpart to the Python-level composition in
    ``ops.py``.  Backends that support fusion may compile the entire
    sequence into a single kernel; the NumPy backend executes faithfully
    step-by-step, with optional fusion support for common patterns.
    """
    ir = optimize_sequence_ir(ir)
    env: dict[str, Any] = dict(inputs)

    i = 0
    while i < len(ir.steps):
        step = ir.steps[i]
        operands = tuple(env[name] for name in step.operands)
        result: Any

        # Check for fusion opportunities
        if step.fusion == "scale_product" and i + 1 < len(ir.steps):
            # Fuse scale + binary_product
            next_step = ir.steps[i + 1]
            if next_step.kind == "binary_product":
                meta = step.metadata or {}
                factor = cast(float, meta.get("factor", 1.0))
                lhs = cast(MVArray, operands[0])
                # Get the RHS from the next step's operands
                rhs_name = next_step.operands[1]
                rhs = env[rhs_name]
                assert isinstance(next_step.ir, ProductIR)
                result = _execute_fused_scale_product(lhs, rhs, next_step.ir, factor)
                env[next_step.output] = result
                i += 2  # Skip both steps
                continue

        if step.fusion == "unary_product" and i + 1 < len(ir.steps):
            # Fuse unary + binary_product
            next_step = ir.steps[i + 1]
            if next_step.kind == "binary_product":
                assert isinstance(step.ir, UnaryIR)
                assert isinstance(next_step.ir, ProductIR)
                mv = cast(MVArray, operands[0])
                rhs_name = next_step.operands[1]
                rhs = env[rhs_name]
                result = _execute_fused_unary_product(mv, rhs, step.ir, next_step.ir)
                env[next_step.output] = result
                i += 2  # Skip both steps
                continue

        # Non-fused execution path
        if step.kind == "binary_product":
            assert isinstance(step.ir, ProductIR)
            result = execute_product_ir(
                cast(MVArray, operands[0]),
                cast(MVArray, operands[1]),
                step.ir,
            )
        elif step.kind == "unary":
            assert isinstance(step.ir, UnaryIR)
            result = execute_unary_ir(cast(MVArray, operands[0]), step.ir)
        elif step.kind == "scale":
            meta = step.metadata or {}
            factor = cast(float, meta.get("factor", 1.0))
            mv = cast(MVArray, operands[0])
            result = MVArray(
                algebra=mv.algebra,
                layout=mv.layout,
                storage=scale_storage(mv.storage, factor),
            )
        elif step.kind == "row_scale":
            meta = step.metadata or {}
            mv = cast(MVArray, operands[0])
            factors = meta.get("scales", operands[1] if len(operands) > 1 else 1.0)
            result = MVArray(
                algebra=mv.algebra,
                layout=mv.layout,
                storage=row_scale_storage(
                    mv.storage,
                    np.asarray(factors),
                ),
            )
        elif step.kind == "add":
            result = _mv_add(cast(MVArray, operands[0]), cast(MVArray, operands[1]))
        elif step.kind == "sub":
            result = _mv_sub(cast(MVArray, operands[0]), cast(MVArray, operands[1]))
        elif step.kind == "neg":
            mv = cast(MVArray, operands[0])
            result = MVArray(
                algebra=mv.algebra,
                layout=mv.layout,
                storage=scale_storage(mv.storage, -1),
            )
        elif step.kind == "component":
            meta = step.metadata or {}
            blade_obj = meta.get("blade", 0)
            blade = blade_obj if isinstance(blade_obj, int) else int(str(blade_obj))
            result = _component_values(cast(MVArray, operands[0]), blade)
        elif step.kind == "elementwise":
            result = _elementwise(
                tuple(np.asarray(operand) for operand in operands),
                step.metadata or {},
            )
        elif step.kind == "predicate":
            result = _predicate(
                tuple(np.asarray(operand) for operand in operands),
                step.metadata or {},
            )
        elif step.kind == "coefficient_norm_squared":
            result = _coefficient_magnitude_squared(cast(MVArray, operands[0]))
        elif step.kind == "exp_coefficients":
            result = _exp_coefficients(np.asarray(operands[0]))
        elif step.kind == "motor_exp_coefficients":
            result = _motor_exp_coefficients(np.asarray(operands[0]), np.asarray(operands[1]))
        elif step.kind == "simple_bivector_log_coefficients":
            result = _simple_bivector_log_coefficients(
                np.asarray(operands[0]),
                np.asarray(operands[1]),
            )
        elif step.kind == "pga3d_motor_log_coefficients":
            result = _pga3d_motor_log_coefficients(
                np.asarray(operands[0]),
                np.asarray(operands[1]),
                np.asarray(operands[2]),
            )
        elif step.kind == "scalar_mv_from_array":
            result = _scalar_mv_from_array(
                cast(MVArray, operands[0]),
                np.asarray(operands[1]),
            )
        elif step.kind == "single_blade_mv":
            meta = step.metadata or {}
            blade_meta = meta.get("blade")
            assert blade_meta is not None
            blade_int = int(blade_meta) if isinstance(blade_meta, int) else int(str(blade_meta))
            result = _single_blade_mv(cast(MVArray, operands[0]), blade_int)
        elif step.kind == "single_blade_mv_from_array":
            meta = step.metadata or {}
            blade_meta = meta.get("blade")
            assert blade_meta is not None
            blade_int = int(blade_meta) if isinstance(blade_meta, int) else int(str(blade_meta))
            result = _single_blade_mv_from_array(
                cast(MVArray, operands[0]),
                blade_int,
                np.asarray(operands[1]),
            )
        else:
            raise ValueError(f"Unknown step kind: {step.kind}")

        env[step.output] = result
        i += 1

    return env[ir.result]


def _component_values(mv: MVArray, blade: int) -> np.ndarray:
    from amsa.storage import storage_component

    try:
        column = mv.layout.blades.index(blade)
    except ValueError:
        return np.zeros(mv.batch_shape, dtype=mv.dtype)
    return np.asarray(storage_component(mv.storage, column), dtype=mv.dtype)


def _elementwise(operands: tuple[np.ndarray, ...], metadata: dict[str, object]) -> np.ndarray:
    function = metadata.get("function")
    if function == "abs":
        return np.asarray(np.abs(operands[0]))
    if function == "sqrt":
        return np.asarray(np.sqrt(operands[0]))
    if function == "sqrt_abs":
        return np.asarray(np.sqrt(np.abs(operands[0])))
    if function == "reciprocal":
        return np.asarray(np.reciprocal(operands[0]))
    raise ValueError(f"Unknown elementwise function: {function!r}")


def _predicate(operands: tuple[np.ndarray, ...], metadata: dict[str, object]) -> bool:
    function = metadata.get("function")
    if function == "allclose":
        return bool(np.allclose(operands[0], operands[1]))
    if function == "allclose_zero":
        return bool(np.allclose(operands[0], 0.0))
    if function == "any_close_zero":
        return bool(np.any(np.isclose(operands[0], 0.0)))
    if function == "any_negative":
        return bool(np.any(operands[0] < 0.0))
    if function == "pga3d_motor_log_pi_singular":
        zero_mask = np.isclose(operands[0], 0.0)
        return bool(np.any(zero_mask & ~np.isclose(operands[1], 1.0)))
    raise ValueError(f"Unknown predicate function: {function!r}")


def _coefficient_magnitude_squared(mv: MVArray) -> np.ndarray:
    return coefficient_magnitude_squared_storage(mv.storage)


def _exp_coefficients(scalar_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scalar_values, dtype=np.result_type(scalar_values.dtype, np.float64))
    positive_mask = values > 0.0
    negative_mask = values < 0.0
    zero_mask = np.isclose(values, 0.0)

    roots = np.sqrt(np.abs(values))
    scalar_coefficients = np.empty_like(roots, dtype=values.dtype)
    linear_coefficients = np.empty_like(roots, dtype=values.dtype)

    scalar_coefficients[positive_mask] = np.cosh(roots[positive_mask])
    linear_coefficients[positive_mask] = np.sinh(roots[positive_mask]) / roots[positive_mask]

    scalar_coefficients[negative_mask] = np.cos(roots[negative_mask])
    linear_coefficients[negative_mask] = np.sin(roots[negative_mask]) / roots[negative_mask]

    scalar_coefficients[zero_mask] = 1.0
    linear_coefficients[zero_mask] = 1.0
    return scalar_coefficients, linear_coefficients


def _motor_exp_coefficients(
    scalar_part: np.ndarray,
    pseudoscalar_part: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.result_type(scalar_part.dtype, pseudoscalar_part.dtype, np.float64)
    scalar = np.asarray(scalar_part, dtype=dtype)
    pseudoscalar = np.asarray(pseudoscalar_part, dtype=dtype)

    scalar_coeff = np.zeros(scalar.shape, dtype=dtype)
    pseudo_coeff = np.zeros(scalar.shape, dtype=dtype)
    linear_coeff = np.zeros(scalar.shape, dtype=dtype)
    dual_linear_coeff = np.zeros(scalar.shape, dtype=dtype)

    zero_mask = np.isclose(scalar, 0.0)
    circular_mask = scalar < 0.0
    hyperbolic_mask = scalar > 0.0

    if np.any(zero_mask):
        scalar_coeff[zero_mask] = 1.0
        linear_coeff[zero_mask] = 1.0
        pseudo_coeff[zero_mask] = 0.5 * pseudoscalar[zero_mask]
        dual_linear_coeff[zero_mask] = pseudoscalar[zero_mask] / 6.0

    if np.any(circular_mask):
        roots = np.sqrt(-scalar[circular_mask])
        delta = -pseudoscalar[circular_mask] / (2.0 * roots)
        sinc = np.sin(roots) / roots
        dsinc = (roots * np.cos(roots) - np.sin(roots)) / (roots * roots)

        scalar_coeff[circular_mask] = np.cos(roots)
        pseudo_coeff[circular_mask] = -delta * np.sin(roots)
        linear_coeff[circular_mask] = sinc
        dual_linear_coeff[circular_mask] = delta * dsinc

    if np.any(hyperbolic_mask):
        roots = np.sqrt(scalar[hyperbolic_mask])
        delta = pseudoscalar[hyperbolic_mask] / (2.0 * roots)
        sinhc = np.sinh(roots) / roots
        dsinhc = (roots * np.cosh(roots) - np.sinh(roots)) / (roots * roots)

        scalar_coeff[hyperbolic_mask] = np.cosh(roots)
        pseudo_coeff[hyperbolic_mask] = delta * np.sinh(roots)
        linear_coeff[hyperbolic_mask] = sinhc
        dual_linear_coeff[hyperbolic_mask] = delta * dsinhc

    return scalar_coeff, pseudo_coeff, linear_coeff, dual_linear_coeff


def _simple_bivector_log_coefficients(
    scalar_values: np.ndarray,
    square_values: np.ndarray,
) -> np.ndarray:
    dtype = np.result_type(scalar_values.dtype, square_values.dtype, np.float64)
    scalar = np.asarray(scalar_values, dtype=dtype)
    square = np.asarray(square_values, dtype=dtype)
    roots = np.sqrt(np.abs(square))
    coefficients = np.zeros_like(roots, dtype=dtype)

    circular_mask = square < 0.0
    hyperbolic_mask = square > 0.0
    null_mask = np.isclose(square, 0.0)

    if np.any(circular_mask):
        coefficients[circular_mask] = (
            np.arctan2(roots[circular_mask], scalar[circular_mask]) / roots[circular_mask]
        )
    if np.any(hyperbolic_mask):
        coefficients[hyperbolic_mask] = (
            np.arctanh(roots[hyperbolic_mask] / scalar[hyperbolic_mask])
            / roots[hyperbolic_mask]
        )
    if np.any(null_mask):
        coefficients[null_mask] = np.reciprocal(scalar[null_mask])

    return coefficients


def _pga3d_motor_log_coefficients(
    scalar_values: np.ndarray,
    pseudoscalar_values: np.ndarray,
    sine_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.result_type(scalar_values.dtype, pseudoscalar_values.dtype, sine_values.dtype)
    scalar = np.asarray(scalar_values, dtype=dtype)
    pseudoscalar = np.asarray(pseudoscalar_values, dtype=dtype)
    sine = np.asarray(sine_values, dtype=dtype)
    nonzero_mask = ~np.isclose(sine, 0.0)

    phi = np.zeros_like(sine, dtype=dtype)
    phi[nonzero_mask] = np.arctan2(sine[nonzero_mask], scalar[nonzero_mask])

    distance = np.zeros_like(sine, dtype=dtype)
    distance[nonzero_mask] = -pseudoscalar[nonzero_mask] / sine[nonzero_mask]

    alpha = np.zeros_like(sine, dtype=dtype)
    beta = np.zeros_like(sine, dtype=dtype)
    alpha[nonzero_mask] = phi[nonzero_mask] / sine[nonzero_mask]
    beta[nonzero_mask] = (
        distance[nonzero_mask]
        * (
            1.0
            - (
                phi[nonzero_mask]
                * scalar[nonzero_mask]
                / sine[nonzero_mask]
            )
        )
        / sine[nonzero_mask]
    )

    return alpha, beta


def _union_layout(lhs: MVArray, rhs: MVArray) -> tuple[MVArray, MVLayout]:
    if lhs.layout == rhs.layout:
        return rhs, lhs.layout
    blades = tuple(sorted(set(lhs.layout.blades) | set(rhs.layout.blades)))
    if len(blades) == lhs.algebra.blade_count:
        return rhs, MVLayout.dense(lhs.algebra)
    return rhs, MVLayout.sparse_pattern(lhs.algebra, blades, name="union")


def _mv_add(lhs: MVArray, rhs: MVArray) -> MVArray:
    _, layout = _union_layout(lhs, rhs)
    lhs_p = lhs.to_layout(layout)
    rhs_p = rhs.to_layout(layout)
    return MVArray(
        algebra=lhs.algebra,
        layout=layout,
        storage=add_storage(lhs_p.storage, rhs_p.storage),
    )


def _mv_sub(lhs: MVArray, rhs: MVArray) -> MVArray:
    _, layout = _union_layout(lhs, rhs)
    lhs_p = lhs.to_layout(layout)
    rhs_p = rhs.to_layout(layout)
    return MVArray(
        algebra=lhs.algebra,
        layout=layout,
        storage=sub_storage(lhs_p.storage, rhs_p.storage),
    )


def _execute_fused_scale_product(
    lhs: MVArray,
    rhs: MVArray,
    ir: ProductIR,
    factor: float,
) -> MVArray:
    """Execute fused scale + binary_product in a single pass.

    This avoids an intermediate allocation by scaling the LHS coefficients
    during the product computation.
    """
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)

    # Gather the minimal set of columns from each operand.
    lhs_columns = tuple(dict.fromkeys(term.lhs_col for term in ir.terms))
    rhs_columns = tuple(dict.fromkeys(term.rhs_col for term in ir.terms))
    lhs_values = gather_storage_columns(lhs.storage, lhs_columns, batch_shape=batch_shape)
    rhs_values = gather_storage_columns(rhs.storage, rhs_columns, batch_shape=batch_shape)
    lhs_col_index = {col: i for i, col in enumerate(lhs_columns)}
    rhs_col_index = {col: i for i, col in enumerate(rhs_columns)}

    layout = output_layout_from_product_ir(ir, lhs.algebra)
    dtype = np.result_type(lhs.dtype, rhs.dtype)
    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)

    # Apply scale factor during accumulation
    for term in ir.terms:
        result[..., term.out_col] += (
            term.coefficient
            * factor
            * lhs_values[..., lhs_col_index[term.lhs_col]]
            * rhs_values[..., rhs_col_index[term.rhs_col]]
        )

    return MVArray(algebra=lhs.algebra, layout=layout, values=result)


def _execute_fused_unary_product(
    mv: MVArray,
    rhs: MVArray,
    unary_ir: UnaryIR,
    product_ir: ProductIR,
) -> MVArray:
    """Execute fused unary + binary_product in a single pass.

    This applies the unary transformation (weights/permutation) during
    the product computation to avoid an intermediate allocation.
    """
    batch_shape = np.broadcast_shapes(mv.batch_shape, rhs.batch_shape)

    # Gather columns for RHS
    rhs_columns = tuple(dict.fromkeys(term.rhs_col for term in product_ir.terms))
    rhs_values = gather_storage_columns(rhs.storage, rhs_columns, batch_shape=batch_shape)
    rhs_col_index = {col: i for i, col in enumerate(rhs_columns)}

    # For LHS, we need to apply unary transformation
    if unary_ir.is_permutation:
        assert unary_ir.permutation is not None
        # Gather columns from permuted sources
        lhs_columns = tuple(dict.fromkeys(term.lhs_col for term in product_ir.terms))
        # Map LHS columns through permutation
        permuted_columns = tuple(unary_ir.permutation[col] for col in lhs_columns)
        lhs_values = gather_storage_columns(mv.storage, permuted_columns, batch_shape=batch_shape)
        # Apply weights
        weights = np.asarray(unary_ir.weights, dtype=mv.dtype)
        lhs_values = lhs_values * weights[np.array(lhs_columns)]
    else:
        # Pure weight case
        lhs_columns = tuple(dict.fromkeys(term.lhs_col for term in product_ir.terms))
        lhs_values = gather_storage_columns(mv.storage, lhs_columns, batch_shape=batch_shape)
        weights = np.asarray(unary_ir.weights, dtype=mv.dtype)
        lhs_values = lhs_values * weights[np.array(lhs_columns)]

    lhs_col_index = {col: i for i, col in enumerate(lhs_columns)}

    layout = output_layout_from_product_ir(product_ir, mv.algebra)
    dtype = np.result_type(mv.dtype, rhs.dtype)
    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)

    for term in product_ir.terms:
        result[..., term.out_col] += (
            term.coefficient
            * lhs_values[..., lhs_col_index[term.lhs_col]]
            * rhs_values[..., rhs_col_index[term.rhs_col]]
        )

    return MVArray(algebra=mv.algebra, layout=layout, values=result)


def _scalar_mv_from_array(reference: MVArray, values: np.ndarray) -> MVArray:
    scalar_layout = MVLayout.grade(reference.algebra, 0)
    dtype = np.result_type(reference.dtype, values.dtype)
    payload = np.asarray(values, dtype=dtype)
    if payload.shape == ():
        payload = np.asarray([payload.item()], dtype=dtype)
    else:
        payload = payload[..., np.newaxis]
    return MVArray(algebra=reference.algebra, layout=scalar_layout, values=payload)


def _single_blade_mv(reference: MVArray, blade: int) -> MVArray:
    """Construct a single-blade MVArray with coefficient 1 from reference."""
    layout = MVLayout.sparse_pattern(
        reference.algebra, (blade,), name=reference.algebra.blade_name(blade)
    )
    values = np.ones(reference.batch_shape + (1,), dtype=reference.dtype)
    return MVArray(algebra=reference.algebra, layout=layout, values=values)


def _single_blade_mv_from_array(reference: MVArray, blade: int, values: np.ndarray) -> MVArray:
    layout = MVLayout.sparse_pattern(
        reference.algebra, (blade,), name=reference.algebra.blade_name(blade)
    )
    dtype = np.result_type(reference.dtype, values.dtype)
    payload = np.asarray(values, dtype=dtype)
    if payload.shape == ():
        payload = np.asarray([payload.item()], dtype=dtype)
    else:
        payload = payload[..., np.newaxis]
    return MVArray(algebra=reference.algebra, layout=layout, values=payload)



class NumpyBackend:
    """NumPy-based execution backend implementing the ``Executor`` protocol."""

    def execute_product(self, lhs: MVArray, rhs: MVArray, ir: ProductIR) -> MVArray:
        return execute_product_ir(lhs, rhs, ir)

    def execute_unary(self, mv: MVArray, ir: UnaryIR) -> MVArray:
        return execute_unary_ir(mv, ir)

    def execute_sequence(self, inputs: dict[str, Any], ir: SequenceIR) -> Any:
        return execute_sequence_ir(inputs, ir)
