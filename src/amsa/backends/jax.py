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

try:
    import jax
    import jax.numpy as jnp
except ImportError as err:
    raise ImportError(
        "JAX is required for the JAX backend. "
        "Install with: uv pip install amsa-ga[jax]"
    ) from err

from amsa.ir import (
    ProductIR,
    SequenceIR,
    UnaryIR,
    output_layout_from_product_ir,
    output_layout_from_unary_ir,
)
from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.specs import AlgebraSpec
from amsa.storage import (
    DenseStorage,
    NumPyPayload,
    gather_storage_columns,
    project_storage,
    reweight_storage,
    row_scale_storage,
    scale_storage,
    storage_component,
)


def _flatten_mvarray(mv: MVArray) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if not isinstance(mv.storage, DenseStorage):
        raise TypeError("JAX pytrees currently support dense MVArray storage only.")
    return (mv.storage._payload.array,), (mv.algebra, mv.layout)


def _unflatten_mvarray(metadata: tuple[Any, ...], children: tuple[Any, ...]) -> MVArray:
    algebra, layout = metadata
    (values,) = children
    if not isinstance(algebra, AlgebraSpec):
        raise TypeError("Invalid MVArray pytree algebra metadata.")
    if not isinstance(layout, MVLayout):
        raise TypeError("Invalid MVArray pytree layout metadata.")
    # Avoid DenseStorage.from_array() here: it normalizes through NumPy, which
    # would reject abstract JAX values during jit/vmap tracing.
    storage = DenseStorage(_payload=NumPyPayload(array=values))
    return MVArray(algebra=algebra, layout=layout, storage=storage)


jax.tree_util.register_pytree_node(MVArray, _flatten_mvarray, _unflatten_mvarray)


def execute_product_ir(
    lhs: MVArray,
    rhs: MVArray,
    ir: ProductIR,
) -> MVArray:
    """Execute a ``ProductIR`` using JAX broadcasting."""
    batch_shape = jnp.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)

    # Gather the minimal set of columns from each operand.
    lhs_columns = tuple(dict.fromkeys(term.lhs_col for term in ir.terms))
    rhs_columns = tuple(dict.fromkeys(term.rhs_col for term in ir.terms))
    lhs_values = gather_storage_columns(lhs.storage, lhs_columns, batch_shape=batch_shape)
    rhs_values = gather_storage_columns(rhs.storage, rhs_columns, batch_shape=batch_shape)
    lhs_col_index = {col: i for i, col in enumerate(lhs_columns)}
    rhs_col_index = {col: i for i, col in enumerate(rhs_columns)}

    layout = output_layout_from_product_ir(ir, lhs.algebra)
    dtype = jnp.result_type(lhs.dtype, rhs.dtype)
    result = jnp.zeros(batch_shape + (layout.size,), dtype=dtype)

    for term in ir.terms:
        result = result.at[..., term.out_col].add(
            term.coefficient
            * lhs_values[..., lhs_col_index[term.lhs_col]]
            * rhs_values[..., rhs_col_index[term.rhs_col]]
        )

    return MVArray(algebra=lhs.algebra, layout=layout, values=result)


def execute_unary_ir(
    mv: MVArray,
    ir: UnaryIR,
) -> MVArray:
    """Execute a ``UnaryIR`` using JAX storage operations."""
    layout = output_layout_from_unary_ir(ir, mv.algebra)

    if ir.is_permutation:
        assert ir.permutation is not None
        # Project each output column from its permuted source column.
        columns = tuple(ir.permutation)
        projected = project_storage(mv.storage, columns)
        # Apply per-column weights.
        transformed = reweight_storage(
            projected, jnp.asarray(ir.weights, dtype=mv.dtype)
        )
        return MVArray(algebra=mv.algebra, layout=layout, storage=transformed)

    # Pure weight case: input and output layouts are identical.
    transformed = reweight_storage(
        mv.storage, jnp.asarray(ir.weights, dtype=mv.dtype)
    )
    return MVArray(algebra=mv.algebra, layout=layout, storage=transformed)


def execute_sequence_ir(
    inputs: dict[str, Any],
    ir: SequenceIR,
) -> Any:
    """Execute a ``SequenceIR`` step-by-step using JAX operations."""
    env: dict[str, Any] = dict(inputs)

    for step in ir.steps:
        operands = tuple(env[name] for name in step.operands)
        result: Any

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
            factor = meta.get("factor", 1.0)
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
                    jnp.asarray(factors),
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
                tuple(jnp.asarray(operand) for operand in operands),
                step.metadata or {},
            )
        elif step.kind == "predicate":
            result = _predicate(
                tuple(jnp.asarray(operand) for operand in operands),
                step.metadata or {},
            )
        elif step.kind == "coefficient_norm_squared":
            result = _coefficient_magnitude_squared(cast(MVArray, operands[0]))
        elif step.kind == "exp_coefficients":
            result = _exp_coefficients(jnp.asarray(operands[0]))
        elif step.kind == "motor_exp_coefficients":
            result = _motor_exp_coefficients(jnp.asarray(operands[0]), jnp.asarray(operands[1]))
        elif step.kind == "simple_bivector_log_coefficients":
            result = _simple_bivector_log_coefficients(
                jnp.asarray(operands[0]),
                jnp.asarray(operands[1]),
            )
        elif step.kind == "pga3d_motor_log_coefficients":
            result = _pga3d_motor_log_coefficients(
                jnp.asarray(operands[0]),
                jnp.asarray(operands[1]),
                jnp.asarray(operands[2]),
            )
        elif step.kind == "scalar_extract":
            result = _extract_scalar(cast(MVArray, operands[0]))
        elif step.kind == "scalar_mv_from_array":
            result = _scalar_mv_from_array(
                cast(MVArray, operands[0]),
                jnp.asarray(operands[1]),
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
                jnp.asarray(operands[1]),
            )
        else:
            raise ValueError(f"Unknown sequence step kind: {step.kind!r}")

        env[step.output] = result

    return env[ir.result]


def _component_values(mv: MVArray, blade: int) -> jnp.ndarray:
    try:
        column = mv.layout.blades.index(blade)
    except ValueError:
        return jnp.zeros(mv.batch_shape, dtype=mv.dtype)
    return jnp.asarray(storage_component(mv.storage, column), dtype=mv.dtype)


def _elementwise(operands: tuple[jnp.ndarray, ...], metadata: dict[str, object]) -> jnp.ndarray:
    function = metadata.get("function")
    if function == "abs":
        return jnp.asarray(jnp.abs(operands[0]))
    if function == "sqrt":
        return jnp.asarray(jnp.sqrt(operands[0]))
    if function == "sqrt_abs":
        return jnp.asarray(jnp.sqrt(jnp.abs(operands[0])))
    if function == "reciprocal":
        return jnp.asarray(jnp.reciprocal(operands[0]))
    raise ValueError(f"Unknown elementwise function: {function!r}")


def _predicate(operands: tuple[jnp.ndarray, ...], metadata: dict[str, object]) -> bool:
    function = metadata.get("function")
    if function == "allclose":
        return bool(jnp.allclose(operands[0], operands[1]))
    if function == "allclose_zero":
        return bool(jnp.allclose(operands[0], 0.0))
    if function == "any_close_zero":
        return bool(jnp.any(jnp.isclose(operands[0], 0.0)))
    if function == "any_negative":
        return bool(jnp.any(operands[0] < 0.0))
    if function == "pga3d_motor_log_pi_singular":
        zero_mask = jnp.isclose(operands[0], 0.0)
        return bool(jnp.any(zero_mask & ~jnp.isclose(operands[1], 1.0)))
    raise ValueError(f"Unknown predicate function: {function!r}")


def _coefficient_magnitude_squared(mv: MVArray) -> jnp.ndarray:
    dtype = jnp.result_type(mv.dtype, jnp.float64)
    values = jnp.asarray(mv.values, dtype=dtype)
    if values.shape[-1] == 0:
        return jnp.zeros(mv.batch_shape, dtype=dtype)
    return jnp.asarray(jnp.sum(values * values, axis=-1), dtype=dtype)


def _exp_coefficients(scalar_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    values = jnp.asarray(scalar_values, dtype=jnp.result_type(scalar_values.dtype, jnp.float64))
    positive_mask = values > 0.0
    negative_mask = values < 0.0
    zero_mask = jnp.isclose(values, 0.0)

    roots = jnp.sqrt(jnp.abs(values))
    scalar_coefficients = jnp.empty_like(roots, dtype=values.dtype)
    linear_coefficients = jnp.empty_like(roots, dtype=values.dtype)

    scalar_coefficients = scalar_coefficients.at[positive_mask].set(
        jnp.cosh(roots[positive_mask])
    )
    linear_coefficients = linear_coefficients.at[positive_mask].set(
        jnp.sinh(roots[positive_mask]) / roots[positive_mask]
    )

    scalar_coefficients = scalar_coefficients.at[negative_mask].set(
        jnp.cos(roots[negative_mask])
    )
    linear_coefficients = linear_coefficients.at[negative_mask].set(
        jnp.sin(roots[negative_mask]) / roots[negative_mask]
    )

    scalar_coefficients = scalar_coefficients.at[zero_mask].set(1.0)
    linear_coefficients = linear_coefficients.at[zero_mask].set(1.0)
    return scalar_coefficients, linear_coefficients


def _motor_exp_coefficients(
    scalar_part: jnp.ndarray,
    pseudoscalar_part: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = jnp.result_type(scalar_part.dtype, pseudoscalar_part.dtype, jnp.float64)
    scalar = jnp.asarray(scalar_part, dtype=dtype)
    pseudoscalar = jnp.asarray(pseudoscalar_part, dtype=dtype)

    scalar_coeff = jnp.zeros(scalar.shape, dtype=dtype)
    pseudo_coeff = jnp.zeros(scalar.shape, dtype=dtype)
    linear_coeff = jnp.zeros(scalar.shape, dtype=dtype)
    dual_linear_coeff = jnp.zeros(scalar.shape, dtype=dtype)

    zero_mask = jnp.isclose(scalar, 0.0)
    circular_mask = scalar < 0.0
    hyperbolic_mask = scalar > 0.0

    if jnp.any(zero_mask):
        scalar_coeff = scalar_coeff.at[zero_mask].set(1.0)
        linear_coeff = linear_coeff.at[zero_mask].set(1.0)
        pseudo_coeff = pseudo_coeff.at[zero_mask].set(0.5 * pseudoscalar[zero_mask])
        dual_linear_coeff = dual_linear_coeff.at[zero_mask].set(pseudoscalar[zero_mask] / 6.0)

    if jnp.any(circular_mask):
        roots = jnp.sqrt(-scalar[circular_mask])
        delta = -pseudoscalar[circular_mask] / (2.0 * roots)
        sinc = jnp.sin(roots) / roots
        dsinc = (roots * jnp.cos(roots) - jnp.sin(roots)) / (roots * roots)

        scalar_coeff = scalar_coeff.at[circular_mask].set(jnp.cos(roots))
        pseudo_coeff = pseudo_coeff.at[circular_mask].set(-delta * jnp.sin(roots))
        linear_coeff = linear_coeff.at[circular_mask].set(sinc)
        dual_linear_coeff = dual_linear_coeff.at[circular_mask].set(delta * dsinc)

    if jnp.any(hyperbolic_mask):
        roots = jnp.sqrt(scalar[hyperbolic_mask])
        delta = pseudoscalar[hyperbolic_mask] / (2.0 * roots)
        sinhc = jnp.sinh(roots) / roots
        dsinhc = (roots * jnp.cosh(roots) - jnp.sinh(roots)) / (roots * roots)

        scalar_coeff = scalar_coeff.at[hyperbolic_mask].set(jnp.cosh(roots))
        pseudo_coeff = pseudo_coeff.at[hyperbolic_mask].set(delta * jnp.sinh(roots))
        linear_coeff = linear_coeff.at[hyperbolic_mask].set(sinhc)
        dual_linear_coeff = dual_linear_coeff.at[hyperbolic_mask].set(delta * dsinhc)

    return scalar_coeff, pseudo_coeff, linear_coeff, dual_linear_coeff


def _simple_bivector_log_coefficients(
    scalar_values: jnp.ndarray,
    square_values: jnp.ndarray,
) -> jnp.ndarray:
    dtype = jnp.result_type(scalar_values.dtype, square_values.dtype, jnp.float64)
    scalar = jnp.asarray(scalar_values, dtype=dtype)
    square = jnp.asarray(square_values, dtype=dtype)
    roots = jnp.sqrt(jnp.abs(square))
    coefficients = jnp.zeros_like(roots, dtype=dtype)

    circular_mask = square < 0.0
    hyperbolic_mask = square > 0.0
    null_mask = jnp.isclose(square, 0.0)

    if jnp.any(circular_mask):
        coefficients = coefficients.at[circular_mask].set(
            jnp.arctan2(roots[circular_mask], scalar[circular_mask]) / roots[circular_mask]
        )
    if jnp.any(hyperbolic_mask):
        coefficients = coefficients.at[hyperbolic_mask].set(
            jnp.arctanh(roots[hyperbolic_mask] / scalar[hyperbolic_mask])
            / roots[hyperbolic_mask]
        )
    if jnp.any(null_mask):
        coefficients = coefficients.at[null_mask].set(jnp.reciprocal(scalar[null_mask]))

    return coefficients


def _pga3d_motor_log_coefficients(
    scalar_values: jnp.ndarray,
    pseudoscalar_values: jnp.ndarray,
    sine_values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = jnp.result_type(scalar_values.dtype, pseudoscalar_values.dtype, sine_values.dtype)
    scalar = jnp.asarray(scalar_values, dtype=dtype)
    pseudoscalar = jnp.asarray(pseudoscalar_values, dtype=dtype)
    sine = jnp.asarray(sine_values, dtype=dtype)
    nonzero_mask = ~jnp.isclose(sine, 0.0)

    phi = jnp.zeros_like(sine, dtype=dtype)
    phi = phi.at[nonzero_mask].set(jnp.arctan2(sine[nonzero_mask], scalar[nonzero_mask]))

    distance = jnp.zeros_like(sine, dtype=dtype)
    distance = distance.at[nonzero_mask].set(-pseudoscalar[nonzero_mask] / sine[nonzero_mask])

    alpha = jnp.zeros_like(sine, dtype=dtype)
    beta = jnp.zeros_like(sine, dtype=dtype)
    alpha = alpha.at[nonzero_mask].set(phi[nonzero_mask] / sine[nonzero_mask])
    beta = beta.at[nonzero_mask].set(
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
    blades = tuple(sorted(set(lhs.layout.blades) | set(rhs.layout.blades)))
    if len(blades) == lhs.algebra.blade_count:
        return rhs, MVLayout.dense(lhs.algebra)
    return rhs, MVLayout.sparse_pattern(lhs.algebra, blades, name="union")


def _mv_add(lhs: MVArray, rhs: MVArray) -> MVArray:
    _, layout = _union_layout(lhs, rhs)
    lhs_p = lhs.to_layout(layout)
    rhs_p = rhs.to_layout(layout)
    return MVArray(algebra=lhs.algebra, layout=layout, values=lhs_p.values + rhs_p.values)


def _mv_sub(lhs: MVArray, rhs: MVArray) -> MVArray:
    _, layout = _union_layout(lhs, rhs)
    lhs_p = lhs.to_layout(layout)
    rhs_p = rhs.to_layout(layout)
    return MVArray(algebra=lhs.algebra, layout=layout, values=lhs_p.values - rhs_p.values)


def _extract_scalar(mv: MVArray) -> MVArray:
    """Extract the scalar (blade 0) component as a grade-0 MVArray."""
    scalar_layout = MVLayout.grade(mv.algebra, 0)
    value = storage_component(mv.storage, 0) if 0 in mv.layout.blades else jnp.zeros(
        mv.batch_shape, dtype=mv.dtype
    )
    if value.ndim == 0:
        value = jnp.asarray([value.item()], dtype=mv.dtype)
    else:
        value = value[..., jnp.newaxis]
    return MVArray(algebra=mv.algebra, layout=scalar_layout, values=value)


def _scalar_mv_from_array(reference: MVArray, values: jnp.ndarray) -> MVArray:
    scalar_layout = MVLayout.grade(reference.algebra, 0)
    dtype = jnp.result_type(reference.dtype, values.dtype)
    payload = jnp.asarray(values, dtype=dtype)
    if payload.shape == ():
        payload = jnp.asarray([payload.item()], dtype=dtype)
    else:
        payload = payload[..., jnp.newaxis]
    return MVArray(algebra=reference.algebra, layout=scalar_layout, values=payload)


def _single_blade_mv(reference: MVArray, blade: int) -> MVArray:
    """Construct a single-blade MVArray with coefficient 1 from reference."""
    layout = MVLayout.sparse_pattern(
        reference.algebra, (blade,), name=reference.algebra.blade_name(blade)
    )
    values = jnp.ones(reference.batch_shape + (1,), dtype=reference.dtype)
    return MVArray(algebra=reference.algebra, layout=layout, values=values)


def _single_blade_mv_from_array(reference: MVArray, blade: int, values: jnp.ndarray) -> MVArray:
    layout = MVLayout.sparse_pattern(
        reference.algebra, (blade,), name=reference.algebra.blade_name(blade)
    )
    dtype = jnp.result_type(reference.dtype, values.dtype)
    payload = jnp.asarray(values, dtype=dtype)
    if payload.shape == ():
        payload = jnp.asarray([payload.item()], dtype=dtype)
    else:
        payload = payload[..., jnp.newaxis]
    return MVArray(algebra=reference.algebra, layout=layout, values=payload)


class JAXBackend:
    """JAX-based execution backend implementing the ``Executor`` protocol.

    This backend provides GPU-accelerated execution through JAX, with dense
    storage parity against the NumPy backend.
    """

    def execute_product(self, lhs: MVArray, rhs: MVArray, ir: ProductIR) -> MVArray:
        return execute_product_ir(lhs, rhs, ir)

    def execute_unary(self, mv: MVArray, ir: UnaryIR) -> MVArray:
        return execute_unary_ir(mv, ir)

    def execute_sequence(self, inputs: dict[str, Any], ir: SequenceIR) -> Any:
        return execute_sequence_ir(inputs, ir)
