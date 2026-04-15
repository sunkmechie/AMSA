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

"""NumPy execution backend for AMSA IR."""

from __future__ import annotations

import numpy as np

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
    gather_storage_columns,
    project_storage,
    reweight_storage,
    row_scale_storage,
    scale_storage,
)


def execute_product_ir(
    lhs: MVArray,
    rhs: MVArray,
    ir: ProductIR,
) -> MVArray:
    """Execute a ``ProductIR`` using NumPy broadcasting.

    This is the IR-native counterpart to ``reference.execute_binary_plan``.
    Where the reference executor works from ``OpPlan`` (blade-indexed), this
    function works from ``ProductIR`` (column-indexed) so that backends
    operate directly on storage layout slots.
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
    inputs: dict[str, MVArray],
    ir: SequenceIR,
) -> MVArray:
    """Execute a ``SequenceIR`` step-by-step using NumPy operations.

    Each step resolves its operand references from the environment dict,
    performs the computation, and stores the result under its ``output``
    name for downstream steps.

    This is the IR-native counterpart to the Python-level composition in
    ``ops.py``.  Backends that support fusion may compile the entire
    sequence into a single kernel; the NumPy backend executes faithfully
    step-by-step.
    """
    env: dict[str, MVArray] = dict(inputs)

    for step in ir.steps:
        operands = tuple(env[name] for name in step.operands)

        if step.kind == "binary_product":
            assert isinstance(step.ir, ProductIR)
            result = execute_product_ir(operands[0], operands[1], step.ir)
        elif step.kind == "unary":
            assert isinstance(step.ir, UnaryIR)
            result = execute_unary_ir(operands[0], step.ir)
        elif step.kind == "scale":
            meta = step.metadata or {}
            factor = meta.get("factor", 1.0)
            result = MVArray(
                algebra=operands[0].algebra,
                layout=operands[0].layout,
                storage=scale_storage(operands[0].storage, factor),
            )
        elif step.kind == "row_scale":
            meta = step.metadata or {}
            result = MVArray(
                algebra=operands[0].algebra,
                layout=operands[0].layout,
                storage=row_scale_storage(
                    operands[0].storage,
                    np.asarray(meta.get("scales", 1.0)),
                ),
            )
        elif step.kind == "add":
            result = _mv_add(operands[0], operands[1])
        elif step.kind == "sub":
            result = _mv_sub(operands[0], operands[1])
        elif step.kind == "neg":
            result = MVArray(
                algebra=operands[0].algebra,
                layout=operands[0].layout,
                storage=scale_storage(operands[0].storage, -1),
            )
        elif step.kind == "scalar_extract":
            result = _extract_scalar(operands[0])
        elif step.kind == "single_blade_mv":
            meta = step.metadata or {}
            blade = meta.get("blade")
            assert blade is not None
            blade_int = int(blade) if isinstance(blade, int) else int(str(blade))
            result = _single_blade_mv(operands[0], blade_int)
        else:
            raise ValueError(f"Unknown sequence step kind: {step.kind!r}")

        env[step.output] = result

    return env[ir.result]


# ---------------------------------------------------------------------------
# Minimal mv-like helpers for sequence execution — mirror ops.py semantics
# ---------------------------------------------------------------------------


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
    from amsa.storage import storage_component

    scalar_layout = MVLayout.grade(mv.algebra, 0)
    value = storage_component(mv.storage, 0) if 0 in mv.layout.blades else np.zeros(
        mv.batch_shape, dtype=mv.dtype
    )
    if value.ndim == 0:
        value = np.asarray([value.item()], dtype=mv.dtype)
    else:
        value = value[..., np.newaxis]
    return MVArray(algebra=mv.algebra, layout=scalar_layout, values=value)


def _single_blade_mv(reference: MVArray, blade: int) -> MVArray:
    """Construct a single-blade MVArray with coefficient 1 from reference."""
    layout = MVLayout.sparse_pattern(
        reference.algebra, (blade,), name=reference.algebra.blade_name(blade)
    )
    values = np.ones(reference.batch_shape + (1,), dtype=reference.dtype)
    return MVArray(algebra=reference.algebra, layout=layout, values=values)


# ---------------------------------------------------------------------------
# NumpyBackend — Executor implementation
# ---------------------------------------------------------------------------


class NumpyBackend:
    """NumPy-based execution backend implementing the ``Executor`` protocol."""

    def execute_product(self, lhs: MVArray, rhs: MVArray, ir: ProductIR) -> MVArray:
        return execute_product_ir(lhs, rhs, ir)

    def execute_unary(self, mv: MVArray, ir: UnaryIR) -> MVArray:
        return execute_unary_ir(mv, ir)

    def execute_sequence(self, inputs: dict[str, MVArray], ir: SequenceIR) -> MVArray:
        return execute_sequence_ir(inputs, ir)
