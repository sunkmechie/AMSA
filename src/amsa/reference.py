from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from amsa.mv import MVArray
from amsa.plans import OpPlan
from amsa.storage import (
    DenseStorage,
    JAXStorage,
    StorageKind,
    gather_storage_columns,
    to_csr_storage,
)


def _infer_output_kind(lhs: MVArray, rhs: MVArray) -> StorageKind:
    if lhs.storage_kind == rhs.storage_kind == "jax":
        return "jax"
    if lhs.storage_kind == rhs.storage_kind == "csr":
        return "csr"
    return "dense"


def _gather_plan_inputs(
    lhs: MVArray,
    rhs: MVArray,
    plan: OpPlan,
    *,
    batch_shape: tuple[int, ...],
    output_kind: StorageKind,
) -> tuple[tuple[int, ...], tuple[int, ...], Any, Any]:
    lhs_columns = tuple(dict.fromkeys(term.lhs_index for term in plan.terms))
    rhs_columns = tuple(dict.fromkeys(term.rhs_index for term in plan.terms))

    if output_kind == "jax":
        import jax.numpy as jnp

        if not isinstance(lhs.storage, JAXStorage) or not isinstance(rhs.storage, JAXStorage):
            raise TypeError("JAX output kind requires JAX-backed inputs.")

        lhs_values = jnp.asarray(lhs.storage.array[..., list(lhs_columns)])
        rhs_values = jnp.asarray(rhs.storage.array[..., list(rhs_columns)])
        lhs_values = jnp.broadcast_to(lhs_values, batch_shape + (len(lhs_columns),))
        rhs_values = jnp.broadcast_to(rhs_values, batch_shape + (len(rhs_columns),))
        return lhs_columns, rhs_columns, lhs_values, rhs_values

    return (
        lhs_columns,
        rhs_columns,
        gather_storage_columns(lhs.storage, lhs_columns, batch_shape=batch_shape),
        gather_storage_columns(rhs.storage, rhs_columns, batch_shape=batch_shape),
    )


def _emit_result(
    lhs: MVArray,
    layout: Any,
    result: Any,
    *,
    output_kind: StorageKind,
) -> MVArray:
    if output_kind == "jax":
        return MVArray(
            algebra=lhs.algebra,
            layout=layout,
            storage=JAXStorage(result),
        )
    if output_kind == "csr":
        dense_storage = DenseStorage(np.asarray(result))
        return MVArray(
            algebra=lhs.algebra,
            layout=layout,
            storage=to_csr_storage(dense_storage),
        )
    return MVArray(algebra=lhs.algebra, layout=layout, values=np.asarray(result))


def _accumulate_numpy_result(
    lhs_values: np.ndarray[Any, Any],
    rhs_values: np.ndarray[Any, Any],
    plan: OpPlan,
    *,
    batch_shape: Sequence[int],
    layout_size: int,
    dtype: np.dtype[Any],
    lhs_column_index: dict[int, int],
    rhs_column_index: dict[int, int],
    out_index: dict[int, int],
) -> np.ndarray[Any, Any]:
    result = np.zeros(tuple(batch_shape) + (layout_size,), dtype=dtype)
    for term in plan.terms:
        result[..., out_index[term.out_blade]] += (
            term.coefficient
            * lhs_values[..., lhs_column_index[term.lhs_index]]
            * rhs_values[..., rhs_column_index[term.rhs_index]]
        )
    return result


def _accumulate_jax_result(
    lhs_values: Any,
    rhs_values: Any,
    plan: OpPlan,
    *,
    batch_shape: Sequence[int],
    layout_size: int,
    dtype: np.dtype[Any],
    lhs_column_index: dict[int, int],
    rhs_column_index: dict[int, int],
    out_index: dict[int, int],
) -> Any:
    import jax.numpy as jnp

    result = jnp.zeros(tuple(batch_shape) + (layout_size,), dtype=dtype)
    for term in plan.terms:
        contribution = (
            term.coefficient
            * lhs_values[..., lhs_column_index[term.lhs_index]]
            * rhs_values[..., rhs_column_index[term.rhs_index]]
        )
        result = result.at[..., out_index[term.out_blade]].add(contribution)
    return result


def execute_binary_plan(lhs: MVArray, rhs: MVArray, plan: OpPlan) -> MVArray:
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    output_kind = _infer_output_kind(lhs, rhs)
    lhs_columns, rhs_columns, lhs_values, rhs_values = _gather_plan_inputs(
        lhs,
        rhs,
        plan,
        batch_shape=batch_shape,
        output_kind=output_kind,
    )
    lhs_column_index = {column: index for index, column in enumerate(lhs_columns)}
    rhs_column_index = {column: index for index, column in enumerate(rhs_columns)}

    layout = plan.output_layout()
    dtype = np.result_type(lhs.dtype, rhs.dtype)
    out_index = {blade: index for index, blade in enumerate(layout.blades)}

    if output_kind == "jax":
        result = _accumulate_jax_result(
            lhs_values,
            rhs_values,
            plan,
            batch_shape=batch_shape,
            layout_size=layout.size,
            dtype=dtype,
            lhs_column_index=lhs_column_index,
            rhs_column_index=rhs_column_index,
            out_index=out_index,
        )
    else:
        result = _accumulate_numpy_result(
            np.asarray(lhs_values),
            np.asarray(rhs_values),
            plan,
            batch_shape=batch_shape,
            layout_size=layout.size,
            dtype=dtype,
            lhs_column_index=lhs_column_index,
            rhs_column_index=rhs_column_index,
            out_index=out_index,
        )

    return _emit_result(lhs, layout, result, output_kind=output_kind)
