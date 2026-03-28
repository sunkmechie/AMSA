from __future__ import annotations

import numpy as np

from amsa.mv import MVArray
from amsa.plans import OpPlan
from amsa.storage import gather_storage_columns


def execute_binary_plan(lhs: MVArray, rhs: MVArray, plan: OpPlan) -> MVArray:
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    lhs_columns = tuple(dict.fromkeys(term.lhs_index for term in plan.terms))
    rhs_columns = tuple(dict.fromkeys(term.rhs_index for term in plan.terms))
    lhs_values = gather_storage_columns(lhs.storage, lhs_columns, batch_shape=batch_shape)
    rhs_values = gather_storage_columns(rhs.storage, rhs_columns, batch_shape=batch_shape)
    lhs_column_index = {column: index for index, column in enumerate(lhs_columns)}
    rhs_column_index = {column: index for index, column in enumerate(rhs_columns)}

    layout = plan.output_layout()
    dtype = np.result_type(lhs.dtype, rhs.dtype)
    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)
    out_index = {blade: index for index, blade in enumerate(layout.blades)}

    for term in plan.terms:
        result[..., out_index[term.out_blade]] += (
            term.coefficient
            * lhs_values[..., lhs_column_index[term.lhs_index]]
            * rhs_values[..., rhs_column_index[term.rhs_index]]
        )

    return MVArray(algebra=lhs.algebra, layout=layout, values=result)
