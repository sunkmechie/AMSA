from __future__ import annotations

from typing import Any

import numpy as np

from amsa.mv import MVArray
from amsa.plans import OpPlan


def execute_binary_plan(lhs: MVArray, rhs: MVArray, plan: OpPlan) -> MVArray:
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    lhs_values = np.broadcast_to(lhs.values, batch_shape + (lhs.layout.size,))
    rhs_values = np.broadcast_to(rhs.values, batch_shape + (rhs.layout.size,))

    layout = plan.output_layout()
    dtype = np.result_type(lhs.values.dtype, rhs.values.dtype)
    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)
    out_index = {blade: index for index, blade in enumerate(layout.blades)}

    for term in plan.terms:
        result[..., out_index[term.out_blade]] += (
            term.coefficient * lhs_values[..., term.lhs_index] * rhs_values[..., term.rhs_index]
        )

    return MVArray(algebra=lhs.algebra, layout=layout, values=result)
