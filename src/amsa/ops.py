from __future__ import annotations

from typing import Any

import numpy as np

from amsa.layouts import MVLayout
from amsa.mv import MVArray


def ensure_compatible(lhs: MVArray, rhs: MVArray) -> None:
    """Validate that two multivectors share algebra and layout metadata."""
    if lhs.algebra != rhs.algebra:
        raise ValueError("Multivectors belong to different algebras.")


def neg(mv: MVArray) -> MVArray:
    return MVArray(algebra=mv.algebra, layout=mv.layout, values=-mv.values)


def _union_layout(lhs: MVArray, rhs: MVArray) -> MVLayout:
    ensure_compatible(lhs, rhs)
    if lhs.layout == rhs.layout:
        return lhs.layout

    blades = tuple(sorted(set(lhs.layout.blades) | set(rhs.layout.blades)))
    if len(blades) == lhs.algebra.blade_count:
        return MVLayout.dense(lhs.algebra)
    return MVLayout.sparse_pattern(lhs.algebra, blades, name="union")


def add(lhs: MVArray, rhs: MVArray) -> MVArray:
    layout = _union_layout(lhs, rhs)
    lhs_projected = lhs.to_layout(layout)
    rhs_projected = rhs.to_layout(layout)
    return MVArray(algebra=lhs.algebra, layout=layout, values=lhs_projected.values + rhs_projected.values)


def sub(lhs: MVArray, rhs: MVArray) -> MVArray:
    layout = _union_layout(lhs, rhs)
    lhs_projected = lhs.to_layout(layout)
    rhs_projected = rhs.to_layout(layout)
    return MVArray(algebra=lhs.algebra, layout=layout, values=lhs_projected.values - rhs_projected.values)


def reverse(mv: MVArray) -> MVArray:
    signs = np.asarray(
        [(-1) ** ((blade.bit_count() * (blade.bit_count() - 1)) // 2) for blade in mv.layout.blades],
        dtype=mv.dtype,
    )
    return MVArray(algebra=mv.algebra, layout=mv.layout, values=mv.values * signs)


def involute(mv: MVArray) -> MVArray:
    signs = np.asarray([(-1) ** blade.bit_count() for blade in mv.layout.blades], dtype=mv.dtype)
    return MVArray(algebra=mv.algebra, layout=mv.layout, values=mv.values * signs)


def conjugate(mv: MVArray) -> MVArray:
    return reverse(involute(mv))


def geometric_product(lhs: MVArray, rhs: MVArray) -> MVArray:
    ensure_compatible(lhs, rhs)
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    lhs_values = np.broadcast_to(lhs.values, batch_shape + (lhs.layout.size,))
    rhs_values = np.broadcast_to(rhs.values, batch_shape + (rhs.layout.size,))

    support: set[int] = set()
    accumulators: dict[int, np.ndarray[Any, Any]] = {}
    dtype = np.result_type(lhs.values.dtype, rhs.values.dtype)
    zero = np.zeros(batch_shape, dtype=dtype)

    for lhs_index, lhs_blade in enumerate(lhs.layout.blades):
        lhs_component = lhs_values[..., lhs_index]
        for rhs_index, rhs_blade in enumerate(rhs.layout.blades):
            coefficient, out_blade = lhs.algebra.blade_product(lhs_blade, rhs_blade)
            if coefficient == 0:
                continue
            support.add(out_blade)
            contribution = coefficient * lhs_component * rhs_values[..., rhs_index]
            if out_blade in accumulators:
                accumulators[out_blade] = accumulators[out_blade] + contribution
            else:
                accumulators[out_blade] = zero + contribution

    blades = tuple(sorted(support))
    if len(blades) == lhs.algebra.blade_count:
        layout = MVLayout.dense(lhs.algebra)
    else:
        layout = MVLayout.sparse_pattern(lhs.algebra, blades, name="gp")

    result = np.zeros(batch_shape + (layout.size,), dtype=dtype)
    for out_index, blade in enumerate(layout.blades):
        result[..., out_index] = accumulators[blade]
    return MVArray(algebra=lhs.algebra, layout=layout, values=result)
