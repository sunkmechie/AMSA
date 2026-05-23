# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number

import numpy as np

from amsa.layouts import MVLayout
from amsa.mv import MVArray
from amsa.ops import (
    add,
    conjugate,
    inner_product,
    involute,
    left_contraction,
    neg,
    norm_squared,
    outer_product,
    project_grades,
    reverse,
    right_contraction,
    scalar_product,
    scale,
    sub,
)
from amsa.specs import AlgebraSpec


@dataclass(frozen=True, slots=True)
class DualMV:
    """Naive forward-mode dual number over an ``MVArray`` value.

    ``real`` carries the primal multivector and ``tangent`` carries the seeded
    directional derivative in the same algebra.  This is intentionally small
    and explicit; it is a reference autodiff surface, not an execution backend.
    """

    real: MVArray
    tangent: MVArray

    def __post_init__(self) -> None:
        if self.real.algebra != self.tangent.algebra:
            raise ValueError("DualMV real and tangent parts must use the same algebra.")

    @property
    def algebra(self) -> AlgebraSpec:
        return self.real.algebra

    @property
    def layout(self) -> MVLayout:
        return self.real.layout

    @property
    def values(self) -> np.ndarray:
        return self.real.values

    def grade(self, *grades: int) -> DualMV:
        return DualMV(project_grades(self.real, *grades), project_grades(self.tangent, *grades))

    def reverse(self) -> DualMV:
        return DualMV(reverse(self.real), reverse(self.tangent))

    def involute(self) -> DualMV:
        return DualMV(involute(self.real), involute(self.tangent))

    def conjugate(self) -> DualMV:
        return DualMV(conjugate(self.real), conjugate(self.tangent))

    def norm_squared(self) -> DualMV:
        return DualMV(
            norm_squared(self.real),
            add(scalar_product(self.tangent, self.real), scalar_product(self.real, self.tangent)),
        )

    def scalar_product(self, other: DualMV | MVArray) -> DualMV:
        other = _as_dual(other, self.real)
        return DualMV(
            scalar_product(self.real, other.real),
            add(scalar_product(self.tangent, other.real), scalar_product(self.real, other.tangent)),
        )

    def inner(self, other: DualMV | MVArray) -> DualMV:
        other = _as_dual(other, self.real)
        return DualMV(
            inner_product(self.real, other.real),
            add(inner_product(self.tangent, other.real), inner_product(self.real, other.tangent)),
        )

    def outer(self, other: DualMV | MVArray) -> DualMV:
        other = _as_dual(other, self.real)
        return DualMV(
            outer_product(self.real, other.real),
            add(outer_product(self.tangent, other.real), outer_product(self.real, other.tangent)),
        )

    def left_contract(self, other: DualMV | MVArray) -> DualMV:
        other = _as_dual(other, self.real)
        return DualMV(
            left_contraction(self.real, other.real),
            add(
                left_contraction(self.tangent, other.real),
                left_contraction(self.real, other.tangent),
            ),
        )

    def right_contract(self, other: DualMV | MVArray) -> DualMV:
        other = _as_dual(other, self.real)
        return DualMV(
            right_contraction(self.real, other.real),
            add(
                right_contraction(self.tangent, other.real),
                right_contraction(self.real, other.tangent),
            ),
        )

    def __neg__(self) -> DualMV:
        return DualMV(neg(self.real), neg(self.tangent))

    def __add__(self, other: DualMV | MVArray | Number) -> DualMV:
        other_dual = _as_dual(other, self.real)
        return DualMV(add(self.real, other_dual.real), add(self.tangent, other_dual.tangent))

    def __radd__(self, other: MVArray | Number) -> DualMV:
        return self.__add__(other)

    def __sub__(self, other: DualMV | MVArray | Number) -> DualMV:
        other_dual = _as_dual(other, self.real)
        return DualMV(sub(self.real, other_dual.real), sub(self.tangent, other_dual.tangent))

    def __rsub__(self, other: MVArray | Number) -> DualMV:
        other_dual = _as_dual(other, self.real)
        return DualMV(sub(other_dual.real, self.real), sub(other_dual.tangent, self.tangent))

    def __mul__(self, other: DualMV | MVArray | Number) -> DualMV:
        if isinstance(other, Number):
            return DualMV(scale(self.real, other), scale(self.tangent, other))
        other = _as_dual(other, self.real)
        return DualMV(
            self.real * other.real,
            add(self.tangent * other.real, self.real * other.tangent),
        )

    def __rmul__(self, other: MVArray | Number) -> DualMV:
        if isinstance(other, Number):
            return self.__mul__(other)
        other_dual = _as_dual(other, self.real)
        return DualMV(
            other_dual.real * self.real,
            add(other_dual.tangent * self.real, other_dual.real * self.tangent),
        )

    def __xor__(self, other: DualMV | MVArray) -> DualMV:
        return self.outer(other)

    def __or__(self, other: DualMV | MVArray) -> DualMV:
        return self.inner(other)


def _zero_like(reference: MVArray) -> MVArray:
    return MVArray.zeros(
        reference.algebra,
        reference.layout,
        batch_shape=reference.batch_shape,
        dtype=reference.dtype,
    )


def _as_dual(value: DualMV | MVArray | Number, reference: MVArray) -> DualMV:
    if isinstance(value, DualMV):
        return value
    if isinstance(value, MVArray):
        return DualMV(value, _zero_like(value))
    if isinstance(value, Number):
        from amsa.algebra import Algebra

        real = Algebra(reference.algebra).scalar(value)
        return DualMV(real, _zero_like(real))
    raise TypeError(f"Unsupported dual value: {type(value)!r}.")


def directional_derivative(
    function: Callable[[DualMV], DualMV | MVArray],
    point: MVArray,
    seed: MVArray,
) -> MVArray:
    """Evaluate a forward-mode directional derivative at ``point``."""
    result = function(DualMV(point, seed))
    if isinstance(result, DualMV):
        return result.tangent
    return _zero_like(result)


def forward_grad(function: Callable[[DualMV], DualMV | MVArray], point: MVArray) -> np.ndarray:
    """Return a naive coefficient gradient for scalar-valued ``function``.

    The implementation seeds one coefficient direction at a time, so it is
    intentionally simple and slow.  It is useful as a reference gradient for
    tests and small robotics objectives.
    """
    if point.batch_shape:
        raise ValueError("forward_grad() currently expects an unbatched MVArray.")

    values = np.zeros(point.layout.size, dtype=np.result_type(point.dtype, np.float64))
    for i in range(point.layout.size):
        seed_values = np.zeros_like(point.values, dtype=values.dtype)
        seed_values[i] = 1.0
        seed = MVArray.from_array(point.algebra, point.layout, seed_values)
        tangent = directional_derivative(function, point, seed)
        if tangent.layout.size != 1 or tangent.layout.blades != (0,):
            raise ValueError("forward_grad() requires a scalar-valued objective.")
        values[i] = tangent.component(0)
    return values


__all__ = ["DualMV", "directional_derivative", "forward_grad"]
