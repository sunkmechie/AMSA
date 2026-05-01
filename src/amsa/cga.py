# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Any

import numpy as np

from amsa.algebra import Algebra
from amsa.mv import MVArray
from amsa.ops import row_scale, scale
from amsa.storage import StorageRequest


def _euclidean_dimension(alg: Algebra) -> int:
    if alg.dimension < 3 or alg.signature[-2:] != (1, -1):
        raise ValueError("CGA helpers require an AMSA CGA algebra with signature (1^n, 1, -1).")
    return alg.dimension - 2


def _basis_vector(alg: Algebra, axis: int, *, backend: StorageRequest = "auto") -> MVArray:
    return alg.blade(1 << axis, backend=backend)


def origin(alg: Algebra, *, backend: StorageRequest = "auto") -> MVArray:
    """Return the conformal null origin vector ``n_o``."""
    n = _euclidean_dimension(alg)
    plus = _basis_vector(alg, n, backend=backend)
    minus = _basis_vector(alg, n + 1, backend=backend)
    return scale(minus - plus, 0.5)


def infinity(alg: Algebra, *, backend: StorageRequest = "auto") -> MVArray:
    """Return the conformal null infinity vector ``n_inf``."""
    n = _euclidean_dimension(alg)
    plus = _basis_vector(alg, n, backend=backend)
    minus = _basis_vector(alg, n + 1, backend=backend)
    return minus + plus


def euclidean_vector(
    alg: Algebra,
    coordinates: Any,
    *,
    backend: StorageRequest = "auto",
) -> MVArray:
    """Embed Euclidean coordinates in the Euclidean vector subspace of a CGA algebra."""
    n = _euclidean_dimension(alg)
    values = np.asarray(coordinates)
    if values.shape[-1:] != (n,):
        raise ValueError(f"Expected coordinates with trailing dimension {n}.")
    layout = alg.sparse_layout(tuple(1 << i for i in range(n)))
    return alg.multivector(values, layout=layout, backend=backend)


def point(
    alg: Algebra,
    coordinates: Any,
    *,
    backend: StorageRequest = "auto",
) -> MVArray:
    """Return the conformal point ``X = n_o + x + 0.5 * (x·x) n_inf``."""
    x = euclidean_vector(alg, coordinates, backend=backend)
    coords = np.asarray(coordinates)
    radius_sq = np.sum(coords * coords, axis=-1)
    return origin(alg, backend=backend) + x + row_scale(
        infinity(alg, backend=backend),
        0.5 * radius_sq,
    )


def sphere(
    alg: Algebra,
    center: Any,
    radius: Any,
    *,
    backend: StorageRequest = "auto",
) -> MVArray:
    """Return a dual sphere ``S = C - 0.5 r^2 n_inf``."""
    return point(alg, center, backend=backend) - row_scale(
        infinity(alg, backend=backend),
        0.5 * np.asarray(radius) ** 2,
    )


def plane(
    alg: Algebra,
    normal: Any,
    distance: Any,
    *,
    backend: StorageRequest = "auto",
) -> MVArray:
    """Return a dual plane ``P = n + d n_inf`` with Euclidean unit normal ``n``."""
    return euclidean_vector(alg, normal, backend=backend) + row_scale(
        infinity(alg, backend=backend),
        np.asarray(distance),
    )


def line_through_points(alg: Algebra, a: MVArray, b: MVArray) -> MVArray:
    """Return the direct line through two conformal points."""
    _euclidean_dimension(alg)
    if a.algebra != alg.spec or b.algebra != alg.spec:
        raise ValueError("CGA objects must belong to the same algebra as the provided algebra.")
    return a ^ b ^ infinity(alg)


def circle_through_points(alg: Algebra, a: MVArray, b: MVArray, c: MVArray) -> MVArray:
    """Return the direct circle through three conformal points."""
    _euclidean_dimension(alg)
    if a.algebra != alg.spec or b.algebra != alg.spec or c.algebra != alg.spec:
        raise ValueError("CGA objects must belong to the same algebra as the provided algebra.")
    return a ^ b ^ c


def distance_squared(alg: Algebra, a: MVArray, b: MVArray) -> Any:
    """Return Euclidean squared distance from normalized conformal points."""
    _euclidean_dimension(alg)
    if a.algebra != alg.spec or b.algebra != alg.spec:
        raise ValueError("CGA objects must belong to the same algebra as the provided algebra.")
    return -2.0 * (a.inner(b)).component(0)


def translate(alg: Algebra, displacement: Any, *, backend: StorageRequest = "auto") -> MVArray:
    """Return the CGA translator ``T = 1 - 0.5 t n_inf``."""
    t = euclidean_vector(alg, displacement, backend=backend)
    return alg.scalar(1.0, backend=backend) - scale(t * infinity(alg, backend=backend), 0.5)


def ensure_same_cga(*values: MVArray) -> None:
    if not values:
        return
    algebra = values[0].algebra
    Algebra(algebra)  # validates construction surface
    for value in values:
        if value.algebra != algebra:
            raise ValueError("CGA objects must belong to the same algebra.")
    _euclidean_dimension(Algebra(algebra))


__all__ = [
    "circle_through_points",
    "distance_squared",
    "euclidean_vector",
    "infinity",
    "line_through_points",
    "origin",
    "plane",
    "point",
    "sphere",
    "translate",
]
