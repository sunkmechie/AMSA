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

from typing import Any

import numpy as np

from amsa.algebra import Algebra
from amsa.mv import MVArray
from amsa.ops import row_scale, scale
from amsa.storage import StorageRequest, is_jax_array


def _euclidean_dimension(alg: Algebra) -> int:
    if alg.dimension < 3 or alg.signature[-2:] != (1, -1):
        raise ValueError("CGA helpers require an AMSA CGA algebra with signature (1^n, 1, -1).")
    return alg.dimension - 2


def _basis_vector(alg: Algebra, axis: int, *, backend: StorageRequest = "auto") -> MVArray:
    return alg.blade(1 << axis, backend=backend)


def _coefficient_array(value: Any) -> Any:
    return value if is_jax_array(value) else np.asarray(value)


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
    values = _coefficient_array(coordinates)
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
    n = _euclidean_dimension(alg)
    coords = _coefficient_array(coordinates)
    if coords.shape[-1:] != (n,):
        raise ValueError(f"Expected coordinates with trailing dimension {n}.")
    radius_sq = (coords * coords).sum(axis=-1)
    xp = _array_namespace(coords)
    conformal = xp.concatenate(
        [
            coords,
            _trailing_column(-0.5 + 0.5 * radius_sq),
            _trailing_column(0.5 + 0.5 * radius_sq),
        ],
        axis=-1,
    )
    layout = alg.sparse_layout(tuple(1 << i for i in range(n + 2)))
    return alg.multivector(conformal, layout=layout, backend=backend)


def _array_namespace(value: Any) -> Any:
    if is_jax_array(value):
        import jax.numpy as jnp

        return jnp
    return np


def _trailing_column(value: Any) -> Any:
    if value.ndim == 0:
        return value[None]
    return value[..., np.newaxis]


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
        0.5 * _coefficient_array(radius) ** 2,
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
        _coefficient_array(distance),
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


def point_pair(alg: Algebra, a: MVArray, b: MVArray) -> MVArray:
    """Return the direct point pair through two conformal points."""
    _euclidean_dimension(alg)
    if a.algebra != alg.spec or b.algebra != alg.spec:
        raise ValueError("CGA objects must belong to the same algebra as the provided algebra.")
    return a ^ b


def line_from_point_direction(
    alg: Algebra,
    point_on_line: Any,
    direction: Any,
    *,
    backend: StorageRequest = "auto",
) -> MVArray:
    """Return a direct line from one Euclidean point and a direction vector."""
    p = _coefficient_array(point_on_line)
    d = _coefficient_array(direction)
    n = _euclidean_dimension(alg)
    if p.shape[-1:] != (n,) or d.shape[-1:] != (n,):
        raise ValueError(f"Expected point and direction with trailing dimension {n}.")
    q = p + d
    return line_through_points(alg, point(alg, p, backend=backend), point(alg, q, backend=backend))


def circle(
    alg: Algebra,
    center: Any,
    radius: Any,
    normal: Any,
    *,
    backend: StorageRequest = "auto",
) -> MVArray:
    """Return a direct circle from Euclidean center, radius, and support normal."""
    n = _euclidean_dimension(alg)
    if n != 3:
        raise ValueError("circle() currently requires Algebra.cga3d().")
    c = np.asarray(center, dtype=float)
    axis = np.asarray(normal, dtype=float)
    if c.shape != (3,) or axis.shape != (3,):
        raise ValueError("Expected 3D center and normal vectors.")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        raise ValueError("Circle normal must be nonzero.")
    axis = axis / axis_norm
    u = _perpendicular_unit(axis)
    v = np.cross(axis, u)
    r = float(radius)
    return circle_through_points(
        alg,
        point(alg, c + r * u, backend=backend),
        point(alg, c + r * v, backend=backend),
        point(alg, c - r * u, backend=backend),
    )


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


def extract_euclidean_vector(mv: MVArray) -> np.ndarray:
    """Return the Euclidean coordinates stored in the blade coefficients.

    In the AMSA CGA convention, Euclidean basis blade components ``e1..en``
    directly carry the vector coordinates.  See ``docs/references.rst#cga``.
    """
    alg = Algebra(mv.algebra)
    n = _euclidean_dimension(alg)
    return _extract_euclidean_blade_components(mv, n)


def extract_point(mv: MVArray) -> np.ndarray:
    """Return Euclidean point coordinates from a conformal point MV.

    A conformal point ``X = n_o + x + 0.5 x² n_inf`` stores the Euclidean
    coordinates ``x`` directly in the ``e1..en`` blade coefficients when the
    point is in canonical form (``X · n_inf = -1``).  After versor actions
    the point may need re-normalization, so this function divides by
    ``-(X · n_inf)`` before extracting.

    See ``docs/references.rst#cga`` for the conformal point references.
    """
    alg = Algebra(mv.algebra)
    _euclidean_dimension(alg)
    n = alg.dimension - 2
    ninf = infinity(alg, backend=mv.storage_kind)
    s = -(mv.inner(ninf)).component(0)
    if np.ndim(s) == 0 and s == 0:
        raise ValueError("extract_point() received a point at infinity (X · n_inf = 0).")
    return np.asarray(_extract_euclidean_blade_components(mv, n) / _expand_scale(s))


def _extract_euclidean_blade_components(mv: MVArray, n: int) -> np.ndarray:
    return np.stack([mv.component(1 << i) for i in range(n)], axis=-1)


def _expand_scale(s: np.ndarray) -> np.ndarray:
    if np.ndim(s) == 0:
        return s
    return np.asarray(s)[..., np.newaxis]


def extract_sphere(mv: MVArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, radius) from a dual-sphere MV.

    The dual sphere ``S = C - 0.5 r² n_inf`` stores the Euclidean center
    coordinates in the ``e1..en`` coefficients, and ``r = sqrt(S²)``.
    See ``docs/references.rst#cga``.
    """
    alg = Algebra(mv.algebra)
    n = _euclidean_dimension(alg)
    no_weight = _no_coefficient(alg, mv)
    if np.any(np.isclose(no_weight, 0.0)):
        raise ValueError("extract_sphere() requires a nonzero n_o component.")
    center = _extract_euclidean_blade_components(mv, n) / _expand_scale(no_weight)
    sq = (mv * mv).component(0)
    radius = np.sqrt(np.abs(sq)) / np.abs(no_weight)
    return center, radius


def extract_plane(mv: MVArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (normal, signed_distance) from a dual-plane MV.

    The dual plane ``P = n + d n_inf`` stores the Euclidean normal ``n``
    in the ``e1..en`` coefficients and the signed distance ``d`` in the
    ``n_inf`` coefficient.  See ``docs/references.rst#cga``.
    """
    alg = Algebra(mv.algebra)
    n_axes = _euclidean_dimension(alg)
    normal = _extract_euclidean_blade_components(mv, n_axes)
    distance = _ninf_coefficient(alg, mv)
    return normal, distance


def extract_line(mv: MVArray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(point, direction)`` from a direct CGA3D line."""
    alg = Algebra(mv.algebra)
    n = _euclidean_dimension(alg)
    if n != 3:
        raise ValueError("extract_line() currently requires cga3d.")
    e_plus = 1 << n
    e_minus = 1 << (n + 1)
    direction = np.array([
        mv.component((1 << i) | e_plus | e_minus) for i in range(n)
    ], dtype=float)
    direction_norm_sq = float(np.dot(direction, direction))
    if direction_norm_sq < 1e-24:
        raise ValueError("Cannot extract geometry from a degenerate CGA line.")
    moment = np.zeros(3, dtype=float)
    for idx, i in enumerate([(1, 2), (2, 0), (0, 1)]):
        blade_plus = (1 << i[0]) | (1 << i[1]) | e_plus
        blade_minus = (1 << i[0]) | (1 << i[1]) | e_minus
        sign = 0.5 if idx != 1 else -0.5
        moment[idx] = sign * float(mv.component(blade_plus) + mv.component(blade_minus))
    point_on_line = np.cross(direction, moment) / direction_norm_sq
    return point_on_line, direction


def extract_circle(mv: MVArray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(center, radius, normal)`` from a direct CGA3D circle."""
    alg = Algebra(mv.algebra)
    if _euclidean_dimension(alg) != 3:
        raise ValueError("extract_circle() currently requires cga3d.")
    ninf = infinity(alg, backend=mv.storage_kind)
    center_point = mv * ninf * mv
    center = extract_point(center_point)

    scale_value = float(-(center_point.inner(ninf)).component(0))
    norm_value = float((mv * mv).component(0))
    if abs(scale_value) < 1e-12:
        raise ValueError("Cannot extract geometry from a degenerate CGA circle.")
    radius_sq = 2.0 * abs(norm_value) / abs(scale_value)
    radius = float(np.sqrt(max(radius_sq, 0.0)))

    support_plane = (mv ^ ninf) * alg.inverse(alg.pseudoscalar(1.0))
    normal, _ = extract_plane(support_plane)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("Cannot extract a support plane from a degenerate CGA circle.")
    return center, radius, normal / normal_norm


def _perpendicular_unit(normal: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    vector = axis - np.dot(axis, normal) * normal
    return np.asarray(vector / np.linalg.norm(vector), dtype=float)


def _no_coefficient(alg: Algebra, mv: MVArray) -> np.ndarray:
    n = alg.dimension - 2
    plus = np.asarray(mv.component(1 << n))
    minus = np.asarray(mv.component(1 << (n + 1)))
    return np.asarray(minus - plus)


def _ninf_coefficient(alg: Algebra, mv: MVArray) -> np.ndarray:
    n = alg.dimension - 2
    plus = np.asarray(mv.component(1 << n))
    minus = np.asarray(mv.component(1 << (n + 1)))
    return np.asarray(0.5 * (plus + minus))


__all__ = [
    "circle_through_points",
    "circle",
    "distance_squared",
    "euclidean_vector",
    "extract_euclidean_vector",
    "extract_plane",
    "extract_circle",
    "extract_point",
    "extract_line",
    "extract_sphere",
    "infinity",
    "line_through_points",
    "line_from_point_direction",
    "origin",
    "plane",
    "point",
    "point_pair",
    "sphere",
    "translate",
]
