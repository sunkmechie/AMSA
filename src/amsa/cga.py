# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from amsa.algebra import Algebra
from amsa.mv import MVArray
from amsa.ops import row_scale, scale
from amsa.storage import StorageRequest

if TYPE_CHECKING:
    from amsa.algebra import EntityInfo


_TOL = 1e-10


def _classify_cga(alg: Algebra, mv: MVArray) -> EntityInfo:
    """Classify a multivector in a CGA algebra.

    Returns an :class:`~amsa.algebra.EntityInfo` with geometric interpretation,
    invariants, and storage metadata.  Never mutates the input.
    """
    from amsa.algebra import EntityInfo

    _euclidean_dimension(alg)
    n = alg.dimension - 2

    def _info(**overrides: Any) -> EntityInfo:
        kw: dict[str, Any] = {
            "algebra": f"cga{n}d",
            "kind": "unknown multivector",
            "representation": "",
            "grades": tuple(sorted(_grades_of_nonzero(mv))),
            "storage": {
                "layout": mv.storage_kind,
                "backend": "numpy",
                "batch_shape": mv.batch_shape,
                "dtype": str(mv.dtype),
            },
            "warnings": [],
            "ambiguous": False,
            "null": False,
            "normalized": False,
            "invariants": {},
            "geometric_data": {},
        }

        kw["null"] = _is_null(mv)
        sq = _scalar_square(mv)
        ninf_inner = _ninf_inner(alg, mv)
        kw["invariants"] = {"X²": float(sq) if sq is not None else float("nan")}
        if not np.isnan(ninf_inner):
            kw["invariants"]["X·n∞"] = float(ninf_inner)
        kw["normalized"] = bool(np.allclose(ninf_inner, -1.0, atol=_TOL))

        if _is_zero_mv(mv):
            kw["kind"] = "zero multivector"
            return EntityInfo(**{**kw, **overrides})

        grades_set = set(kw["grades"])
        single_grade = len(grades_set) == 1
        grade1 = grades_set == {1}
        grade3 = grades_set == {3}
        only_even = grades_set.issubset({0, 2, 4})
        only_scalar_bivector = grades_set.issubset({0, 2})
        is_null = bool(kw["null"])

        no_c = _no_coefficient(alg, mv)
        ninf_c = _ninf_coefficient(alg, mv)
        has_ninf_in_support = _any_blade_contains_conformal_axes(alg, mv)

        if single_grade:
            if grade1:
                if is_null:
                    if kw["normalized"]:
                        kw["kind"] = "normalized conformal point"
                    elif np.allclose(ninf_inner, 0.0, atol=_TOL):
                        kw["kind"] = "point at infinity"
                    else:
                        kw["kind"] = "conformal point"
                    kw["representation"] = "direct"
                    try:
                        kw["geometric_data"] = {
                            "coordinates": extract_point(mv),
                        }
                    except Exception:
                        kw["warnings"].append("could not extract point coordinates")
                    return EntityInfo(**{**kw, **overrides})

                if abs(no_c) > _TOL:
                    kw["kind"] = "dual sphere"
                    kw["representation"] = "dual"
                    try:
                        center, radius = extract_sphere(mv)
                        kw["geometric_data"] = {"center": center, "radius": radius}
                    except Exception:
                        kw["warnings"].append("could not extract sphere parameters")
                    return EntityInfo(**{**kw, **overrides})

                if abs(no_c) <= _TOL and abs(ninf_c) > _TOL:
                    kw["kind"] = "dual plane"
                    kw["representation"] = "dual"
                    try:
                        normal, distance = extract_plane(mv)
                        kw["geometric_data"] = {"normal": normal, "signed_distance": distance}
                    except Exception:
                        kw["warnings"].append("could not extract plane parameters")
                    if abs(no_c) <= _TOL and abs(ninf_c) <= _TOL:
                        kw["ambiguous"] = True
                        kw["warnings"].append("matches both dual plane and Euclidean vector")
                    return EntityInfo(**{**kw, **overrides})

                kw["kind"] = "generic vector"
                return EntityInfo(**{**kw, **overrides})

            if grade3:
                if has_ninf_in_support:
                    kw["kind"] = "direct line"
                else:
                    kw["kind"] = "direct circle"
                kw["representation"] = "direct"
                return EntityInfo(**{**kw, **overrides})

            kw["kind"] = "generic blade"
            return EntityInfo(**{**kw, **overrides})

        if only_scalar_bivector:
            if has_ninf_in_support:
                kw["kind"] = "translator candidate"
                kw["representation"] = "direct"
                return EntityInfo(**{**kw, **overrides})
            kw["kind"] = "even versor"
            return EntityInfo(**{**kw, **overrides})

        if only_even:
            kw["kind"] = "even multivector"
            return EntityInfo(**{**kw, **overrides})

        return EntityInfo(**{**kw, **overrides})

    return _info()


def _is_null(mv: MVArray) -> bool:
    sq = _scalar_square(mv)
    if sq is None:
        return False
    return bool(np.allclose(sq, 0.0, atol=_TOL))


def _grades_of_nonzero(mv: MVArray) -> set[int]:
    """Return grades of blades with significant coefficients."""
    from amsa.specs import grade_of_blade

    values = np.asarray(mv.values)
    if values.ndim > 1:
        nonzero = np.any(np.abs(values) > _TOL, axis=tuple(range(values.ndim - 1)))
    else:
        nonzero = np.abs(values) > _TOL
    if not np.any(nonzero):
        return set()
    return {grade_of_blade(mv.layout.blades[i]) for i in np.where(nonzero)[0]}


def _scalar_square(mv: MVArray) -> float | None:
    try:
        dense = (
            mv if mv.storage_kind == "dense"
            else mv.to_layout(Algebra(mv.algebra).dense_layout())
        )
        return float((dense * dense).component(0))
    except Exception:
        return None


def _ninf_inner(alg: Algebra, mv: MVArray) -> float:
    try:
        return float(mv.inner(infinity(alg, backend=mv.storage_kind)).component(0))
    except Exception:
        return float("nan")


def _no_coefficient(alg: Algebra, mv: MVArray) -> float:
    n = alg.dimension - 2
    c4 = float(mv.component(1 << n))
    c5 = float(mv.component(1 << (n + 1)))
    return c5 - c4


def _ninf_coefficient(alg: Algebra, mv: MVArray) -> float:
    n = alg.dimension - 2
    c4 = float(mv.component(1 << n))
    c5 = float(mv.component(1 << (n + 1)))
    return (c4 + c5) / 2.0


def _is_zero_mv(mv: MVArray) -> bool:
    if mv.layout.size == 0:
        return True
    return bool(np.allclose(mv.values, 0.0, atol=_TOL))


def _any_blade_contains_conformal_axes(alg: Algebra, mv: MVArray) -> bool:
    n = alg.dimension - 2
    e4_bit = 1 << n
    e5_bit = 1 << (n + 1)
    for blade in mv.layout.blades:
        if blade == 0:
            continue
        if (blade & e4_bit) or (blade & e5_bit):
            return True
    return False


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


def extract_euclidean_vector(mv: MVArray) -> np.ndarray:
    """Return the Euclidean coordinates stored in the blade coefficients.

    Citation: In the AMSA CGA convention, Euclidean basis blade components
    ``e1..en`` directly carry the vector coordinates.  See Dorst, Fontijne,
    Mann (2007), *Geometric Algebra for Computer Science*, Table 13.1
    (conformal point representation).
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

    Citation: Dorst, Fontijne, Mann (2007), *Geometric Algebra for Computer
    Science*, Morgan Kaufmann, Table 13.1 (conformal point representation),
    and the inverse mapping in Perwass (2009), §4.3.2.
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

    Citation: Dorst, Fontijne, Mann (2007), *Geometric Algebra for Computer
    Science*, Morgan Kaufmann, Table 13.2 (dual sphere representation).
    """
    alg = Algebra(mv.algebra)
    n = _euclidean_dimension(alg)
    center = _extract_euclidean_blade_components(mv, n)
    sq = (mv * mv).component(0)
    radius = np.sqrt(np.abs(sq))
    return center, radius


def extract_plane(mv: MVArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (normal, signed_distance) from a dual-plane MV.

    The dual plane ``P = n + d n_inf`` stores the Euclidean normal ``n``
    in the ``e1..en`` coefficients and the signed distance ``d`` in the
    ``n_inf`` basis coefficient.

    Citation: Dorst, Fontijne, Mann (2007), *Geometric Algebra for Computer
    Science*, Morgan Kaufmann, Table 13.2 (dual plane representation).
    """
    alg = Algebra(mv.algebra)
    n_axes = _euclidean_dimension(alg)
    normal = _extract_euclidean_blade_components(mv, n_axes)
    n_inf_axis = n_axes
    distance = mv.component(1 << n_inf_axis)
    return normal, distance


__all__ = [
    "circle_through_points",
    "distance_squared",
    "euclidean_vector",
    "extract_euclidean_vector",
    "extract_plane",
    "extract_point",
    "extract_sphere",
    "infinity",
    "line_through_points",
    "origin",
    "plane",
    "point",
    "sphere",
    "translate",
]
