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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from amsa.algebra import Algebra
from amsa.mv import MVArray

if TYPE_CHECKING:
    pass

_TOL = 1e-10


# -- EntityInfo ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EntityInfo:
    """Describes the geometric interpretation of a multivector.

    ``classify()`` returns this for structured inspection and display — it
    never mutates the underlying multivector.
    """

    algebra: str
    kind: str
    representation: str = ""

    grades: tuple[int, ...] = ()
    null: bool = False
    normalized: bool = False

    invariants: dict[str, Any] = field(default_factory=dict)
    geometric_data: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def __str__(self) -> str:
        lines: list[str] = []
        title = f"{self.algebra.upper()} Classification"
        lines.append(title)
        lines.append("-" * len(title))
        lines.append(f"kind:           {self.kind}")
        if self.representation:
            lines.append(f"representation: {self.representation}")
        lines.append("")

        if self.ambiguous:
            lines.append("⚠  Ambiguous — multiple interpretations possible.")
            lines.append("")

        if self.warnings:
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        lines.append(f"grades:        {{{', '.join(str(g) for g in self.grades)}}}")
        lines.append(f"null:          {'yes' if self.null else 'no'}")
        lines.append(f"normalized:    {'yes' if self.normalized else 'no'}")
        lines.append("")

        if self.invariants:
            lines.append("invariants:")
            width = max(len(k) for k in self.invariants)
            for k, v in self.invariants.items():
                if isinstance(v, float):
                    lines.append(f"  {k:<{width}} = {v:.4g}")
                else:
                    lines.append(f"  {k:<{width}} = {v}")
            lines.append("")

        if self.geometric_data:
            lines.append("geometric data:")
            for k, v in self.geometric_data.items():
                if isinstance(v, np.ndarray):
                    lines.append(f"  {k}: {np.array_str(v, precision=4)}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        if self.storage:
            lines.append("storage:")
            for k, v in self.storage.items():
                lines.append(f"  {k:<12} {v}")
            lines.append("")

        return "\n".join(lines)


# -- general inspection helpers ------------------------------------------------


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


def _is_zero_mv(mv: MVArray) -> bool:
    if mv.layout.size == 0:
        return True
    return bool(np.allclose(mv.values, 0.0, atol=_TOL))


# -- CGA classification --------------------------------------------------------


def classify_cga(alg: Algebra, mv: MVArray) -> EntityInfo:
    """Classify a multivector in a CGA algebra.

    Returns an :class:`EntityInfo` with geometric interpretation, invariants,
    and storage metadata.  Never mutates the input.
    """
    from amsa.cga import (
        _euclidean_dimension,
        extract_plane,
        extract_point,
        extract_sphere,
        infinity,
    )

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
        ninf_inner = _cga_ninf_inner(alg, mv, infinity)
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

        no_c = _cga_no_coefficient(alg, mv)
        ninf_c = _cga_ninf_coefficient(alg, mv)
        has_ninf_in_support = _cga_any_blade_contains_conformal_axes(alg, mv)

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
                        kw["geometric_data"] = {"coordinates": extract_point(mv)}
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


def _cga_ninf_inner(alg: Algebra, mv: MVArray, infinity_fn: Any) -> float:
    try:
        return float(
            mv.inner(infinity_fn(alg, backend=mv.storage_kind)).component(0)
        )
    except Exception:
        return float("nan")


def _cga_no_coefficient(alg: Algebra, mv: MVArray) -> float:
    n = alg.dimension - 2
    c4 = float(mv.component(1 << n))
    c5 = float(mv.component(1 << (n + 1)))
    return c5 - c4


def _cga_ninf_coefficient(alg: Algebra, mv: MVArray) -> float:
    n = alg.dimension - 2
    c4 = float(mv.component(1 << n))
    c5 = float(mv.component(1 << (n + 1)))
    return (c4 + c5) / 2.0


def _cga_any_blade_contains_conformal_axes(alg: Algebra, mv: MVArray) -> bool:
    n = alg.dimension - 2
    e4_bit = 1 << n
    e5_bit = 1 << (n + 1)
    for blade in mv.layout.blades:
        if blade == 0:
            continue
        if (blade & e4_bit) or (blade & e5_bit):
            return True
    return False


# -- PGA classification --------------------------------------------------------


def classify_pga(alg: Algebra, mv: MVArray) -> EntityInfo:
    """Classify a multivector in a PGA algebra.

    Returns an :class:`EntityInfo` with geometric interpretation, invariants,
    and storage metadata.  Never mutates the input.
    """
    n = alg.dimension  # 3 for pga2d, 4 for pga3d
    is_2d = n == 3
    point_grade = 2 if is_2d else 3  # grade of a PGA point

    def _info(**overrides: Any) -> EntityInfo:
        kw: dict[str, Any] = {
            "algebra": f"pga{n - 1}d",
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
        kw["invariants"] = {"X²": float(sq) if sq is not None else float("nan")}

        if _is_zero_mv(mv):
            kw["kind"] = "zero multivector"
            return EntityInfo(**{**kw, **overrides})

        grades_set = set(kw["grades"])
        single_grade = len(grades_set) == 1
        has_e0 = _pga_any_blade_contains_e0(mv)
        e0_coeff = float(mv.component(1))

        if single_grade:
            if grades_set == {point_grade}:
                weight = _pga_point_weight(mv, is_2d)
                kw["invariants"]["weight"] = float(weight)
                kw["invariants"]["e₀ coefficient"] = e0_coeff

                normalized = bool(np.allclose(weight, 1.0, atol=_TOL))
                ideal = bool(np.allclose(weight, 0.0, atol=_TOL))

                if ideal:
                    kw["kind"] = "ideal point"
                elif normalized:
                    kw["kind"] = "normalized Euclidean point"
                else:
                    kw["kind"] = "Euclidean point"
                kw["representation"] = "direct"

                if not ideal:
                    try:
                        kw["geometric_data"] = {"coordinates": _pga_extract_point(mv, is_2d)}
                    except Exception:
                        kw["warnings"].append("could not extract point coordinates")
                else:
                    try:
                        kw["geometric_data"] = {"direction": _pga_extract_point(mv, is_2d)}
                    except Exception:
                        pass
                return EntityInfo(**{**kw, **overrides})

            if grades_set == {1}:
                if is_2d:
                    kw["kind"] = "line"
                else:
                    kw["kind"] = "plane"
                kw["representation"] = "dual"
                return EntityInfo(**{**kw, **overrides})

            if not is_2d and grades_set == {2}:
                kw["kind"] = "line"
                kw["representation"] = "direct"
                return EntityInfo(**{**kw, **overrides})

            kw["kind"] = "generic blade"
            return EntityInfo(**{**kw, **overrides})

        if is_2d:
            even_versor_grades = {0, 2}
            motor_grades = {0, 2}
        else:
            even_versor_grades = {0, 2, 4}
            motor_grades = {0, 2, 4}

        if grades_set == {0, 2}:
            if has_e0:
                kw["kind"] = "translator"
                kw["representation"] = "direct"
            else:
                kw["kind"] = "rotor"
                kw["representation"] = "direct"
            return EntityInfo(**{**kw, **overrides})

        if grades_set == motor_grades and 0 in grades_set:
            kw["kind"] = "motor"
            kw["representation"] = "direct"
            return EntityInfo(**{**kw, **overrides})

        if grades_set.issubset(even_versor_grades):
            kw["kind"] = "even multivector"
            return EntityInfo(**{**kw, **overrides})

        return EntityInfo(**{**kw, **overrides})

    return _info()


def _pga_any_blade_contains_e0(mv: MVArray) -> bool:
    e0_bit = 1
    for blade in mv.layout.blades:
        if blade == 0:
            continue
        if blade & e0_bit:
            return True
    return False


def _pga_point_weight(mv: MVArray, is_2d: bool) -> float:
    if is_2d:
        return float(mv.component(6))
    return float(mv.component(14))


def _pga_extract_point(mv: MVArray, is_2d: bool) -> np.ndarray:
    if is_2d:
        w = float(mv.component(6))
        x = float(mv.component(3))
        y = float(mv.component(5))
        return np.array([x, y]) / w if abs(w) > _TOL else np.array([x, y])
    else:
        w = float(mv.component(14))
        x = -float(mv.component(13))
        y = float(mv.component(11))
        z = -float(mv.component(7))
        with np.errstate(divide="ignore", invalid="ignore"):
            coords = np.array([x, y, z]) / w if abs(w) > _TOL else np.array([x, y, z])
        return coords


# -- VGA classification --------------------------------------------------------


def classify_vga(alg: Algebra, mv: MVArray) -> EntityInfo:
    """Classify a multivector in a VGA algebra.  (Stub — full pass 3.)"""
    rank = alg.dimension

    def _info(**overrides: Any) -> EntityInfo:
        kw: dict[str, Any] = {
            "algebra": f"vga{rank}d",
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
        kw["invariants"] = {"X²": float(sq) if sq is not None else float("nan")}

        if _is_zero_mv(mv):
            kw["kind"] = "zero multivector"
            return EntityInfo(**{**kw, **overrides})

        grades_set = set(kw["grades"])
        single_grade = len(grades_set) == 1
        only_scalar_bivector = grades_set.issubset({0, 2})

        if single_grade:
            grade = next(iter(grades_set))
            names = {0: "scalar", 1: "vector", 2: "bivector"}
            if rank > 2:
                names[3] = "trivector"
            if grade == rank and rank >= 3:
                kw["kind"] = "pseudoscalar"
            else:
                kw["kind"] = names.get(grade, "generic blade")
            return EntityInfo(**{**kw, **overrides})

        if only_scalar_bivector:
            kw["kind"] = "even versor"
            return EntityInfo(**{**kw, **overrides})

        if grades_set.issubset({0, 2, 4}):
            kw["kind"] = "even multivector"
            return EntityInfo(**{**kw, **overrides})

        return EntityInfo(**{**kw, **overrides})

    return _info()
