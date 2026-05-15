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

import math

import numpy as np

from amsa.algebra import Algebra
from amsa.mv import MVArray


def sphere_sphere(lhs: MVArray, rhs: MVArray) -> MVArray:
    """Return the direct circle where two CGA dual spheres meet."""
    from amsa.robo._validation import _validate_same_cga

    _validate_same_cga(lhs, rhs)
    return lhs.regress(rhs)


def line_plane(line: MVArray, plane: MVArray) -> MVArray:
    """Return the conformal point where a direct CGA line meets a dual plane."""
    from amsa.robo._validation import _validate_same_cga

    _validate_same_cga(line, plane)
    alg = Algebra(line.algebra)
    point_on_line, direction = _line_geometry(line)
    normal, distance = alg.extract_plane(plane)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-12:
        raise ValueError("CGA line and plane are parallel or coincident.")
    t = (float(distance) - float(np.dot(normal, point_on_line))) / denominator
    return alg.point(point_on_line + t * direction)


def point_circle_projection(point: MVArray, circle: MVArray) -> MVArray:
    """Project a conformal point onto a direct CGA circle."""
    from amsa.robo._validation import _validate_cga3d, _validate_same_cga

    _validate_same_cga(point, circle)
    alg = Algebra(point.algebra)
    _validate_cga3d(alg)

    point_coords = alg.extract_point(point)
    center, radius, normal = _circle_geometry(circle)
    radial = point_coords - center
    radial = radial - np.dot(radial, normal) * normal
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-12:
        radial = _perpendicular_unit(normal)
    else:
        radial = radial / radial_norm
    return alg.point(center + radius * radial)


def _circle_geometry(circle: MVArray) -> tuple[np.ndarray, float, np.ndarray]:
    from amsa.robo._validation import _validate_cga3d

    alg = Algebra(circle.algebra)
    _validate_cga3d(alg)
    ninf = alg.infinity()
    center_point = circle * ninf * circle
    center = alg.extract_point(center_point)

    scale_value = float(-(center_point.inner(ninf)).component(0))
    norm_value = float((circle * circle).component(0))
    if abs(scale_value) < 1e-12:
        raise ValueError("Cannot extract geometry from a degenerate CGA circle.")
    radius_sq = 2.0 * abs(norm_value) / abs(scale_value)
    radius = math.sqrt(max(radius_sq, 0.0))

    plane = (circle ^ ninf) * alg.inverse(alg.pseudoscalar(1.0))
    normal, _ = alg.extract_plane(plane)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("Cannot extract a support plane from a degenerate CGA circle.")
    return center, radius, normal / normal_norm


def _line_geometry(line: MVArray) -> tuple[np.ndarray, np.ndarray]:
    from amsa.robo._validation import _validate_cga3d

    alg = Algebra(line.algebra)
    _validate_cga3d(alg)
    direction = np.array([
        line.component("e145"),
        line.component("e245"),
        line.component("e345"),
    ], dtype=float)
    direction_norm_sq = float(np.dot(direction, direction))
    if direction_norm_sq < 1e-24:
        raise ValueError("Cannot extract geometry from a degenerate CGA line.")
    moment = np.array([
        0.5 * (line.component("e234") + line.component("e235")),
        -0.5 * (line.component("e134") + line.component("e135")),
        0.5 * (line.component("e124") + line.component("e125")),
    ], dtype=float)
    point = np.cross(direction, moment) / direction_norm_sq
    return point, direction


def _perpendicular_unit(normal: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    vector = axis - np.dot(axis, normal) * normal
    return np.asarray(vector / np.linalg.norm(vector), dtype=float)
