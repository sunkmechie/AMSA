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

from dataclasses import dataclass

import numpy as np

type ColorLike = str | tuple[float, float, float] | tuple[float, float, float, float]


@dataclass(kw_only=True)
class VizPrimitive:
    """Base class for neutral geometric primitives."""

    label: str | None = None
    color: ColorLike | None = None


@dataclass
class Point(VizPrimitive):
    """
    A point in space.

    position: array of shape (D,) or (..., D) representing coordinates.
    """

    position: np.ndarray


@dataclass
class Line(VizPrimitive):
    """
    An infinite line or a directed line segment depending on backend rendering.

    origin: array of shape (D,) representing a point on the line.
    direction: array of shape (D,) representing the line's direction vector.
    """

    origin: np.ndarray
    direction: np.ndarray


@dataclass
class LineSegments(VizPrimitive):
    """
    A collection of line segments or a continuous path.

    positions: array of shape (N, D) representing vertex coordinates.
    connect: 'segments' for independent (p1-p2, p3-p4) or 'strip' for (p1-p2-p3).
    """

    positions: np.ndarray
    connect: str = "segments"


@dataclass
class Plane(VizPrimitive):
    """
    A 2D plane in 3D (or D-dimensional) space.

    origin: array of shape (D,) representing a point on the plane.
    normal: array of shape (D,) representing the normal vector.
    """

    origin: np.ndarray
    normal: np.ndarray


@dataclass
class Rotor(VizPrimitive):
    """
    A backend-friendly visualization of a rigid transform frame.

    This is intentionally a derived visualization primitive, not a core AMSA rotor or
    motor representation. Keeping the linear map here is acceptable because it stays
    isolated inside the visualization layer.
    """

    origin: np.ndarray
    matrix: np.ndarray


@dataclass
class Circle(VizPrimitive):
    """
    A circle in 2D space.

    center: array of shape (2,) representing the center coordinates.
    radius: float representing the circle radius.
    """

    center: np.ndarray
    radius: float
