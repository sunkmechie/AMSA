from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(kw_only=True)
class VizPrimitive:
    """Base class for neutral geometric primitives."""
    label: str | None = None
    color: Any | None = None


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
    A rotation/transformation frame.
    
    origin: array of shape (D,) representing the translation part.
    matrix: array of shape (D, D) representing the linear transformation (rotation).
    """
    origin: np.ndarray
    matrix: np.ndarray
