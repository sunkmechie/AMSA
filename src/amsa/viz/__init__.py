# AMSA Visualization Adapters and Primitives
#
# This module provides:
# - Adapters: Convert multivectors to geometric data (points, lines, etc.)
# - Primitives: Geometric primitives for visualization (Point, Line, Circle, etc.)
#
# For visualization, import backends directly:
#   - For 2D plots: from amsa.viz.backends import mpl
#   - For 3D interactive: from amsa.viz.backends import vispy

from amsa.viz.adapters import (
    to_circle,
    to_line,
    to_line_segments,
    to_plane,
    to_point,
    to_rotor,
)
from amsa.viz.primitives import (
    Circle,
    Line,
    LineSegments,
    Plane,
    Point,
    Rotor,
    VizPrimitive,
)

__all__ = [
    # Adapters
    "to_circle",
    "to_line",
    "to_line_segments",
    "to_plane",
    "to_point",
    "to_rotor",
    # Primitives
    "Circle",
    "Line",
    "LineSegments",
    "Plane",
    "Point",
    "Rotor",
    "VizPrimitive",
]
