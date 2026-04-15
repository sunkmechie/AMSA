# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Visualization Suite

This module provides a unified, backend-agnostic interface for visualizing 
multivector arrays and geometric primitives.
"""

from amsa.viz.adapters import to_point
from amsa.viz.core import plot, show, use_backend, view
from amsa.viz.primitives import Line, Plane, Point, Rotor, VizPrimitive

__all__ = [
    "use_backend",
    "plot",
    "show",
    "view",
    "to_point",
    "Line",
    "Plane",
    "Point",
    "Rotor",
    "VizPrimitive",
]
