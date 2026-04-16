# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
VisPy backend for AMSA high-performance visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import vispy
    from vispy import scene
except ImportError as exc:
    raise ImportError(
        "vispy is required for amsa.viz.backends.vispy. "
        "Install it via `pip install 'amsa[viz]'`."
    ) from exc

from amsa.mv import MVArray
from amsa.viz.adapters import to_point
from amsa.viz.primitives import (
    Line,
    LineSegments,
    Plane,
    Point,
    Rotor,
    VizPrimitive,
)

if TYPE_CHECKING:
    from amsa.viz.core import Layer

def _prepare_vispy_pos(mv: MVArray | Point | np.ndarray) -> np.ndarray:
    """Prepare multivector, Point, or raw coordinate data for VisPy visuals."""
    if isinstance(mv, np.ndarray):
        pos = mv
    elif isinstance(mv, MVArray):
        pos = to_point(mv).position
    else:
        # Assume it's a primitive with a .position attribute (like Point)
        pos = mv.position
        
    # Ensure we are at least 2D (batch dimension)
    if pos.ndim == 1:
        pos = pos[np.newaxis, :]
        
    # Handle batch shapes (..., D) by flattening to (N, D)
    if pos.ndim > 2:
        pos = pos.reshape(-1, pos.shape[-1])
    
    # In 2D, expand to (x, y, 0) for VisPy's 3D coordinate system
    if pos.shape[-1] == 2:
        z = np.zeros((pos.shape[0], 1), dtype=pos.dtype)
        pos = np.concatenate([pos, z], axis=-1)
    return pos

def plot(view: scene.widgets.ViewBox, primitive: VizPrimitive, **kwargs: Any) -> scene.visuals.Visual:
    """
    Plots a viz primitive onto a VisPy ViewBox.
    """
    if isinstance(primitive, Point):
        pos = _prepare_vispy_pos(primitive)
        markers = scene.visuals.Markers()
        markers.set_data(
            pos,
            edge_color=kwargs.get("edge_color", None),
            face_color=kwargs.get("color", primitive.color or "white"),
            size=kwargs.get("size", 5),
        )
        markers.parent = view.scene
        return markers

    if isinstance(primitive, LineSegments):
        pos = _prepare_vispy_pos(primitive.positions)
        line = scene.visuals.Line(
            pos=pos,
            color=kwargs.get("color", primitive.color or "white"),
            width=kwargs.get("width", 2),
            connect=primitive.connect
        )
        line.parent = view.scene
        return line

    if isinstance(primitive, Line):
        # For infinite lines in 3D, draw a very long segment
        p = primitive.origin
        d = primitive.direction
        d_norm = d / np.linalg.norm(d, axis=-1, keepdims=True)
        p1 = p - d_norm * 1000
        p2 = p + d_norm * 1000
        pos = np.stack([p1, p2], axis=-2)
        
        line = scene.visuals.Line(
            pos=_prepare_vispy_pos(pos),
            color=kwargs.get("color", primitive.color or "white"),
            width=kwargs.get("width", 1)
        )
        line.parent = view.scene
        return line

    if isinstance(primitive, Rotor):
        # XYZAxis visual for frames
        axis = scene.visuals.XYZAxis(width=2)
        axis.parent = view.scene
        
        # We need to apply the transform (origin + matrix)
        # VisPy MatrixTransform takes a 4x4
        mat = np.eye(4)
        mat[:3, :3] = primitive.matrix.T # VisPy uses column-major/transposed convention usually
        mat[:3, 3] = primitive.origin
        axis.transform = scene.transforms.MatrixTransform(mat)
        return axis

    if isinstance(primitive, Plane):
        # We represent the infinite plane as a large finite plane for visualization
        # VisPy's Plane visual defaults to XY plane (normal z)
        plane = scene.visuals.Plane(width=100, height=100, color=kwargs.get("color", "blue"), alpha=0.3)
        plane.parent = view.scene
        
        # Calculate transform from XY to the target normal
        # origin + normal
        n = primitive.normal
        o = primitive.origin
        
        # Simple alignment: calculate rotation from (0,0,1) to n
        n_unit = n / np.linalg.norm(n, axis=-1, keepdims=True)
        z_axis = np.array([0, 0, 1])
        
        # Cross product and angle
        cross = np.cross(z_axis, n_unit)
        cross_norm = np.linalg.norm(cross)
        
        mat = np.eye(4)
        if cross_norm > 1e-6:
            # Rodrigues-like rotation
            angle = np.arccos(np.dot(z_axis, n_unit))
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            u, v, w = cross / cross_norm
            rot = np.array([
                [t*u*u + c,   t*u*v - s*w, t*u*w + s*v],
                [t*u*v + s*w, t*v*v + c,   t*v*w - s*u],
                [t*u*w - s*v, t*v*w + s*u, t*w*w + c]
            ])
            mat[:3, :3] = rot
        
        mat[:3, 3] = o
        plane.transform = scene.transforms.MatrixTransform(mat)
        return plane

    raise NotImplementedError(f"Plotting for {type(primitive)} is not implemented yet in VisPy.")

def update_layer(layer: Layer, mv: MVArray | np.ndarray) -> None:
    """
    Efficiently update a VisPy visual with new multivector data or raw positions.
    """
    if layer.primitive == Point:
        pos = _prepare_vispy_pos(mv)
        layer.artist.set_data(pos=pos)
    elif layer.primitive == LineSegments:
        from amsa.viz.adapters import to_line_segments
        # Extract and interleave segments via the adapter
        prim = to_line_segments(mv)
        pos = _prepare_vispy_pos(prim.positions)
        layer.artist.set_data(pos=pos)
    elif layer.primitive == Rotor:
        from amsa.viz.adapters import to_rotor
        prim = to_rotor(mv)
        mat = np.eye(4)
        mat[:3, :3] = prim.matrix.T
        mat[:3, 3] = prim.origin
        layer.artist.transform.matrix = mat
    elif layer.primitive == Plane:
        from amsa.viz.adapters import to_plane
        prim = to_plane(mv)
        # Re-calculate and update matrix as above...
        # (For brevity in update, we'll implement the matrix update logic here too)
        n = prim.normal
        o = prim.origin
        n_unit = n / np.linalg.norm(n, axis=-1, keepdims=True)
        z_axis = np.array([0, 0, 1])
        cross = np.cross(z_axis, n_unit)
        cross_norm = np.linalg.norm(cross)
        mat = np.eye(4)
        if cross_norm > 1e-6:
            angle = np.arccos(np.clip(np.dot(z_axis, n_unit), -1, 1))
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            u, v, w = cross / cross_norm
            rot = np.array([
                [t*u*u+c, t*u*v-s*w, t*u*w+s*v],
                [t*u*v+s*w, t*v*v+c, t*v*w-s*u],
                [t*u*w-s*v, t*v*w+s*u, t*w*w+c]
            ])
            mat[:3, :3] = rot
        mat[:3, 3] = o
        layer.artist.transform.matrix = mat
    elif layer.primitive == Line:
        from amsa.viz.adapters import to_line
        prim = to_line(mv)
        p = prim.origin
        d = prim.direction
        d_norm = d / np.linalg.norm(d, axis=-1, keepdims=True)
        p1 = p - d_norm * 1000
        p2 = p + d_norm * 1000
        pos = np.stack([p1, p2], axis=-2)
        layer.artist.set_data(pos=_prepare_vispy_pos(pos))

class AMSAScene:
    """
    High-level interactive scene for AMSA multivector visualization.
    """
    def __init__(self, title: str = "AMSA Visualization", keys: str = "interactive") -> None:
        self.canvas = scene.SceneCanvas(title=title, keys=keys, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "arcball" # Default to 3D orbit
        self.visuals: list[scene.visuals.Visual] = []

    def add(self, primitive: VizPrimitive, **kwargs: Any) -> Layer:
        """Add a viz primitive to the scene."""
        from amsa.viz.core import Layer
        visual = plot(self.view, primitive, **kwargs)
        self.visuals.append(visual)
        return Layer(artist=visual, primitive=type(primitive), backend="vispy", parent=self.view)

    def set_camera(self, kind: str = "turntable") -> None:
        """Set the camera type (e.g. 'turntable', 'arcball', 'panzoom')."""
        self.view.camera = kind

    def show(self) -> None:
        """Run the application."""
        import vispy.app
        vispy.app.run()

def show() -> None:
    """
    Global convenience wrapper to start the VisPy event loop.
    """
    import vispy.app
    vispy.app.run()
