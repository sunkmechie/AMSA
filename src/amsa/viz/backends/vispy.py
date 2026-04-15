# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
VisPy backend for AMSA high-performance visualization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import vispy
    from vispy import scene
except ImportError as exc:
    raise ImportError(
        "vispy is required for amsa.viz.backends.vispy. "
        "Install it via `pip install 'amsa[viz]'`."
    ) from exc

from amsa.viz.primitives import VizPrimitive

from amsa.viz.primitives import Line, Point, VizPrimitive

def plot(view: scene.widgets.ViewBox, primitive: VizPrimitive, **kwargs: Any) -> scene.visuals.Visual:
    """
    Plots a viz primitive onto a VisPy ViewBox.
    """
    if isinstance(primitive, Point):
        pos = primitive.position
        
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

        markers = scene.visuals.Markers()
        markers.set_data(
            pos,
            edge_color=kwargs.get("edge_color", None),
            face_color=kwargs.get("color", primitive.color or "white"),
            size=kwargs.get("size", 5),
        )
        markers.parent = view.scene
        return markers

    if isinstance(primitive, Line):
        # For segments, VisPy expects a flat (N*2, D) array
        p1 = primitive.origin
        p2 = primitive.origin + primitive.direction
        pts = np.stack([p1, p2], axis=1) # Shape (..., 2, D)
        pos = pts.reshape(-1, pts.shape[-1])
            
        line = scene.visuals.Line(
            pos=pos,
            color=kwargs.get("color", primitive.color or "white"),
            width=kwargs.get("width", 2),
            connect="segments"
        )
        line.parent = view.scene
        return line

    raise NotImplementedError(f"Plotting for {type(primitive)} is not implemented yet in VisPy.")

class AMSAScene:
    """
    High-level interactive scene for AMSA multivector visualization.
    """
    def __init__(self, title: str = "AMSA Visualization", keys: str = "interactive") -> None:
        self.canvas = scene.SceneCanvas(title=title, keys=keys, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "arcball" # Default to 3D orbit
        self.visuals: list[scene.visuals.Visual] = []

    def add(self, primitive: VizPrimitive, **kwargs: Any) -> scene.visuals.Visual:
        """Add a viz primitive to the scene."""
        visual = plot(self.view, primitive, **kwargs)
        self.visuals.append(visual)
        return visual

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
