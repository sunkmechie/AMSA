# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

from amsa.mv import MVArray
from amsa.viz.adapters import to_point
from amsa.viz.primitives import Line, LineSegments, Plane, Point, Rotor, VizPrimitive

_logger = logging.getLogger(__name__)

type BackendKind = Literal["auto", "mpl", "vispy"]

@dataclass(frozen=True)
class Layer:
    """
    A handle to a backend-specific visualization layer.
    """
    artist: Any      # The backend artist (MPL Artist or VisPy Visual)
    primitive: type  # The type of primitive (Point, Line, etc.)
    backend: BackendKind
    parent: Any      # The Axes (MPL) or ViewBox (VisPy) containing the artist
    container: Any = None # The Scene (VisPy) or Figure (MPL)

# Internal state
_ACTIVE_BACKEND: BackendKind = "auto"

def use_backend(kind: BackendKind) -> None:
    """Manually set the visualization backend."""
    global _ACTIVE_BACKEND
    if kind not in ("auto", "mpl", "vispy"):
        raise ValueError(f"Unsupported backend: {kind}")
    _ACTIVE_BACKEND = kind

def _resolve_backend(mv_hint: Any = None) -> Literal["mpl", "vispy"]:
    """Select the best available backend based on environment and data size."""
    if _ACTIVE_BACKEND != "auto":
        return _ACTIVE_BACKEND  # type: ignore

    try:
        import vispy
        # Use VisPy for large batches (>100 elements)
        if isinstance(mv_hint, MVArray) and prod(mv_hint.batch_shape) > 100:
            return "vispy"
        # For a clean first impression, we default to MPL for small batches
        return "mpl"
    except ImportError:
        return "mpl"

def plot(primitive: VizPrimitive, **kwargs: Any) -> Layer:
    """
    Plot a geometric primitive using the active backend.
    
    Returns a Layer handle for updates.
    """
    parent = kwargs.pop("parent", kwargs.pop("view", kwargs.pop("ax", None)))
    backend = kwargs.pop("backend", None)
    
    # Auto-detect backend from parent if not specified
    if backend is None and parent is not None:
        if hasattr(parent, "scene"): # VisPy ViewBox has a .scene
            backend = "vispy"
        else:
            backend = "mpl"
    
    if backend is None:
        backend = _resolve_backend(mv_hint=primitive)
   
    if backend == "vispy":
        from amsa.viz.backends.vispy import plot as vispy_plot
        if parent is None:
            raise RuntimeError(
                "VisPy plot() requires a parent view. "
                "Use viz.view() to initialize a scene first."
            )
        artist = vispy_plot(parent, primitive, **kwargs)
        return Layer(
            artist=artist, 
            primitive=type(primitive), 
            backend="vispy", 
            parent=parent
        )
    else:
        import matplotlib.pyplot as plt
        from amsa.viz.backends.mpl import plot as mpl_plot
        
        ax = parent if parent is not None else plt.gca()
        artist = mpl_plot(ax, primitive, **kwargs)
        return Layer(
            artist=artist, 
            primitive=type(primitive), 
            backend="mpl", 
            parent=ax,
            container=ax.get_figure()
        )

def show() -> None:
    """
    Start the visualization event loop for the active backend.
    """
    backend = _resolve_backend()
    if backend == "vispy":
        from amsa.viz.backends.vispy import show as vispy_show
        vispy_show()
    else:
        import matplotlib.pyplot as plt
        plt.show()

def update(layer: Layer, mv: MVArray) -> None:
    """
    Update the data of an existing layer efficiently.
    """
    if layer.backend == "vispy":
        from amsa.viz.backends.vispy import update_layer
        update_layer(layer, mv)
    else:
        from amsa.viz.backends.mpl import update_layer
        update_layer(layer, mv)

def view(
    data: MVArray | VizPrimitive, 
    backend: BackendKind = "auto", 
    **kwargs: Any
) -> Layer:
    """
    High-level entry point to visualize a multivector array or a primitive.
    """
    resolved_backend = backend if backend != "auto" else _resolve_backend(mv_hint=data)
    
    # Make the backend sticky for show() and subsequent plots
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND == "auto":
        _ACTIVE_BACKEND = resolved_backend

    if isinstance(data, VizPrimitive):
        primitive = data
    else:
        # It's a multivector, apply adapter
        adapter = kwargs.pop("adapter", to_point)
        primitive = adapter(data, **kwargs)

    if resolved_backend == "vispy":
        from amsa.viz.backends.vispy import AMSAScene
        scene = AMSAScene(title=kwargs.get("title", "AMSA View"))
        layer = scene.add(primitive, **kwargs)
        # Update layer with container reference
        return Layer(
            artist=layer.artist, 
            primitive=layer.primitive, 
            backend=layer.backend, 
            parent=layer.parent, 
            container=scene
        )
    else:
        import matplotlib.pyplot as plt
        from amsa.viz.backends.mpl import plot as mpl_plot
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
        title = kwargs.pop("title", "AMSA View")
        
        # Determine 3D vs 2D
        is_3d = False
        if isinstance(primitive, Point):
             is_3d = (primitive.position.shape[-1] == 3)
        elif isinstance(primitive, (Line, LineSegments, Plane, Rotor)):
             is_3d = True
             
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
            
        artist = mpl_plot(ax, primitive, **kwargs)
        ax.set_title(title)
        return Layer(
            artist=artist, 
            primitive=type(primitive), 
            backend="mpl", 
            parent=ax, 
            container=fig
        )


class Scene:
    """
    A unified, backend-agnostic scene for interactive visualization.
    """
    def __init__(self, title: str = "AMSA Visualization", **kwargs: Any) -> None:
        self.backend = kwargs.pop("backend", _resolve_backend())
        if self.backend == "vispy":
            from amsa.viz.backends.vispy import AMSAScene
            self._impl = AMSAScene(title=title, **kwargs)
        else:
            # For MPL, a Scene is just a figure + axes
            import matplotlib.pyplot as plt
            self._fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            self._ax = self._fig.add_subplot(111, projection='3d' if kwargs.get("dim") == 3 else None)
            self._ax.set_title(title)
            self._impl = self._ax

    def add(self, primitive: VizPrimitive | MVArray, **kwargs: Any) -> Layer:
        """
        Add a primitive or multivector to the scene.
        """
        # Auto-convert multivector to point if needed
        if isinstance(primitive, MVArray):
            from amsa.viz.adapters import to_point
            primitive = to_point(primitive)

        if self.backend == "vispy":
            # This returns a Layer now as per my previous fix in vispy.py
            return self._impl.add(primitive, **kwargs)
        else:
            from amsa.viz.backends.mpl import plot as mpl_plot
            artist = mpl_plot(self._ax, primitive, **kwargs)
            return Layer(artist=artist, primitive=type(primitive), backend="mpl", parent=self._ax)

    def set_camera(self, **kwargs: Any) -> None:
        """Configure the scene camera."""
        if self.backend == "vispy":
            self._impl.set_camera(**kwargs)
        # MPL camera settings could be added here

    def show(self) -> None:
        """Show the scene and start the event loop."""
        if self.backend == "vispy":
            self._impl.show()
        else:
            import matplotlib.pyplot as plt
            plt.show()

    @property
    def canvas(self) -> Any:
        """Access the underlying backend canvas (e.g. for timers)."""
        if self.backend == "vispy":
            return self._impl.canvas
        return self._fig.canvas
