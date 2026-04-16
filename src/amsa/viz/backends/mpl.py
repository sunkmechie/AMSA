from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for amsa.viz.backends.mpl. "
        "Install it via `pip install matplotlib`."
    ) from exc

import numpy as np

from amsa.mv import MVArray
from amsa.viz.adapters import to_point
from amsa.viz.primitives import Line, LineSegments, Plane, Point, Rotor, VizPrimitive

if TYPE_CHECKING:
    from amsa.viz.core import Layer

def _map_kwargs(kwargs: dict[str, Any], artist_type: str = "line") -> dict[str, Any]:
    """Map generic viz kwargs to Matplotlib-specific equivalents."""
    mapped = kwargs.copy()
    if "width" in mapped:
        mapped["linewidth"] = mapped.pop("width")
    if "marker_size" in mapped:
        if artist_type == "scatter":
            mapped["s"] = mapped.pop("marker_size")
        else:
            mapped["markersize"] = mapped.pop("marker_size")
    return mapped

def plot(ax: Axes, primitive: VizPrimitive, **kwargs: Any) -> Any:
    """
    Plots a viz primitive onto a matplotlib Axes. Returns the created artist.
    """
    color = kwargs.pop("color", primitive.color)
    label = kwargs.pop("label", primitive.label)

    if isinstance(primitive, Point):
        pos = primitive.position
        kw = _map_kwargs(kwargs, artist_type="scatter")
        if pos.shape[-1] == 2:
            return ax.scatter(pos[..., 0], pos[..., 1], color=color, label=label, **kw)
        elif pos.shape[-1] == 3:
            return ax.scatter(
                pos[..., 0], pos[..., 1], pos[..., 2], color=color, label=label, **kw
            )

    if isinstance(primitive, (Line, LineSegments)):
        kw = _map_kwargs(kwargs, artist_type="line")
        if isinstance(primitive, Line):
            p1 = primitive.origin
            p2 = primitive.origin + primitive.direction
            # Interleave p1, p2 for segments
            pts = np.stack([p1, p2], axis=-2)
        else:
            pts = primitive.positions
            
        if pts.shape[-1] == 2:
            # We can use LineCollection for better performance with segments
            from matplotlib.collections import LineCollection
            if pts.ndim == 2 and primitive.connect == "segments":
                 # Shape (N*2, 2) -> (N, 2, 2)
                 segments = pts.reshape(-1, 2, 2)
                 lc = LineCollection(segments, color=color, label=label, **kw)
                 ax.add_collection(lc)
                 return lc
            return ax.plot(pts[..., 0], pts[..., 1], color=color, label=label, **kw)[0]
        else:
            # 3D segments
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            if pts.ndim == 2 and (isinstance(primitive, Line) or primitive.connect == "segments"):
                 segments = pts.reshape(-1, 2, 3)
                 lc = Line3DCollection(segments, color=color, label=label, **kw)
                 ax.add_collection3d(lc)
                 return lc
            return ax.plot(pts[..., 0], pts[..., 1], pts[..., 2], color=color, label=label, **kw)[0]

    if isinstance(primitive, Rotor):
        # RGB axes frame
        origin = primitive.origin
        matrix = primitive.matrix
        kw = _map_kwargs(kwargs, artist_type="line")
        length = kw.pop("length", 1.0)
        
        # matrix is (..., 3, 3)
        if origin.ndim == 1:
            artists = []
            colors = ["r", "g", "b"]
            for i in range(3):
                a = ax.quiver(
                    origin[0], origin[1], origin[2],
                    matrix[0, i], matrix[1, i], matrix[2, i],
                    color=colors[i], length=length, **kw
                )
                artists.append(a)
            return artists # Handle multiple artists
        
    if isinstance(primitive, Plane):
        # Plot a finite grid
        o = primitive.origin
        n = primitive.normal
        # For simplicity, we just draw a small proxy around origin
        # (This is hard in MPL without complex triangulation)
        pass

    raise NotImplementedError(f"Plotting for {type(primitive)} is not implemented yet in mpl.")

def update_layer(layer: Layer, mv: MVArray) -> None:
    """
    Efficiently update a Matplotlib artist with new multivector data.
    """
    if layer.primitive == Point:
        pos = to_point(mv).position
        if pos.shape[-1] == 2:
            layer.artist.set_offsets(pos)
        else:
            layer.artist._offsets3d = (pos[..., 0], pos[..., 1], pos[..., 2])
    elif layer.primitive == LineSegments:
        from amsa.viz.adapters import to_line_segments
        prim = to_line_segments(mv)
        pts = prim.positions
        if pts.shape[-1] == 2:
            if hasattr(layer.artist, "set_segments"):
                layer.artist.set_segments(pts.reshape(-1, 2, 2))
            else:
                layer.artist.set_data(pts[..., 0], pts[..., 1])
        else:
            # 3D Line3DCollection update
            if hasattr(layer.artist, "set_segments"):
                 layer.artist.set_segments(pts.reshape(-1, 2, 3))
    elif layer.primitive == Rotor:
        from amsa.viz.adapters import to_rotor
        prim = to_rotor(mv)
        # Quiver updates are complex, might need to re-create or use low-level set_segments
        pass

def show() -> None:
    """Convenience wrapper for plt.show()"""
    plt.show()
