from __future__ import annotations

from typing import Any

import numpy as np

from amsa.viz.primitives import Circle, Line, LineSegments, Plane, Point, Rotor, VizPrimitive

# isort: off
try:
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle as MplCircle
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "amsa.viz.backends.mpl requires matplotlib. Install AMSA with the `viz` extra."
    ) from exc
# isort: on


def show(*args: Any, **kwargs: Any) -> Any:
    return plt.show(*args, **kwargs)


def plot(ax: Any, primitive: VizPrimitive, **kwargs: Any) -> Any:
    if isinstance(primitive, Point):
        return _plot_point(ax, primitive, **kwargs)
    if isinstance(primitive, Line):
        return _plot_line(ax, primitive, **kwargs)
    if isinstance(primitive, LineSegments):
        return _plot_line_segments(ax, primitive, **kwargs)
    if isinstance(primitive, Plane):
        return _plot_plane(ax, primitive, **kwargs)
    if isinstance(primitive, Circle):
        return _plot_circle(ax, primitive, **kwargs)
    if isinstance(primitive, Rotor):
        return _plot_rotor(ax, primitive, **kwargs)
    raise TypeError(f"Unsupported primitive type: {type(primitive)!r}")


def _effective_color(primitive: VizPrimitive, kwargs: dict[str, Any]) -> Any:
    color = kwargs.pop("color", None)
    return primitive.color if color is None else color


def _coerce_points(values: np.ndarray) -> np.ndarray:
    pts = np.asarray(values)
    if pts.ndim == 1:
        return pts.reshape(1, -1)
    if pts.ndim > 2:
        return pts.reshape(-1, pts.shape[-1])
    return pts


def _coerce_line_batch(
    origin: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin_arr = np.asarray(origin)
    direction_arr = np.asarray(direction)
    if origin_arr.ndim == 1:
        return origin_arr.reshape(1, -1), direction_arr.reshape(1, -1)
    return (
        origin_arr.reshape(-1, origin_arr.shape[-1]),
        direction_arr.reshape(-1, direction_arr.shape[-1]),
    )


def _line_segment(origin: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return np.stack([origin, origin], axis=0)
    delta = direction / direction_norm * scale
    return np.stack([origin - delta, origin + delta], axis=0)


def _draw_segment_collection(ax: Any, segments: np.ndarray, *, color: Any, **kwargs: Any) -> Any:
    segments_seq = segments.tolist()
    if segments.shape[-1] == 2:
        collection = LineCollection(segments_seq, colors=color, **kwargs)
        ax.add_collection(collection)
        ax.autoscale_view()
        return collection

    collection = Line3DCollection(segments_seq, colors=color, **kwargs)
    ax.add_collection3d(collection)
    ax.autoscale_view()
    return collection


def _plot_point(ax: Any, primitive: Point, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    size = kwargs.pop("s", kwargs.pop("size", 30))
    coords = _coerce_points(primitive.position)
    if coords.shape[-1] == 2:
        return ax.scatter(coords[:, 0], coords[:, 1], c=color, s=size, label=label, **kwargs)
    if coords.shape[-1] == 3:
        return ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=color,
            s=size,
            label=label,
            **kwargs,
        )
    raise ValueError("Point coordinates must have two or three dimensions.")


def _plot_line(ax: Any, primitive: Line, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    scale = float(kwargs.pop("scale", 1.0))
    origins, directions = _coerce_line_batch(primitive.origin, primitive.direction)
    artists: list[Any] = []
    for index, (origin, direction) in enumerate(zip(origins, directions, strict=True)):
        segment = _line_segment(origin, direction, scale)
        current_label = label if index == 0 else None
        if segment.shape[-1] == 2:
            (artist,) = ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                label=current_label,
                **kwargs,
            )
        else:
            (artist,) = ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=color,
                label=current_label,
                **kwargs,
            )
        artists.append(artist)
    return artists[0] if len(artists) == 1 else artists


def _plot_line_segments(ax: Any, primitive: LineSegments, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    positions = np.asarray(primitive.positions)
    if positions.ndim == 2:
        batches = positions.reshape(1, *positions.shape)
    else:
        batches = positions.reshape(-1, *positions.shape[-2:])

    artists: list[Any] = []
    for index, batch in enumerate(batches):
        current_label = label if index == 0 else None
        if primitive.connect == "strip":
            if batch.shape[-1] == 2:
                (artist,) = ax.plot(
                    batch[:, 0],
                    batch[:, 1],
                    color=color,
                    label=current_label,
                    **kwargs,
                )
            else:
                (artist,) = ax.plot(
                    batch[:, 0],
                    batch[:, 1],
                    batch[:, 2],
                    color=color,
                    label=current_label,
                    **kwargs,
                )
            artists.append(artist)
            continue

        if batch.shape[0] % 2 != 0:
            raise ValueError("Segments connectivity expects an even number of vertices.")

        paired = batch.reshape(-1, 2, batch.shape[-1])
        _draw_segment_collection(ax, paired, color=color, label=current_label, **kwargs)
        artists.append(None)

    return artists[0] if len(artists) == 1 else artists


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Plane normal cannot be zero.")
    unit = normal / normal_norm

    if unit.shape[-1] == 2:
        tangent = np.array([unit[1], -unit[0]])
        return tangent, np.zeros_like(tangent)

    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(unit, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    u = np.cross(unit, basis)
    u = u / np.linalg.norm(u)
    v = np.cross(unit, u)
    return u, v


def _plot_plane(ax: Any, primitive: Plane, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    scale = float(kwargs.pop("scale", 1.0))
    origin = np.asarray(primitive.origin, dtype=float)
    normal = np.asarray(primitive.normal, dtype=float)

    if origin.shape[-1] == 2:
        tangent = np.array([normal[1], -normal[0]], dtype=float)
        return _plot_line(
            ax,
            Line(origin=origin, direction=tangent, color=color, label=label),
            scale=scale,
            **kwargs,
        )

    u, v = _plane_basis(normal)
    if np.allclose(v, 0.0):
        return _plot_line(
            ax,
            Line(origin=origin, direction=u, color=color, label=label),
            scale=scale,
            **kwargs,
        )

    corners = np.array(
        [
            origin - scale * u - scale * v,
            origin + scale * u - scale * v,
            origin + scale * u + scale * v,
            origin - scale * u + scale * v,
        ]
    )
    poly = Poly3DCollection([corners], alpha=kwargs.pop("alpha", 0.15), facecolor=color, **kwargs)
    ax.add_collection3d(poly)
    outline = np.vstack([corners, corners[0]])
    (artist,) = ax.plot(
        outline[:, 0],
        outline[:, 1],
        outline[:, 2],
        color=color,
        label=label,
    )
    ax.autoscale_view()
    return artist


def _plot_circle(ax: Any, primitive: Circle, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    center = np.asarray(primitive.center, dtype=float)
    center_2d: tuple[float, float] = (float(center[0]), float(center[1]))
    patch = MplCircle(center_2d, primitive.radius, fill=False)
    patch.set_color(color)
    patch.set_label(label)
    patch.set(**kwargs)
    ax.add_patch(patch)
    return patch


def _plot_rotor(ax: Any, primitive: Rotor, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    scale = float(kwargs.pop("scale", 1.0))
    origin = np.asarray(primitive.origin, dtype=float)
    matrix = np.asarray(primitive.matrix, dtype=float)
    if matrix.shape[-2:] != (origin.shape[-1], origin.shape[-1]):
        raise ValueError("Rotor matrix must be square and match the origin dimension.")

    if origin.shape[-1] == 2:
        directions = [matrix[..., :, 0], matrix[..., :, 1]]
        colors = kwargs.pop("axis_colors", (color, color))
        artists: list[Any] = []
        for index, direction in enumerate(directions):
            artist = _plot_line(
                ax,
                Line(
                    origin=origin,
                    direction=direction,
                    color=colors[index],
                    label=label if index == 0 else None,
                ),
                scale=scale,
                **kwargs,
            )
            artists.append(artist)
        return artists

    artists_3d: list[Any] = []
    axis_colors = kwargs.pop("axis_colors", ("r", "g", "b"))
    for index in range(matrix.shape[-1]):
        direction = matrix[..., :, index]
        artist = _plot_line(
            ax,
            Line(
                origin=origin,
                direction=direction,
                color=axis_colors[index],
                label=label if index == 0 else None,
            ),
            scale=scale,
            **kwargs,
        )
        artists_3d.append(artist)
    return artists_3d
