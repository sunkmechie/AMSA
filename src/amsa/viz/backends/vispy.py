from __future__ import annotations

from typing import Any

import numpy as np

from amsa.viz.primitives import Circle, Line, LineSegments, Plane, Point, Rotor, VizPrimitive

try:
    from vispy import app, scene  # type: ignore[import-untyped]
    from vispy.visuals.transforms import MatrixTransform  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when vispy is absent
    raise ModuleNotFoundError(
        "amsa.viz.backends.vispy requires vispy. Install AMSA with the `viz` extra."
    ) from exc


def show(*args: Any, **kwargs: Any) -> Any:
    return app.run(*args, **kwargs)


def plot(parent: Any, primitive: VizPrimitive, **kwargs: Any) -> Any:
    if isinstance(primitive, Point):
        return _plot_point(parent, primitive, **kwargs)
    if isinstance(primitive, Line):
        return _plot_line(parent, primitive, **kwargs)
    if isinstance(primitive, LineSegments):
        return _plot_line_segments(parent, primitive, **kwargs)
    if isinstance(primitive, Plane):
        return _plot_plane(parent, primitive, **kwargs)
    if isinstance(primitive, Circle):
        return _plot_circle(parent, primitive, **kwargs)
    if isinstance(primitive, Rotor):
        return _plot_rotor(parent, primitive, **kwargs)

    raise TypeError(f"Unsupported primitive type: {type(primitive)!r}")


def _effective_color(primitive: VizPrimitive, kwargs: dict[str, Any]) -> Any:
    color = kwargs.pop("color", None)
    return primitive.color if color is None else color


def _coerce_points(values: np.ndarray) -> np.ndarray:
    pts = np.asarray(values, dtype=float)
    if pts.ndim == 1:
        return pts.reshape(1, -1)
    if pts.ndim > 2:
        return pts.reshape(-1, pts.shape[-1])
    return pts


def _line_segment(origin: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return np.stack([origin, origin], axis=0)
    delta = direction / direction_norm * scale
    return np.stack([origin - delta, origin + delta], axis=0)


def _flatten_batches(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values.reshape(1, *values.shape)
    return values.reshape(-1, *values.shape[-2:])


def _plot_point(parent: Any, primitive: Point, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    label = kwargs.pop("label", primitive.label)
    size = kwargs.pop("size", 8)
    symbol = kwargs.pop("symbol", "o")
    coords = _coerce_points(primitive.position)
    markers = scene.visuals.Markers(parent=parent)
    markers.set_data(
        coords,
        face_color=color,
        edge_color=color,
        size=size,
        symbol=symbol,
        **kwargs,
    )
    if label is not None:
        markers.set_gl_state(depth_test=True)
    return markers


def _plot_line(parent: Any, primitive: Line, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    scale = float(kwargs.pop("scale", 1.0))
    connect = kwargs.pop("connect", "strip")
    width = kwargs.pop("width", 2.0)
    origins = np.asarray(primitive.origin, dtype=float)
    directions = np.asarray(primitive.direction, dtype=float)
    if origins.ndim == 1:
        origins = origins.reshape(1, -1)
        directions = directions.reshape(1, -1)

    visuals: list[Any] = []
    for origin, direction in zip(origins, directions, strict=True):
        segment = _line_segment(origin, direction, scale)
        visual = scene.visuals.Line(
            pos=segment,
            color=color,
            width=width,
            connect=connect,
            parent=parent,
        )
        visuals.append(visual)
    return visuals[0] if len(visuals) == 1 else visuals


def _plot_line_segments(parent: Any, primitive: LineSegments, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    connect = primitive.connect
    width = kwargs.pop("width", 2.0)
    positions = np.asarray(primitive.positions, dtype=float)
    batches = _flatten_batches(positions)
    visuals: list[Any] = []
    for batch in batches:
        visual = scene.visuals.Line(
            pos=batch,
            color=color,
            width=width,
            connect=connect,
            parent=parent,
            **kwargs,
        )
        visuals.append(visual)
    return visuals[0] if len(visuals) == 1 else visuals


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(unit)
    if norm == 0:
        raise ValueError("Plane normal cannot be zero.")
    unit = unit / norm

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


def _plot_plane(parent: Any, primitive: Plane, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    scale = float(kwargs.pop("scale", 1.0))
    origin = np.asarray(primitive.origin, dtype=float)
    normal = np.asarray(primitive.normal, dtype=float)

    if origin.shape[-1] == 2:
        tangent = np.array([normal[1], -normal[0]], dtype=float)
        return _plot_line(
            parent,
            Line(origin=origin, direction=tangent, color=color),
            scale=scale,
            **kwargs,
        )

    u, v = _plane_basis(normal)
    if np.allclose(v, 0.0):
        return _plot_line(
            parent,
            Line(origin=origin, direction=u, color=color),
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
    return scene.visuals.Line(
        pos=np.vstack([corners, corners[0]]),
        color=color,
        connect="strip",
        parent=parent,
    )


def _plot_circle(parent: Any, primitive: Circle, **kwargs: Any) -> Any:
    color = _effective_color(primitive, kwargs)
    center = np.asarray(primitive.center, dtype=float)
    if center.shape[-1] != 2:
        raise ValueError("VisPy circle plotting currently expects 2D centers.")

    segments = int(kwargs.pop("segments", 128))
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    points = np.column_stack(
        [
            center[0] + primitive.radius * np.cos(angles),
            center[1] + primitive.radius * np.sin(angles),
        ]
    )
    return scene.visuals.Line(
        pos=points,
        color=color,
        connect="strip",
        parent=parent,
        **kwargs,
    )


def _plot_rotor(parent: Any, primitive: Rotor, **kwargs: Any) -> Any:
    scale = float(kwargs.pop("scale", 1.0))
    origin = np.asarray(primitive.origin, dtype=float)
    matrix = np.asarray(primitive.matrix, dtype=float)
    if matrix.shape[-2:] != (origin.shape[-1], origin.shape[-1]):
        raise ValueError("Rotor matrix must be square and match the origin dimension.")

    axis = scene.visuals.XYZAxis(parent=parent, width=kwargs.pop("width", 2))
    transform = MatrixTransform()
    matrix4 = np.eye(4)
    dim = origin.shape[-1]
    matrix4[:dim, :dim] = matrix
    matrix4[:dim, 3] = origin
    matrix4[:dim, :dim] *= scale
    transform.matrix = matrix4
    axis.transform = transform
    return axis
