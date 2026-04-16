#/src/amsa/viz/adapters.py
from __future__ import annotations

import numpy as np

from amsa.mv import MVArray
from amsa.viz.primitives import (
    ColorLike,
    Line,
    LineSegments,
    Plane,
    Point,
    Rotor,
)


def to_point(
    mv: MVArray,
    *,
    color: ColorLike | None = None,
    label: str | None = None,
) -> Point:
    """
    Extract geometric point data from a multivector based on its algebra type.
    """
    signature = mv.algebra.signature

    if signature == (0, 1, 1):
        # PGA2D (0, 1, 1)
        # Point is represented as a bivector: x * e01 + y * e02 + w * e12
        try:
            w = mv.component("e12")
            x = mv.component("e01")
            y = mv.component("e02")

            # Handle division by zero robustly for ideal points (points at infinity)
            with np.errstate(divide="ignore", invalid="ignore"):
                px = np.where(w != 0, x / w, x)
                py = np.where(w != 0, y / w, y)

            position = np.stack([px, py], axis=-1)
            return Point(position=position, color=color, label=label)
        except KeyError as exc:
            raise ValueError(
                "Multivector layout does not contain the necessary basis blades "
                "(e01, e02, e12) to extract a PGA2D point."
            ) from exc

    if signature == (0, 1, 1, 1):
        # PGA3D (0, 1, 1, 1)
        # Point is represented as a trivector. Using canonical sorted names:
        # x is associated with -e023 (i.e. e032)
        # y is associated with e013
        # z is associated with -e012 (i.e. e021)
        # w is associated with e123
        try:
            w = mv.component("e123")
            x = -mv.component("e023")
            y = mv.component("e013")
            z = -mv.component("e012")

            with np.errstate(divide="ignore", invalid="ignore"):
                px = np.where(w != 0, x / w, x)
                py = np.where(w != 0, y / w, y)
                pz = np.where(w != 0, z / w, z)

            position = np.stack([px, py, pz], axis=-1)
            return Point(position=position, color=color, label=label)
        except KeyError as exc:
            raise ValueError(
                "Multivector layout does not contain the necessary basis blades "
                "to extract a PGA3D point."
            ) from exc
    
    raise NotImplementedError(
        f"to_point is not currently implemented for algebra signature {signature}"
    )


def to_plane(
    mv: MVArray,
    *,
    color: ColorLike | None = None,
    label: str | None = None,
) -> Plane:
    """
    Extract geometric plane data from a multivector based on its algebra type.
    """
    signature = mv.algebra.signature

    if signature == (0, 1, 1):
        # PGA2D (0, 1, 1): A line is a grade-1 vector: a*e1 + b*e2 + d*e0
        a = mv.component("e1")
        b = mv.component("e2")
        d = mv.component("e0")
        
        # In 2D, the plane is a line. Normal (a, b), Distance d.
        # We can represent it as origin + normal.
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_sq = a**2 + b**2
            origin = np.stack([-a * d / mag_sq, -b * d / mag_sq], axis=-1)
            normal = np.stack([a, b], axis=-1)
            
        return Plane(origin=origin, normal=normal, color=color, label=label)

    if signature == (0, 1, 1, 1):
        # PGA3D (0, 1, 1, 1): A plane is a grade-1 vector: a*e1 + b*e2 + c*e3 + d*e0
        a = mv.component("e1")
        b = mv.component("e2")
        c = mv.component("e3")
        d = mv.component("e0")
        
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_sq = a**2 + b**2 + c**2
            origin = np.stack([-a * d / mag_sq, -b * d / mag_sq, -c * d / mag_sq], axis=-1)
            normal = np.stack([a, b, c], axis=-1)
            
        return Plane(origin=origin, normal=normal, color=color, label=label)

    raise NotImplementedError(
        f"to_plane is not currently implemented for algebra signature {signature}"
    )


def to_line(
    mv: MVArray,
    *,
    color: ColorLike | None = None,
    label: str | None = None,
) -> Line:
    """
    Extract geometric line data from a multivector based on its algebra type.
    """
    signature = mv.algebra.signature

    if signature == (0, 1, 1, 1):
        # PGA3D (0, 1, 1, 1): A line is a grade-2 bivector.
        # Direction d from (e23, e31, e12), Moment m from (e01, e02, e03)
        # Standard: L = d_x e23 + d_y e31 + d_z e12 + m_x e01 + m_y e02 + m_z e03
        dx = mv.component("e23")
        dy = -mv.component("e13") # e31 = -e13
        dz = mv.component("e12")
        mx = -mv.component("e01") # Sign correction for dual moment
        my = -mv.component("e02")
        mz = -mv.component("e03")
        
        direction = np.stack([dx, dy, dz], axis=-1)
        moment = np.stack([mx, my, mz], axis=-1)
        
        # Origin as point on line closest to (0,0,0): p = (d x m) / |d|^2
        d_cross_m = np.cross(direction, moment)
        d_mag_sq = np.sum(direction**2, axis=-1, keepdims=True)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            origin = d_cross_m / d_mag_sq
            
        return Line(origin=origin, direction=direction, color=color, label=label)

    raise NotImplementedError(
        f"to_line is not currently implemented for algebra signature {signature}"
    )


def to_rotor(
    mv: MVArray,
    *,
    color: ColorLike | None = None,
    label: str | None = None,
) -> Rotor:
    """
    Extract rigid transform frame data from a motor/rotor.
    """
    signature = mv.algebra.signature
    
    if signature == (0, 1, 1, 1):
        # 1. Extract Origin (Point position)
        from amsa.algebra import Algebra
        alg = Algebra(mv.algebra)
        origin_pt = alg.multivector({"e123": 1.0})
        origin = to_point(mv.sandwich(origin_pt)).position
        
        # 2. Extract 3x3 Rotation Matrix
        # columns are M * ei * ~M
        e1 = alg.multivector({"e1": 1.0})
        e2 = alg.multivector({"e2": 1.0})
        e3 = alg.multivector({"e3": 1.0})
        
        # Extract individual components as the core API expects
        s1 = mv.sandwich(e1)
        r1 = np.stack([s1.component("e1"), s1.component("e2"), s1.component("e3")], axis=-1)
        
        s2 = mv.sandwich(e2)
        r2 = np.stack([s2.component("e1"), s2.component("e2"), s2.component("e3")], axis=-1)
        
        s3 = mv.sandwich(e3)
        r3 = np.stack([s3.component("e1"), s3.component("e2"), s3.component("e3")], axis=-1)
        
        # Matrix shape (..., 3, 3)
        matrix = np.stack([r1, r2, r3], axis=-1)
        
        return Rotor(origin=origin, matrix=matrix, color=color, label=label)

    raise NotImplementedError(
        f"to_rotor is not currently implemented for algebra signature {signature}"
    )


def to_line_segments(
    mv: MVArray,
    *,
    connect: str = "segments",
    color: ColorLike | None = None,
    label: str | None = None,
) -> LineSegments:
    """
    Extract vectorized line segment data from a multivector batch.
    """
    # Use to_point to get the raw vertex coordinates
    pts_primitive = to_point(mv)
    pos = pts_primitive.position # Shape (..., D)
    
    if connect == "segments":
        # Handle the common case of closing loops for polygons (e.g. triangles)
        # If shape is (..., S, D), we produce (..., S*2, D) segments
        if pos.ndim >= 2:
            s_dim = pos.shape[-2]
            if s_dim > 1:
                # p0-p1, p1-p2, ..., p(S-1)-p0
                idx = np.arange(s_dim)
                next_idx = (idx + 1) % s_dim
                # Interleave current and next indices
                interleaved = np.stack([idx, next_idx], axis=1).flatten()
                pos = pos[..., interleaved, :]
        
    return LineSegments(
        positions=pos,
        connect=connect,
        color=color,
        label=label
    )
