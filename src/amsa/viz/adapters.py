#/src/amsa/viz/adapters.py
from __future__ import annotations

import numpy as np

from amsa.mv import MVArray
from amsa.viz.primitives import ColorLike, Point


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
