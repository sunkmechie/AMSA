# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from amsa.algebra import Algebra
from amsa.mv import MVArray
from amsa.ops import scale

EXPERIMENTAL_WARNING = (
    "amsa.robo is experimental and not ready for production robotics use. "
    "APIs and file formats may change before amsa-robo is split out."
)


@dataclass(frozen=True, slots=True)
class Link:
    name: str


@dataclass(frozen=True, slots=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RobotModel:
    name: str
    links: tuple[Link, ...] = ()
    joints: tuple[Joint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _triple(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    values = tuple(float(part) for part in text.split())
    if len(values) != 3:
        raise ValueError("Expected three floating point values.")
    return values


def importurdf(path: str | Path) -> RobotModel:
    """Import URDF topology into the experimental robotics model.

    This is intentionally a topology/metadata bridge, not AMSA's native robot
    representation.
    """
    root = ET.parse(path).getroot()
    links = tuple(Link(node.attrib["name"]) for node in root.findall("link"))
    joints: list[Joint] = []
    for node in root.findall("joint"):
        origin = node.find("origin")
        axis = node.find("axis")
        parent = node.find("parent")
        child = node.find("child")
        if parent is None or child is None:
            joint_name = node.attrib.get("name", "<unnamed>")
            raise ValueError(f"URDF joint {joint_name} is missing parent/child.")
        joints.append(
            Joint(
                name=node.attrib["name"],
                kind=node.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                axis=_triple(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                origin_xyz=_triple(
                    origin.attrib.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
                origin_rpy=_triple(
                    origin.attrib.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
            )
        )
    return RobotModel(
        name=root.attrib.get("name", Path(path).stem),
        links=links,
        joints=tuple(joints),
    )


def load_crobot(path: str | Path) -> RobotModel:
    """Load AMSA's experimental Clifford-native robot JSON format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RobotModel(
        name=data["name"],
        links=tuple(Link(item["name"]) for item in data.get("links", [])),
        joints=tuple(
            Joint(
                name=item["name"],
                kind=item["kind"],
                parent=item["parent"],
                child=item["child"],
                axis=tuple(item.get("axis", (0.0, 0.0, 1.0))),
                origin_xyz=tuple(item.get("origin_xyz", (0.0, 0.0, 0.0))),
                origin_rpy=tuple(item.get("origin_rpy", (0.0, 0.0, 0.0))),
            )
            for item in data.get("joints", [])
        ),
        metadata=dict(data.get("metadata", {})),
    )


def dump_crobot(model: RobotModel) -> dict[str, Any]:
    """Return a serializable draft `.crobot` document."""
    return {
        "format": "amsa-crobot",
        "version": 0,
        "name": model.name,
        "model": "cga3d",
        "links": [{"name": link.name} for link in model.links],
        "joints": [
            {
                "name": joint.name,
                "kind": joint.kind,
                "parent": joint.parent,
                "child": joint.child,
                "axis": joint.axis,
                "origin_xyz": joint.origin_xyz,
                "origin_rpy": joint.origin_rpy,
                "motion": "bivector-generator",
            }
            for joint in model.joints
        ],
        "metadata": dict(model.metadata),
    }


def ik(*args: Any, **kwargs: Any) -> Any:
    """Experimental inverse-kinematics namespace entry point."""
    if kwargs.get("solver") == "planar_two_link" and len(args) >= 2:
        return planar_two_link_ik(args[0], args[1], elbow=kwargs.get("elbow", "up"))
    raise NotImplementedError(
        "amsa.robo.ik is experimental; currently use solver='planar_two_link'."
    )


def planar_two_link_ik(
    link_lengths: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    elbow: str = "up",
) -> tuple[float, float]:
    """Minimal analytic 2R IK helper for smoke-testing the robotics namespace."""
    l1, l2 = link_lengths
    x, y = target_xy
    cos_q2 = np.clip((x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    sin_q2 = math.sqrt(max(0.0, 1.0 - float(cos_q2 * cos_q2)))
    if elbow == "down":
        sin_q2 = -sin_q2
    q2 = math.atan2(sin_q2, float(cos_q2))
    q1 = math.atan2(y, x) - math.atan2(l2 * sin_q2, l1 + l2 * float(cos_q2))
    return q1, q2


def fk(
    alg: Algebra,
    dh_params: list[tuple[float, float, float, float]],
    *,
    joint_types: list[str] | None = None,
) -> list[dict[str, object]]:
    """CGA forward kinematics via Denavit–Hartenberg motor composition.

    Each joint-link pair is defined by four DH parameters ``(α, a, d, θ)``
    and composed as the product of four motors:

        M_i = M_{i-1} · T_z(d) · R_z(θ) · T_x(a) · R_x(α)

    where ``T_{axis}(distance)`` translates along the axis and
    ``R_{axis}(angle)`` rotates about the axis.  For revolute joints
    *θ* is the variable; for prismatic joints *d* is the variable.

    Returns a list of dictionaries, one per link:

    .. code-block:: python

        {
            "motor":       MVArray,        # CGA motor (full pose)
            "position":    np.ndarray,      # (x, y, z)
            "orientation": np.ndarray,      # quaternion (w, x, y, z)
        }

    This formulation handles arbitrary N‑DOF serial chains including
    non‑planar arms with twisted joint axes (*α* ≠ 0), unlike matrix‑based
    Jacobian methods that require per‑configuration derivatives.

    Citation: Bayro‑Corrochano and Zamora‑Esquivel (2007), "Differential
    and inverse kinematics of robot devices using conformal geometric
    algebra", Robotica 25(1), pp. 43–61 — motor‑based DH parameterization
    for CGA serial chains (§3.1, eqs. 15–20).

    See also: Dorst, Fontijne, Mann (2007), *Geometric Algebra for
    Computer Science*, Morgan Kaufmann, §15.5 (versors for Euclidean
    motion).
    """
    n = len(dh_params)
    if joint_types is None:
        joint_types = ["revolute"] * n

    motor: MVArray = alg.scalar(1.0)
    results: list[dict[str, object]] = []

    for i in range(n):
        alpha, a, d, theta = dh_params[i]

        if joint_types[i] == "prismatic":
            T_z = alg.translate([0.0, 0.0, float(d)])
            R_z = alg.scalar(1.0)
        else:
            T_z = alg.translate([0.0, 0.0, float(d)])
            R_z = _rotor_axis(alg, theta, "z")

        T_x = alg.translate([float(a), 0.0, 0.0])
        R_x = _rotor_axis(alg, alpha, "x") if abs(alpha) > 1e-15 else alg.scalar(1.0)

        motor = motor * T_z * R_z * T_x * R_x
        pos = motor_to_position(motor, alg)
        quat = motor_to_quaternion(motor, alg)
        results.append({
            "motor": motor,
            "position": pos,
            "orientation": quat,
        })

    return results


def _rotor_axis(alg: Algebra, angle: float, axis: str) -> MVArray:
    """Return a CGA rotor for rotation by ``angle`` about ``axis``."""
    half_angle = -0.5 * angle
    if axis == "z":
        B = alg.blade("e1") ^ alg.blade("e2")
    elif axis == "x":
        B = alg.blade("e2") ^ alg.blade("e3")
    elif axis == "y":
        B = alg.blade("e3") ^ alg.blade("e1")
    else:
        raise ValueError(f"Unknown axis '{axis}'. Use 'x', 'y', or 'z'.")
    return alg.exp(scale(B, half_angle))


def _sandwich(motor: MVArray, target: MVArray) -> MVArray:
    from amsa.ops import sandwich as _op_sandwich

    return _op_sandwich(motor, target)


def motor_to_quaternion(motor: MVArray, alg: Algebra) -> np.ndarray:
    """Extract the orientation quaternion (w, x, y, z) from a CGA motor.

    Applies the motor to the canonical Euclidean basis vectors to recover
    the rotation matrix, then converts to a unit quaternion.

    Citation: Perwass (2009), *Geometric Algebra with Applications in
    Engineering*, Springer, §4.3 (versor-to-matrix decomposition).
    """
    R = motor_to_matrix(motor, alg)
    return _matrix_to_quaternion(R)


def motor_to_matrix(motor: MVArray, alg: Algebra) -> np.ndarray:
    """Extract the 3×3 rotation matrix from a CGA motor."""
    e1 = _sandwich(motor, alg.euclidean_vector([1.0, 0.0, 0.0]))
    e2 = _sandwich(motor, alg.euclidean_vector([0.0, 1.0, 0.0]))
    e3 = _sandwich(motor, alg.euclidean_vector([0.0, 0.0, 1.0]))
    r1 = alg.extract_euclidean_vector(e1)
    r2 = alg.extract_euclidean_vector(e2)
    r3 = alg.extract_euclidean_vector(e3)
    return np.column_stack((r1, r2, r3))


def motor_to_position(motor: MVArray, alg: Algebra) -> np.ndarray:
    """Extract the translational position (x, y, z) from a CGA motor."""
    origin = alg.origin()
    tip = _sandwich(motor, origin)
    return alg.extract_point(tip)


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses the numerically stable trace-based method from
    Shepperd (1978), "Quaternion from Rotation Matrix", JGCD 1(3).
    """
    t = float(np.trace(R))
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


__all__ = [
    "EXPERIMENTAL_WARNING",
    "Joint",
    "Link",
    "RobotModel",
    "dump_crobot",
    "fk",
    "ik",
    "importurdf",
    "load_crobot",
    "motor_to_matrix",
    "motor_to_position",
    "motor_to_quaternion",
    "planar_two_link_ik",
]
