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
    child_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    child_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion: dict[str, Any] = field(default_factory=dict)


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
                child_offset_xyz=tuple(item.get("child_offset_xyz", (0.0, 0.0, 0.0))),
                child_offset_rpy=tuple(item.get("child_offset_rpy", (0.0, 0.0, 0.0))),
                motion=_load_joint_motion(item),
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
                "child_offset_xyz": joint.child_offset_xyz,
                "child_offset_rpy": joint.child_offset_rpy,
                "motion": _joint_motion(joint),
            }
            for joint in model.joints
        ],
        "metadata": dict(model.metadata),
    }


def _load_joint_motion(item: dict[str, Any]) -> dict[str, Any]:
    motion = item.get("motion")
    if isinstance(motion, dict):
        return dict(motion)
    return _default_joint_motion(item.get("kind", "fixed"), item.get("axis", (0.0, 0.0, 1.0)))


def _joint_motion(joint: Joint) -> dict[str, Any]:
    if joint.motion:
        return dict(joint.motion)
    return _default_joint_motion(joint.kind, joint.axis)


def _default_joint_motion(kind: str, axis: Any) -> dict[str, Any]:
    axis_tuple = tuple(float(item) for item in axis)
    if kind == "fixed":
        return {
            "model": "cga3d",
            "kind": "fixed",
            "parameter": "none",
            "generator": {"kind": "identity"},
        }
    if kind == "prismatic":
        return {
            "model": "cga3d",
            "kind": "prismatic",
            "parameter": "distance",
            "generator": {
                "kind": "translation-direction",
                "axis": axis_tuple,
            },
        }
    if kind == "revolute":
        return {
            "model": "cga3d",
            "kind": "revolute",
            "parameter": "angle",
            "generator": {
                "kind": "rotation-axis",
                "axis": axis_tuple,
            },
        }
    return {
        "model": "cga3d",
        "kind": kind,
        "parameter": "unknown",
        "generator": {"kind": "unsupported", "axis": axis_tuple},
    }


def model_from_dh(
    dh_params: list[tuple[float, float, float, float]],
    *,
    joint_types: list[str] | None = None,
    name: str = "dh_chain",
) -> RobotModel:
    """Build an executable draft ``RobotModel`` from DH parameters.

    The resulting model stores each variable joint as a CGA generator axis plus
    fixed child offsets.  DH is treated as an adapter shape, not as the native
    execution representation.
    """
    n = len(dh_params)
    joint_types = _validate_joint_types(n, joint_types)
    links = [Link("base")]
    joints: list[Joint] = []
    for i, ((alpha, a, d, theta), kind) in enumerate(zip(dh_params, joint_types, strict=True)):
        parent = links[-1].name
        child = f"link_{i + 1}"
        links.append(Link(child))
        joints.append(
            Joint(
                name=f"joint_{i + 1}",
                kind=kind,
                parent=parent,
                child=child,
                axis=(0.0, 0.0, 1.0),
                origin_xyz=(0.0, 0.0, float(d)),
                origin_rpy=(0.0, 0.0, float(theta) if kind == "prismatic" else 0.0),
                child_offset_xyz=(float(a), 0.0, 0.0),
                child_offset_rpy=(float(alpha), 0.0, 0.0),
                motion=_default_joint_motion(kind, (0.0, 0.0, 1.0)),
            )
        )
    return RobotModel(
        name=name,
        links=tuple(links),
        joints=tuple(joints),
        metadata={"source": "dh", "format": "amsa-crobot-draft"},
    )


def ik(*args: Any, **kwargs: Any) -> Any:
    """Experimental inverse-kinematics namespace entry point."""
    solver = kwargs.get("solver")
    if solver == "planar_two_link" and len(args) >= 2:
        return planar_two_link_ik(args[0], args[1], elbow=kwargs.get("elbow", "up"))
    if solver in {"cga_spherical_wrist", "cga_full_chain"} and len(args) >= 3:
        return ik_cga_spherical_wrist(
            args[0],
            args[1],
            args[2],
            joint_types=kwargs.get("joint_types"),
            initial_angles=kwargs.get("initial_angles"),
            max_iterations=kwargs.get("max_iterations", 100),
            position_tolerance=kwargs.get("position_tolerance", 1e-6),
            orientation_tolerance=kwargs.get("orientation_tolerance", 1e-6),
            damping=kwargs.get("damping", 0.1),
            joint_limits=kwargs.get("joint_limits"),
        )
    if solver == "cga_sphere_sphere" and len(args) >= 2:
        return sphere_sphere(args[0], args[1])
    if solver == "cga_point_circle" and len(args) >= 2:
        return point_circle_projection(args[0], args[1])
    raise NotImplementedError(
        "amsa.robo.ik is experimental; supported solvers are "
        "'planar_two_link', 'cga_spherical_wrist', 'cga_full_chain', "
        "'cga_sphere_sphere', and 'cga_point_circle'."
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


def sphere_sphere(lhs: MVArray, rhs: MVArray) -> MVArray:
    """Return the direct circle where two CGA dual spheres meet."""
    _validate_same_cga(lhs, rhs)
    return lhs.regress(rhs)


def line_plane(line: MVArray, plane: MVArray) -> MVArray:
    """Return the conformal point where a direct CGA line meets a dual plane."""
    _validate_same_cga(line, plane)
    alg = Algebra(line.algebra)
    point_on_line, direction = _line_geometry(line)
    normal, distance = alg.extract_plane(plane)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-12:
        raise ValueError("CGA line and plane are parallel or coincident.")
    t = (float(distance) - float(np.dot(normal, point_on_line))) / denominator
    return alg.point(point_on_line + t * direction)


def point_circle_projection(point: MVArray, circle: MVArray) -> MVArray:
    """Project a conformal point onto a direct CGA circle."""
    _validate_same_cga(point, circle)
    alg = Algebra(point.algebra)
    _validate_cga3d(alg)

    point_coords = alg.extract_point(point)
    center, radius, normal = _circle_geometry(circle)
    radial = point_coords - center
    radial = radial - np.dot(radial, normal) * normal
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-12:
        radial = _perpendicular_unit(normal)
    else:
        radial = radial / radial_norm
    return alg.point(center + radius * radial)


def _circle_geometry(circle: MVArray) -> tuple[np.ndarray, float, np.ndarray]:
    alg = Algebra(circle.algebra)
    _validate_cga3d(alg)
    ninf = alg.infinity()
    center_point = circle * ninf * circle
    center = alg.extract_point(center_point)

    scale_value = float(-(center_point.inner(ninf)).component(0))
    norm_value = float((circle * circle).component(0))
    if abs(scale_value) < 1e-12:
        raise ValueError("Cannot extract geometry from a degenerate CGA circle.")
    radius_sq = 2.0 * abs(norm_value) / abs(scale_value)
    radius = math.sqrt(max(radius_sq, 0.0))

    plane = (circle ^ ninf) * alg.inverse(alg.pseudoscalar(1.0))
    normal, _ = alg.extract_plane(plane)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("Cannot extract a support plane from a degenerate CGA circle.")
    return center, radius, normal / normal_norm


def _line_geometry(line: MVArray) -> tuple[np.ndarray, np.ndarray]:
    alg = Algebra(line.algebra)
    _validate_cga3d(alg)
    direction = np.array([
        line.component("e145"),
        line.component("e245"),
        line.component("e345"),
    ], dtype=float)
    direction_norm_sq = float(np.dot(direction, direction))
    if direction_norm_sq < 1e-24:
        raise ValueError("Cannot extract geometry from a degenerate CGA line.")
    moment = np.array([
        0.5 * (line.component("e234") + line.component("e235")),
        -0.5 * (line.component("e134") + line.component("e135")),
        0.5 * (line.component("e124") + line.component("e125")),
    ], dtype=float)
    point = np.cross(direction, moment) / direction_norm_sq
    return point, direction


def _perpendicular_unit(normal: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    vector = axis - np.dot(axis, normal) * normal
    return np.asarray(vector / np.linalg.norm(vector), dtype=float)


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

    This formulation supports arbitrary-length serial DH chains, including
    non-planar arms with twisted joint axes (*α* != 0).  It does not yet model
    arbitrary robot graphs.  See ``docs/references.rst#robotics``.
    """
    _validate_cga3d(alg)
    n = len(dh_params)
    joint_types = _validate_joint_types(n, joint_types)

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


def fk_model(
    alg: Algebra,
    model: RobotModel,
    joint_values: np.ndarray | list[float] | tuple[float, ...],
) -> list[dict[str, object]]:
    """Forward kinematics for a draft Clifford-native serial ``RobotModel``.

    Each joint is executed as fixed parent attachment motor, variable joint
    motor, then fixed child attachment motor.  The joint motion itself is
    generated from the joint axis metadata.
    """
    _validate_cga3d(alg)
    values = np.asarray(joint_values, dtype=float)
    if values.shape != (len(model.joints),):
        raise ValueError(f"Expected {len(model.joints)} joint values, got shape {values.shape}.")

    motor: MVArray = alg.scalar(1.0)
    results: list[dict[str, object]] = []
    for joint, value in zip(model.joints, values, strict=True):
        motor = (
            motor
            * _fixed_pose_motor(alg, joint.origin_xyz, joint.origin_rpy)
            * joint_motion_motor(alg, joint, float(value))
            * _fixed_pose_motor(alg, joint.child_offset_xyz, joint.child_offset_rpy)
        )
        results.append({
            "joint": joint.name,
            "link": joint.child,
            "motor": motor,
            "position": motor_to_position(motor, alg),
            "orientation": motor_to_quaternion(motor, alg),
        })
    return results


def joint_motion_motor(alg: Algebra, joint: Joint, value: float) -> MVArray:
    """Return the variable CGA motor for one draft robot joint."""
    _validate_cga3d(alg)
    motion = _joint_motion(joint)
    motion_kind = str(motion.get("kind", joint.kind))
    generator = motion.get("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("Joint motion generator must be a mapping.")

    if motion_kind == "fixed":
        return alg.scalar(1.0)
    axis = _unit_axis(tuple(generator.get("axis", joint.axis)))
    if motion_kind == "prismatic":
        return alg.translate(axis * float(value))
    if motion_kind == "revolute":
        return _rotor_about_axis(alg, axis, float(value))
    raise ValueError(f"Unsupported joint motion kind: {motion_kind!r}.")


def _fixed_pose_motor(
    alg: Algebra,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> MVArray:
    roll, pitch, yaw = rpy
    motor: MVArray = alg.translate(xyz)
    if abs(roll) > 1e-15:
        motor = motor * _rotor_axis(alg, float(roll), "x")
    if abs(pitch) > 1e-15:
        motor = motor * _rotor_axis(alg, float(pitch), "y")
    if abs(yaw) > 1e-15:
        motor = motor * _rotor_axis(alg, float(yaw), "z")
    return motor


def _unit_axis(axis: tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3D joint axis, got shape {vector.shape}.")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Joint axis must be non-zero.")
    return vector / norm


def _rotor_about_axis(alg: Algebra, axis: np.ndarray, angle: float) -> MVArray:
    e1 = alg.blade("e1")
    e2 = alg.blade("e2")
    e3 = alg.blade("e3")
    bivector = (
        scale(e2 ^ e3, float(axis[0]))
        + scale(e3 ^ e1, float(axis[1]))
        + scale(e1 ^ e2, float(axis[2]))
    )
    return alg.exp(scale(bivector, -0.5 * angle))


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
    an interop rotation matrix, then converts to a unit quaternion.
    See ``docs/references.rst#robotics``.
    """
    _validate_cga3d(alg)
    _validate_motor_algebra(motor, alg)
    R = motor_to_matrix(motor, alg)
    return _matrix_to_quaternion(R)


def motor_to_matrix(motor: MVArray, alg: Algebra) -> np.ndarray:
    """Extract a 3×3 rotation matrix from a CGA motor for interop/reporting."""
    _validate_cga3d(alg)
    _validate_motor_algebra(motor, alg)
    e1 = _sandwich(motor, alg.euclidean_vector([1.0, 0.0, 0.0]))
    e2 = _sandwich(motor, alg.euclidean_vector([0.0, 1.0, 0.0]))
    e3 = _sandwich(motor, alg.euclidean_vector([0.0, 0.0, 1.0]))
    r1 = alg.extract_euclidean_vector(e1)
    r2 = alg.extract_euclidean_vector(e2)
    r3 = alg.extract_euclidean_vector(e3)
    return np.column_stack((r1, r2, r3))


def motor_to_position(motor: MVArray, alg: Algebra) -> np.ndarray:
    """Extract the translational position (x, y, z) from a CGA motor."""
    _validate_cga3d(alg)
    _validate_motor_algebra(motor, alg)
    origin = alg.origin()
    tip = _sandwich(motor, origin)
    return alg.extract_point(tip)


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses a numerically stable trace-based method.  See
    ``docs/references.rst#robotics``.
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


# ---------------------------------------------------------------------------
# Numerical Inverse Kinematics (DLS)
# ---------------------------------------------------------------------------


@dataclass
class IKResult:
    """Result of a numerical inverse-kinematics solve.

    Attributes
    ----------
    success : bool
        Whether the solver converged within tolerances.
    joint_angles : np.ndarray
        The solved joint angles (radians for revolute, metres for prismatic).
    motor : MVArray or None
        The CGA motor at the solved configuration (end-effector pose).
    position : np.ndarray or None
        The achieved end-effector position (x, y, z) in metres.
    orientation : np.ndarray or None
        The achieved end-effector orientation as a unit quaternion (w, x, y, z).
    iterations : int
        Number of iterations taken.
    position_error : float
        Final Euclidean position error in metres.
    orientation_error : float
        Final orientation error in radians (axis-angle magnitude).
    converged_position : bool
        Whether the position tolerance was met.
    converged_orientation : bool
        Whether the orientation tolerance was met.

    See ``docs/references.rst#robotics`` for the DLS IK references.
    """

    success: bool
    joint_angles: np.ndarray
    motor: MVArray | None = None
    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    iterations: int = 0
    position_error: float = float("inf")
    orientation_error: float = float("inf")
    converged_position: bool = False
    converged_orientation: bool = False


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Extract axis-angle representation from a 3×3 rotation matrix.

    Returns ``(angle, axis)`` where ``axis`` is a unit 3-vector and
    ``angle`` is in radians in [0, pi].
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_theta))

    if angle < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])

    if np.pi - angle < 1e-12:
        A = R + np.eye(3)
        for j in range(3):
            v = A[:, j]
            nrm = np.linalg.norm(v)
            if nrm > 1e-10:
                return float(np.pi), v / nrm
        return float(np.pi), np.array([0.0, 0.0, 1.0])

    skew = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    axis = skew / (2.0 * np.sin(angle))
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    return angle, axis / nrm


def _fk_frames(
    alg: Algebra,
    dh_params: list[tuple[float, float, float, float]],
    joint_angles: np.ndarray,
    *,
    joint_types: list[str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, MVArray]:
    """Compute per-joint world-frame data for Jacobian construction.

    Returns ``(joint_positions, joint_z_axes, ee_position, ee_rotation, ee_motor)``.
    ``joint_positions[i]`` and ``joint_z_axes[i]`` are the origin and Z-axis
    of frame {i-1} in world coordinates — i.e. the state *before* joint ``i``'s
    motion is applied.  This matches the standard DH convention where joint ``i``
    acts about Z_{i-1}.
    """
    _validate_cga3d(alg)
    n = len(dh_params)
    joint_types = _validate_joint_types(n, joint_types)
    if joint_angles.shape != (n,):
        raise ValueError(f"Expected {n} joint values, got shape {joint_angles.shape}.")

    motor: MVArray = alg.scalar(1.0)
    joint_positions: list[np.ndarray] = []
    joint_z_axes: list[np.ndarray] = []

    for i in range(n):
        joint_positions.append(motor_to_position(motor, alg))
        R_prev = motor_to_matrix(motor, alg)
        joint_z_axes.append(R_prev[:, 2])

        alpha, a, d, _theta = dh_params[i]

        if joint_types[i] == "prismatic":
            T_z = alg.translate([0.0, 0.0, float(joint_angles[i])])
            R_z = alg.scalar(1.0)
        else:
            T_z = alg.translate([0.0, 0.0, float(d)])
            R_z = _rotor_axis(alg, float(joint_angles[i]), "z")

        T_x = alg.translate([float(a), 0.0, 0.0])
        R_x = _rotor_axis(alg, float(alpha), "x") if abs(float(alpha)) > 1e-15 else alg.scalar(1.0)

        motor = motor * T_z * R_z * T_x * R_x

    ee_position = motor_to_position(motor, alg)
    ee_rotation = motor_to_matrix(motor, alg)

    return joint_positions, joint_z_axes, ee_position, ee_rotation, motor


def _geometric_jacobian(
    joint_positions: list[np.ndarray],
    joint_z_axes: list[np.ndarray],
    ee_position: np.ndarray,
    *,
    joint_types: list[str] | None = None,
) -> np.ndarray:
    """Build the 6 × n geometric Jacobian in world-frame coordinates.

    For a revolute joint *i* with world-frame axis vector
    :math:`\\mathbf{z}_i` and world-frame origin :math:`\\mathbf{p}_i`,
    the linear-velocity column is
    :math:`\\mathbf{z}_i \\times (\\mathbf{p}_{ee} - \\mathbf{p}_i)`
    and the angular-velocity column is :math:`\\mathbf{z}_i`.
    For a prismatic joint the linear column is :math:`\\mathbf{z}_i`
    and the angular column is zero.

    See ``docs/references.rst#robotics``.
    """
    n = len(joint_positions)
    joint_types = _validate_joint_types(n, joint_types)

    J = np.zeros((6, n))

    for i in range(n):
        z = np.asarray(joint_z_axes[i], dtype=float)
        p = np.asarray(joint_positions[i], dtype=float)

        if joint_types[i] == "prismatic":
            J[:3, i] = z
        else:
            J[:3, i] = np.cross(z, ee_position - p)
            J[3:, i] = z

    return J


def _task_error(
    p_current: np.ndarray,
    R_current: np.ndarray,
    p_target: np.ndarray,
    R_target: np.ndarray,
) -> np.ndarray:
    """Compute the 6‑vector task‑space error.

    First three components are position error ``p_target - p_current``,
    last three are the axis‑angle representation of the orientation error
    ``R_target @ R_current.T``.
    """
    pos_err = np.asarray(p_target, dtype=float) - np.asarray(p_current, dtype=float)
    R_err = np.asarray(R_target, dtype=float) @ np.asarray(R_current, dtype=float).T
    angle, axis = _rotation_matrix_to_axis_angle(R_err)
    orient_err = float(angle) * np.asarray(axis, dtype=float)
    return np.concatenate([pos_err, orient_err])


def ik_dls(
    alg: Algebra,
    dh_params: list[tuple[float, float, float, float]],
    target_motor: MVArray,
    *,
    joint_types: list[str] | None = None,
    initial_angles: np.ndarray | None = None,
    max_iterations: int = 100,
    position_tolerance: float = 1e-6,
    orientation_tolerance: float = 1e-6,
    damping: float = 0.1,
    min_damping: float = 1e-6,
    damping_factor: float = 0.5,
    joint_limits: list[tuple[float, float]] | None = None,
) -> IKResult:
    """Damped least-squares inverse kinematics for a DH-parameterised serial chain.

    The solver uses a Levenberg-Marquardt (damped pseudoinverse) method with
    a geometric Jacobian built from CGA forward-kinematics frames.  At each
    iteration the task-space error is decomposed into position error
    (Euclidean) and orientation error (axis-angle of
    :math:`\\mathbf{R}_{target} \\mathbf{R}_{current}^\\mathsf{T}`),
    and the damping factor :math:`\\lambda` is adapted based on error
    improvement.

    Parameters
    ----------
    alg : Algebra
        A CGA algebra (e.g. ``Algebra.cga3d()``).
    dh_params : list of (α, a, d, θ)
        DH parameters. For revolute joints *θ* is replaced by the solver;
        for prismatic joints *d* is replaced (the tuple value is ignored).
    target_motor : MVArray
        The desired end-effector pose expressed as a CGA motor.
    joint_types : list of str, optional
        ``"revolute"`` (default) or ``"prismatic"``.
    initial_angles : ndarray, optional
        Starting joint configuration. Defaults to all zeros.
    max_iterations : int
        Maximum solver iterations (default 100).
    position_tolerance : float
        Convergence threshold for Euclidean position error in metres.
    orientation_tolerance : float
        Convergence threshold for axis‑angle orientation error in radians.
    damping : float
        Initial damping factor λ for the Levenberg‑Marquardt step.
    min_damping : float
        Floor for damping reduction.
    damping_factor : float
        Multiplier for adaptive damping (0 < factor < 1 to reduce, factor > 1
        is used as 1/factor when error improves).
    joint_limits : list of (min, max), optional
        Per-joint limits. Joints are clamped after every iteration.

    Returns
    -------
    IKResult

    See ``docs/references.rst#robotics`` for the DLS, geometric Jacobian, and
    CGA motor-DH references.
    """
    _validate_cga3d(alg)
    _validate_motor_algebra(target_motor, alg)
    n = len(dh_params)
    joint_types = _validate_joint_types(n, joint_types)
    if joint_limits is not None and len(joint_limits) != n:
        raise ValueError(f"Expected {n} joint limits, got {len(joint_limits)}.")

    p_target = motor_to_position(target_motor, alg)
    R_target = motor_to_matrix(target_motor, alg)

    q = np.asarray(initial_angles, dtype=float) if initial_angles is not None else np.zeros(n)
    if q.shape != (n,):
        raise ValueError(f"Expected {n} initial joint values, got shape {q.shape}.")
    lam = float(damping)
    if damping < 0.0 or min_damping < 0.0:
        raise ValueError("Damping values must be non-negative.")
    if damping_factor <= 0.0:
        raise ValueError("damping_factor must be positive.")

    iteration = -1

    for iteration in range(max_iterations):
        joint_pos, joint_z, p_curr, R_curr, _ = _fk_frames(
            alg, dh_params, q, joint_types=joint_types,
        )

        e = _task_error(p_curr, R_curr, p_target, R_target)
        pos_err = float(np.linalg.norm(e[:3]))
        orient_err = float(np.linalg.norm(e[3:]))

        converged_pos = pos_err < position_tolerance
        converged_orient = orient_err < orientation_tolerance

        if converged_pos and converged_orient:
            motor, pos, quat = _resolve_fk_result(alg, dh_params, q, joint_types)
            return IKResult(
                success=True,
                joint_angles=q,
                motor=motor,
                position=pos,
                orientation=quat,
                iterations=iteration + 1,
                position_error=pos_err,
                orientation_error=orient_err,
                converged_position=True,
                converged_orientation=True,
            )

        J = _geometric_jacobian(joint_pos, joint_z, p_curr, joint_types=joint_types)

        A = J @ J.T + lam * lam * np.eye(6)
        try:
            dq = J.T @ np.linalg.solve(A, e)
        except np.linalg.LinAlgError:
            lam = max(lam * 2.0, 1e-6)
            continue

        q_new = q + dq

        if joint_limits is not None:
            for j in range(n):
                lo, hi = joint_limits[j]
                q_new[j] = float(np.clip(q_new[j], lo, hi))

        _, _, p_new, R_new, _ = _fk_frames(
            alg, dh_params, q_new, joint_types=joint_types,
        )
        e_new = _task_error(p_new, R_new, p_target, R_target)
        err_new = float(np.linalg.norm(e_new))
        err_old = float(np.linalg.norm(e))

        if err_new < err_old:
            q = q_new
            lam = max(lam * damping_factor, min_damping)
        else:
            lam = lam * 2.0
            if lam > 1e3:
                q = q_new
                lam = max(lam * damping_factor, min_damping)

        if float(np.linalg.norm(dq)) < 1e-12:
            break

    joint_pos, joint_z, p_curr, R_curr, motor = _fk_frames(
        alg, dh_params, q, joint_types=joint_types,
    )
    e = _task_error(p_curr, R_curr, p_target, R_target)
    pos_err = float(np.linalg.norm(e[:3]))
    orient_err = float(np.linalg.norm(e[3:]))

    return IKResult(
        success=False,
        joint_angles=q,
        motor=motor,
        position=p_curr,
        orientation=motor_to_quaternion(motor, alg),
        iterations=max(iteration + 1, 0),
        position_error=pos_err,
        orientation_error=orient_err,
        converged_position=pos_err < position_tolerance,
        converged_orientation=orient_err < orientation_tolerance,
    )


def ik_cga_spherical_wrist(
    alg: Algebra,
    dh_params: list[tuple[float, float, float, float]],
    target_motor: MVArray,
    *,
    joint_types: list[str] | None = None,
    initial_angles: np.ndarray | None = None,
    max_iterations: int = 100,
    position_tolerance: float = 1e-6,
    orientation_tolerance: float = 1e-6,
    damping: float = 0.1,
    joint_limits: list[tuple[float, float]] | None = None,
) -> IKResult:
    """Full-chain CGA IK for DH serial arms with a spherical wrist.

    The solver follows the CGA spherical-wrist split used in the robotics
    references: construct the wrist-centre target as a conformal point, build
    shoulder/elbow branch seeds from sphere intersections, then solve the full
    end-effector motor target with the CGA-frame DLS solver.  The returned
    object is an ``IKResult`` with joint values, not an intermediate meet.
    """
    _validate_cga3d(alg)
    _validate_motor_algebra(target_motor, alg)
    n = len(dh_params)
    joint_types = _validate_joint_types(n, joint_types)
    if n < 6:
        raise ValueError("cga_spherical_wrist expects at least 6 DH joints.")
    if any(kind != "revolute" for kind in joint_types):
        raise ValueError("cga_spherical_wrist currently supports revolute joints.")
    if joint_limits is not None and len(joint_limits) != n:
        raise ValueError(f"Expected {n} joint limits, got {len(joint_limits)}.")

    seeds = _cga_spherical_wrist_seeds(alg, dh_params, target_motor)
    if initial_angles is not None:
        q0 = np.asarray(initial_angles, dtype=float)
        if q0.shape != (n,):
            raise ValueError(f"Expected {n} initial joint values, got shape {q0.shape}.")
        seeds.insert(0, q0)

    best: IKResult | None = None
    for seed in seeds:
        result = ik_dls(
            alg,
            dh_params,
            target_motor,
            joint_types=joint_types,
            initial_angles=seed,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            damping=damping,
            joint_limits=joint_limits,
        )
        if best is None or _ik_result_error(result) < _ik_result_error(best):
            best = result
        if result.success:
            return result

    assert best is not None
    return best


def _ik_result_error(result: IKResult) -> float:
    return float(result.position_error + result.orientation_error)


def _cga_spherical_wrist_seeds(
    alg: Algebra,
    dh_params: list[tuple[float, float, float, float]],
    target_motor: MVArray,
) -> list[np.ndarray]:
    n = len(dh_params)
    target_pos = motor_to_position(target_motor, alg)
    target_rot = motor_to_matrix(target_motor, alg)
    d6 = float(dh_params[5][2])
    wrist = target_pos - d6 * target_rot[:, 2]

    base_z = float(dh_params[0][2])
    shoulder = np.array([0.0, 0.0, base_z])
    a2 = abs(float(dh_params[1][1]))
    a3 = abs(float(dh_params[2][1]))

    seeds: list[np.ndarray] = [np.zeros(n)]
    if a2 < 1e-12 or a3 < 1e-12:
        return seeds

    # CGA branch construction: the elbow lies on the intersection circle of
    # the shoulder and wrist reach spheres.  The angles below are only seeds;
    # the final motor target is solved by CGA-frame DLS.
    shoulder_sphere = alg.sphere(shoulder, a2)
    wrist_sphere = alg.sphere(wrist, a3)
    try:
        circle = sphere_sphere(shoulder_sphere, wrist_sphere)
    except Exception:
        circle = None

    q1_base = math.atan2(float(wrist[1]), float(wrist[0]))
    for q1 in (q1_base, q1_base + math.pi):
        c1 = math.cos(q1)
        s1 = math.sin(q1)
        radial = c1 * wrist[0] + s1 * wrist[1]
        z = wrist[2] - base_z
        r = math.hypot(float(radial), float(z))
        if r < 1e-12:
            continue
        cos_q3 = np.clip((r * r - a2 * a2 - a3 * a3) / (2.0 * a2 * a3), -1.0, 1.0)
        for sign in (1.0, -1.0):
            sin_q3 = sign * math.sqrt(max(0.0, 1.0 - float(cos_q3 * cos_q3)))
            q3 = math.atan2(sin_q3, float(cos_q3))
            q2 = math.atan2(z, radial) - math.atan2(a3 * sin_q3, a2 + a3 * float(cos_q3))
            seed = np.zeros(n)
            seed[:3] = [q1, q2, q3]
            if circle is not None:
                seed[3] = -q2 - q3
            seeds.append(seed)

    return seeds


def _resolve_fk_result(
    alg: Algebra,
    dh_params: list[tuple[float, float, float, float]],
    joint_angles: np.ndarray,
    joint_types: list[str],
) -> tuple[MVArray, np.ndarray, np.ndarray]:
    """Run FK for given angles and return ``(motor, position, quaternion)``."""
    _, _, pos, rot, motor = _fk_frames(alg, dh_params, joint_angles, joint_types=joint_types)
    quat = motor_to_quaternion(motor, alg)
    return motor, pos, quat


__all__ = [
    "EXPERIMENTAL_WARNING",
    "IKResult",
    "Joint",
    "Link",
    "RobotModel",
    "dump_crobot",
    "fk",
    "fk_model",
    "ik",
    "ik_cga_spherical_wrist",
    "ik_dls",
    "importurdf",
    "joint_motion_motor",
    "line_plane",
    "load_crobot",
    "model_from_dh",
    "motor_to_matrix",
    "motor_to_position",
    "motor_to_quaternion",
    "planar_two_link_ik",
    "point_circle_projection",
    "sphere_sphere",
]


def _validate_cga3d(alg: Algebra) -> None:
    if alg.dimension != 5 or alg.signature != (1, 1, 1, 1, -1):
        raise ValueError("Experimental robotics CGA helpers require Algebra.cga3d().")


def _validate_motor_algebra(motor: MVArray, alg: Algebra) -> None:
    if motor.algebra != alg.spec:
        raise ValueError("Motor must belong to the provided algebra.")


def _validate_same_cga(*values: MVArray) -> None:
    if not values:
        return
    algebra = values[0].algebra
    alg = Algebra(algebra)
    _validate_cga3d(alg)
    for value in values:
        if value.algebra != algebra:
            raise ValueError("CGA objects must belong to the same algebra.")


def _validate_joint_types(n: int, joint_types: list[str] | None) -> list[str]:
    resolved = ["revolute"] * n if joint_types is None else list(joint_types)
    if len(resolved) != n:
        raise ValueError(f"Expected {n} joint types, got {len(resolved)}.")
    invalid = sorted(set(resolved) - {"revolute", "prismatic"})
    if invalid:
        names = ", ".join(repr(item) for item in invalid)
        raise ValueError(f"Unsupported joint type(s): {names}.")
    return resolved
