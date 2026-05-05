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

    References
    ----------
    Buss, S. R. (2004).  Introduction to Inverse Kinematics with Jacobian
    Transpose, Pseudoinverse and Damped Least Squares methods.  IEEE Journal
    of Robotics and Automation.
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

    References
    ----------
    Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2010).
    *Robotics: Modelling, Planning and Control*, Springer, §3 (Differential
    Kinematics and Statics).
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

    References
    ----------
    Buss, S. R. (2004).  Introduction to Inverse Kinematics with Jacobian
    Transpose, Pseudoinverse and Damped Least Squares methods.  *IEEE Journal
    of Robotics and Automation*.

    Wampler, C. W. (1986).  Manipulator Inverse Kinematic Solutions Based on
    Vector Formulations and Damped Least-Squares Methods.  *IEEE Transactions
    on Systems, Man, and Cybernetics* 16(1).

    Nakamura, Y. & Hanafusa, H. (1986).  Inverse Kinematic Solutions With
    Singularity Robustness for Robot Manipulator Control.  *Journal of Dynamic
    Systems, Measurement, and Control* 108(3).

    The geometric Jacobian formulation follows Siciliano et al. (2010),
    *Robotics: Modelling, Planning and Control*, Springer, §3.

    The Denavit-Hartenberg parameterisation and motor composition are drawn
    from Bayro-Corrochano & Zamora-Esquivel (2007), "Differential and inverse
    kinematics of robot devices using conformal geometric algebra", *Robotica*
    25(1), pp. 43–61.
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
    "ik",
    "ik_dls",
    "importurdf",
    "load_crobot",
    "motor_to_matrix",
    "motor_to_position",
    "motor_to_quaternion",
    "planar_two_link_ik",
]


def _validate_cga3d(alg: Algebra) -> None:
    if alg.dimension != 5 or alg.signature != (1, 1, 1, 1, -1):
        raise ValueError("Experimental robotics CGA helpers require Algebra.cga3d().")


def _validate_motor_algebra(motor: MVArray, alg: Algebra) -> None:
    if motor.algebra != alg.spec:
        raise ValueError("Motor must belong to the provided algebra.")


def _validate_joint_types(n: int, joint_types: list[str] | None) -> list[str]:
    resolved = ["revolute"] * n if joint_types is None else list(joint_types)
    if len(resolved) != n:
        raise ValueError(f"Expected {n} joint types, got {len(resolved)}.")
    invalid = sorted(set(resolved) - {"revolute", "prismatic"})
    if invalid:
        names = ", ".join(repr(item) for item in invalid)
        raise ValueError(f"Unsupported joint type(s): {names}.")
    return resolved
