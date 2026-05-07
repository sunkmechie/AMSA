# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""Compare AMSA's supported IK solver surfaces on UR5 geometry.

AMSA exposes two full serial-chain IK solver surfaces plus lower-level CGA
geometric IK primitives:

- ``robo.ik_dls()`` solves a full DH-chain end-effector motor target.
- ``robo.ik(..., solver="cga_spherical_wrist")`` solves a full spherical-wrist
  DH-chain target and returns joint angles.
- ``robo.ik(..., solver="cga_sphere_sphere")`` solves a sphere-sphere meet.
- ``robo.ik(..., solver="cga_point_circle")`` projects a point onto a CGA
  circle, selecting one point from that meet circle.

The full CGA spherical-wrist solver uses the primitive pair to seed the
shoulder/elbow branches, then solves the complete motor target.

References are collected in docs/references.rst under Robotics.

Run with:

    uv run python examples/robotics/cga_ik_ur5_solver_comparison.py
"""

from __future__ import annotations

import math

import numpy as np

import amsa.robo as robo
from amsa import Algebra

UR5_DH: list[tuple[float, float, float, float]] = [
    (math.pi / 2, 0.0, 0.089159, 0.0),
    (0.0, -0.42500, 0.0, 0.0),
    (0.0, -0.39225, 0.0, 0.0),
    (math.pi / 2, 0.0, 0.10915, 0.0),
    (-math.pi / 2, 0.0, 0.09465, 0.0),
    (0.0, 0.0, 0.08230, 0.0),
]

UR5_LIMITS: list[tuple[float, float]] = [(-2.0 * math.pi, 2.0 * math.pi)] * 6


def fmt_vec(values: np.ndarray, *, width: int = 10) -> str:
    return "  ".join(f"{float(value):{width}.5f}" for value in values)


def fmt_angles(values: np.ndarray | list[float]) -> str:
    return "  ".join(f"q{i + 1}={float(value): .5f}" for i, value in enumerate(values))


def distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)))


def main() -> None:
    alg = Algebra.cga3d()

    target_angles = [0.45, -1.05, 0.85, -0.35, 0.65, -0.40]
    dh_target = [
        (alpha, a, d, theta)
        for (alpha, a, d, _), theta in zip(UR5_DH, target_angles, strict=True)
    ]
    target_chain = robo.fk(alg, dh_target)
    target_motor = target_chain[-1]["motor"]

    print("\n" + "=" * 72)
    print("  UR5 IK solver comparison")
    print("=" * 72)
    print()
    print("Target joint configuration:")
    print(f"  {fmt_angles(target_angles)}")
    print()
    print("Target end-effector pose:")
    print(f"  position:    {fmt_vec(np.asarray(target_chain[-1]['position']))}")
    print(f"  quaternion:  {fmt_vec(np.asarray(target_chain[-1]['orientation']))}")
    print()

    print("1. Full-chain IK: damped least squares")
    dls = robo.ik_dls(
        alg,
        UR5_DH,
        target_motor,
        joint_limits=UR5_LIMITS,
        position_tolerance=1e-8,
        orientation_tolerance=1e-8,
    )
    print(f"  converged:          {dls.success}")
    print(f"  iterations:         {dls.iterations}")
    print(f"  position error:     {dls.position_error:.3e} m")
    print(f"  orientation error:  {dls.orientation_error:.3e} rad")
    print(f"  solved joints:      {fmt_angles(dls.joint_angles)}")
    print()

    print("2. Full-chain CGA IK: spherical-wrist branch solver")
    cga_full = robo.ik(
        alg,
        UR5_DH,
        target_motor,
        solver="cga_spherical_wrist",
        joint_limits=UR5_LIMITS,
        position_tolerance=1e-8,
        orientation_tolerance=1e-8,
    )
    print(f"  converged:          {cga_full.success}")
    print(f"  iterations:         {cga_full.iterations}")
    print(f"  position error:     {cga_full.position_error:.3e} m")
    print(f"  orientation error:  {cga_full.orientation_error:.3e} rad")
    print(f"  solved joints:      {fmt_angles(cga_full.joint_angles)}")
    joint_delta = np.linalg.norm(cga_full.joint_angles - dls.joint_angles)
    print(f"  joint delta vs DLS:  {joint_delta:.3e} rad")
    print()

    print("3. CGA geometric IK primitive: sphere-sphere elbow circle")
    shoulder = np.asarray(target_chain[0]["position"], dtype=float)
    elbow = np.asarray(target_chain[1]["position"], dtype=float)
    wrist = np.asarray(target_chain[2]["position"], dtype=float)
    shoulder_radius = distance(shoulder, elbow)
    wrist_radius = distance(wrist, elbow)

    shoulder_sphere = alg.sphere(shoulder, shoulder_radius)
    wrist_sphere = alg.sphere(wrist, wrist_radius)
    elbow_circle = robo.ik(
        shoulder_sphere,
        wrist_sphere,
        solver="cga_sphere_sphere",
    )

    incidence = alg.point(elbow) ^ elbow_circle
    print(f"  shoulder:           {fmt_vec(shoulder)}")
    print(f"  wrist:              {fmt_vec(wrist)}")
    print(f"  target elbow:       {fmt_vec(elbow)}")
    print(f"  meet grades:        {elbow_circle.grades}")
    print(f"  target incidence:   {np.max(np.abs(incidence.values)):.3e}")
    print()

    print("4. CGA geometric IK primitive: point-circle elbow selection")
    direct_elbow_circle = alg.circle_through_points(
        alg.point(shoulder),
        alg.point(elbow),
        alg.point(wrist),
    )
    circle_normal = np.cross(elbow - shoulder, wrist - shoulder)
    circle_normal = circle_normal / np.linalg.norm(circle_normal)
    preferred_side = elbow + 0.05 * circle_normal
    recovered_elbow = robo.ik(
        alg.point(preferred_side),
        direct_elbow_circle,
        solver="cga_point_circle",
    )
    recovered_elbow_xyz = alg.extract_point(recovered_elbow)
    recovered_incidence = recovered_elbow ^ direct_elbow_circle
    print(f"  preferred side:     {fmt_vec(preferred_side)}")
    print(f"  recovered elbow:    {fmt_vec(recovered_elbow_xyz)}")
    print(f"  elbow error:        {distance(recovered_elbow_xyz, elbow):.3e} m")
    print(f"  circle incidence:   {np.max(np.abs(recovered_incidence.values)):.3e}")
    print()

    print("Summary")
    print("  DLS and cga_spherical_wrist both solve the complete UR5 motor target.")
    print("  The sphere-sphere and point-circle paths remain exposed as CGA")
    print("  geometric building blocks used by branch-aware full-chain solvers.")
    print()


if __name__ == "__main__":
    main()
