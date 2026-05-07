# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA inverse kinematics — UR5 6-DOF industrial arm
Algebra: 3D Conformal Geometric Algebra (CGA)

Given a target end-effector pose expressed as a CGA motor, this example
uses damped least-squares (Levenberg-Marquardt) numerical IK to solve for
the six joint angles θ₁…θ₆.

The solver uses the geometric Jacobian derived from the DH-parameterised
CGA forward-kinematics chain and operates in task space (position error +
axis-angle orientation error).  Joint limits are enforced via clamping.

DH source: Universal Robots official documentation
  https://www.universal-robots.com/articles/ur/
    application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

References
----------
Bayro-Corrochano & Zamora-Esquivel (2007), "Differential and inverse
kinematics of robot devices using conformal geometric algebra", Robotica
25(1), pp. 43–61.

Buss, S. R. (2004).  Introduction to Inverse Kinematics with Jacobian
Transpose, Pseudoinverse and Damped Least Squares methods.  IEEE JRA.
"""

import math

import numpy as np

import amsa.robo as robo
from amsa import Algebra

print("\n" + "=" * 64)
print("  CGA Inverse Kinematics — UR5 6-DOF Arm (DLS)")
print("=" * 64)
print()

alg = Algebra.cga3d()

# ---------------------------------------------------------------------------
# UR5 classic DH parameters (metres, radians)
# (α, a, d, θ) — θ is the variable for revolute joints
# ---------------------------------------------------------------------------
UR5_DH: list[tuple[float, float, float, float]] = [
    (math.pi / 2,  0.0,       0.089159,  0.0),
    (0.0,         -0.42500,   0.0,        0.0),
    (0.0,         -0.39225,   0.0,        0.0),
    (math.pi / 2,  0.0,       0.10915,   0.0),
    (-math.pi / 2, 0.0,       0.09465,   0.0),
    (0.0,          0.0,       0.08230,   0.0),
]

# Full-range joint limits (radians)
UR5_LIMITS: list[tuple[float, float]] = [
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
    (-2 * math.pi, 2 * math.pi),
]


def _fmt_angles(angles: np.ndarray | list[float], width: int = 9) -> str:
    return "  ".join(f"θ{i + 1}={float(a):{width}.4f}" for i, a in enumerate(angles))


def _fmt_vec(v: np.ndarray, width: int = 9) -> str:
    return "  ".join(f"{float(x):{width}.4f}" for x in v)


# ---------------------------------------------------------------------------
# 1.  Pick a target joint configuration and compute its FK
# ---------------------------------------------------------------------------
target_thetas = [0.4, -0.9, 0.6, -0.3, 0.7, -0.5]

print("Target joint angles:")
print(f"  {_fmt_angles(target_thetas)}")
print()

dh_target = [(α, a, d, θ) for (α, a, d, _), θ in zip(UR5_DH, target_thetas, strict=True)]
fk_target = robo.fk(alg, dh_target)
target_ee = fk_target[-1]
target_motor = target_ee["motor"]

print("Target end-effector pose (from FK):")
print(f"  position:     {_fmt_vec(target_ee['position'])}")
print(f"  orientation:  {_fmt_vec(target_ee['orientation'])}  (quaternion w,x,y,z)")
print()

# ---------------------------------------------------------------------------
# 2.  Solve IK for the target motor
# ---------------------------------------------------------------------------
print("Solving IK (DLS with adaptive damping and joint limits) …\n")

result = robo.ik_dls(
    alg,
    UR5_DH,
    target_motor,
    joint_limits=UR5_LIMITS,
    position_tolerance=1e-8,
    orientation_tolerance=1e-8,
)

# ---------------------------------------------------------------------------
# 3.  Report results
# ---------------------------------------------------------------------------
if result.success:
    print("  IK converged ✓")
else:
    print("  IK did not fully converge (best-effort result below)")

print()
print(f"  Iterations taken:      {result.iterations}")
print(f"  Final position error:  {result.position_error:.2e} m")
print(f"  Final orientation err: {result.orientation_error:.2e} rad")
print()

print("Solved joint angles:")
print(f"  {_fmt_angles(result.joint_angles)}")
print()

# Compare to target
angle_delta = np.linalg.norm(result.joint_angles - np.array(target_thetas))
print(f"  ‖θ_solved − θ_target‖ = {angle_delta:.2e} rad")
print()

# ---------------------------------------------------------------------------
# 4.  Verify:  FK of solved angles should match target pose
# ---------------------------------------------------------------------------
if result.position is not None:
    print("Verification (FK of solved angles):")
    print(f"  position:     {_fmt_vec(result.position)}")
    pos_err = np.linalg.norm(np.asarray(result.position)
                             - np.asarray(target_ee["position"]))
    print(f"  ‖p_solved − p_target‖ = {pos_err:.2e} m")
    print()

    print("  achieved orientation:  "
          f"{_fmt_vec(result.orientation)}")
    print()

# ---------------------------------------------------------------------------
# 5.  Self-consistency round-trip test
# ---------------------------------------------------------------------------
print("─" * 64)
print("Self-consistency round-trip")
print("─" * 64)
print()

dh_solved = [(α, a, d, float(th)) for (α, a, d, _), th
             in zip(UR5_DH, result.joint_angles, strict=True)]
fk_solved = robo.fk(alg, dh_solved)
ee_solved = fk_solved[-1]

pos_solved = np.asarray(ee_solved["position"])
pos_target = np.asarray(target_ee["position"])
quat_solved = np.asarray(ee_solved["orientation"])
quat_target = np.asarray(target_ee["orientation"])

print(f"  FK(θ_solved) position:    {_fmt_vec(pos_solved)}")
print(f"  FK(θ_target) position:    {_fmt_vec(pos_target)}")
print(f"  Position mismatch:        "
      f"{np.max(np.abs(pos_solved - pos_target)):.2e} m")
print()

print(f"  FK(θ_solved) orientation: {_fmt_vec(quat_solved)}")
print(f"  FK(θ_target) orientation: {_fmt_vec(quat_target)}")
print(f"  Quaternion inner product: "
      f"{np.dot(quat_solved, quat_target):.6f}  (≈1 for same pose)")
print()

# ---------------------------------------------------------------------------
# 6.  Single‑joint solve (shoulder‑only reach, no orientation target)
# ---------------------------------------------------------------------------
print("─" * 64)
print("Single-joint solve (q₁ only)")
print("─" * 64)
print()

theta1 = 1.2
dh_single = [(α, a, d, theta1 if i == 0 else 0.0)
             for i, (α, a, d, _) in enumerate(UR5_DH)]
fk_single = robo.fk(alg, dh_single)
single_target = fk_single[-1]["motor"]

result_single = robo.ik_dls(
    alg, UR5_DH, single_target,
    joint_limits=UR5_LIMITS,
    position_tolerance=1e-8,
    orientation_tolerance=1e-8,
)

print(f"  Target θ₁ = {theta1:.4f}")
print(f"  IK result: iterations={result_single.iterations}, "
      f"converged={result_single.success}")
if result_single.success:
    best_theta = float(result_single.joint_angles[0])
    print(f"  For joint 1, cyclic equivalent: "
          f"sin={math.sin(best_theta):.4f}, "
          f"cos={math.cos(best_theta):.4f}")
    print(f"  (sin, cos) of target θ₁: "
          f"sin={math.sin(theta1):.4f}, "
          f"cos={math.cos(theta1):.4f}")

print()
