# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA forward kinematics — UR5 6-DOF industrial arm
Algebra: 3D Conformal Geometric Algebra (CGA)

Input: joint angles (θ₁…θ₆) applied to UR5 Denavit–Hartenberg parameters
Output: end-effector Cartesian pose — position + quaternion orientation

Each DH tuple (α, a, d, θ) is composed into a CGA motor:

    M_i = M_{i-1} · T_z(d) · R_z(θ) · T_x(a) · R_x(α)

The motor M_6 encodes the full pose.  ``robo.fk()`` pre‑computes
the position and orientation (quaternion) for each link.

DH source: Universal Robots official documentation
  https://www.universal-robots.com/articles/ur/
    application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

Citation: Bayro-Corrochano & Zamora-Esquivel (2007), Robotica 25(1), pp. 43–61.
"""

import math

import amsa.robo as robo
from amsa import Algebra

print("\n=== CGA FK — UR5 6-DOF Arm ===\n")

alg = Algebra.cga3d()

# UR5 classic DH parameters (metres, radians)
# (α, a, d, θ) — θ varies per configuration
UR5_DH = [
    (math.pi / 2,  0.0,       0.089159,  0.0),
    (0.0,         -0.42500,   0.0,        0.0),
    (0.0,         -0.39225,   0.0,        0.0),
    (math.pi / 2,  0.0,       0.10915,   0.0),
    (-math.pi / 2, 0.0,       0.09465,   0.0),
    (0.0,          0.0,       0.08230,   0.0),
]


def _fmt_vec(v, width: int = 8) -> str:
    return ", ".join(f"{x:{width}.4f}" for x in v)


# ---- Home configuration (all θ = 0) -----------------------------------------

print("--- Joint Input: home (θ = 0, 0, 0, 0, 0, 0) ---\n")
angles = [(α, a, d, 0.0) for α, a, d, _ in UR5_DH]
results = robo.fk(alg, angles)

for i, r in enumerate(results):
    p = r["position"]
    print(f"  joint {i + 1}: pos ({_fmt_vec(p)})")

# End-effector pose
ee = results[5]
print("\n  Cartesian output (end-effector pose):")
print(f"    position:    ({_fmt_vec(ee['position'])})")
print(f"    orientation: ({_fmt_vec(ee['orientation'])}  )  ← quaternion (w, x, y, z)")

# ---- Working configuration --------------------------------------------------

print("""

--- Joint Input: working (θ = 0, -π/4, -π/2, 0,  π/4, 0) ---
""")
angles2 = [
    (math.pi / 2,  0.0,       0.089159,  0.0),
    (0.0,         -0.42500,   0.0,       -math.pi / 4),
    (0.0,         -0.39225,   0.0,       -math.pi / 2),
    (math.pi / 2,  0.0,       0.10915,   0.0),
    (-math.pi / 2, 0.0,       0.09465,   math.pi / 4),
    (0.0,          0.0,       0.08230,   0.0),
]
results2 = robo.fk(alg, angles2)

for i, r in enumerate(results2):
    p = r["position"]
    print(f"  joint {i + 1}: pos ({_fmt_vec(p)})")

ee2 = results2[5]
print("\n  Cartesian output (end-effector pose):")
print(f"    position:    ({_fmt_vec(ee2['position'])})")
print(f"    orientation: ({_fmt_vec(ee2['orientation'])}  )  ← quaternion (w, x, y, z)")

# ---- Rotation matrix (via motor_to_matrix) ----------------------------------
print("\n  Full rotation matrix (via robo.motor_to_matrix):")
R = robo.motor_to_matrix(ee2["motor"], alg)
for j in range(3):
    print(f"    | {_fmt_vec(R[:, j], width=9)} |")

# ---- Verify quaternion ↔ matrix round-trip ----------------------------------
q = robo.motor_to_quaternion(ee2["motor"], alg)
print(f"\n  Quaternion norm: {sum(x * x for x in q):.6f}  (should be 1.0)")
