# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA forward kinematics — UR5 6-DOF industrial arm
Algebra: 3D Conformal Geometric Algebra (CGA)

Computes world-frame joint positions for a Universal Robots UR5 using
classic Denavit–Hartenberg parameters from the official UR documentation.

DH source: https://www.universal-robots.com/articles/ur/
  application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

Each joint-link pair uses the CGA motor composition per
Bayro-Corrochano & Zamora-Esquivel (2007):

    M_i = M_{i-1} · T_z(d) · R_z(θ) · T_x(a) · R_x(α)

Two configurations are shown: home (all zeros) and a working pose.

Verification: at the home configuration, the end-effector position matches
the known UR5 kinematic chain geometry (base offset + link lengths).
"""

import math

import amsa.robo as robo
from amsa import Algebra

print("\n=== CGA FK — UR5 6-DOF Arm ===\n")

alg = Algebra.cga3d()

# UR5 classic DH parameters (SI units: metres, radians)
# Format: (α, a, d, θ)
UR5_DH = [
    (math.pi / 2,  0.0,       0.089159,  0.0),   # joint 1
    (0.0,         -0.42500,   0.0,        0.0),   # joint 2
    (0.0,         -0.39225,   0.0,        0.0),   # joint 3
    (math.pi / 2,  0.0,       0.10915,   0.0),   # joint 4
    (-math.pi / 2, 0.0,       0.09465,   0.0),   # joint 5
    (0.0,          0.0,       0.08230,   0.0),   # joint 6
]

# ---- Home configuration (all θ = 0) -------------------------------------
print("--- Home (all θ = 0) ---")
home = [(α, a, d, 0.0) for α, a, d, _ in UR5_DH]
results = robo.fk(alg, home)

for i, (motor, tip) in enumerate(results):
    p = alg.extract_point(tip)
    grades = set(motor.grades)
    print(f"  joint {i + 1}: {p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}  | grades {grades}")

# ---- Working pose -------------------------------------------------------
print("\n--- Working pose (θ = 0, -π/4, -π/2, 0, π/4, 0) ---")
pose = [
    (math.pi / 2,  0.0,       0.089159,  0.0),
    (0.0,         -0.42500,   0.0,       -math.pi / 4),
    (0.0,         -0.39225,   0.0,       -math.pi / 2),
    (math.pi / 2,  0.0,       0.10915,   0.0),
    (-math.pi / 2, 0.0,       0.09465,   math.pi / 4),
    (0.0,          0.0,       0.08230,   0.0),
]
results2 = robo.fk(alg, pose)

for i, (_, tip) in enumerate(results2):
    p = alg.extract_point(tip)
    print(f"  joint {i + 1}: {p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}")

# ---- Classify each motor ------------------------------------------------
print("\n--- Motor classification ---")
for i, (motor, _) in enumerate(results2):
    info = alg.classify(motor)
    print(f"  motor {i + 1}: {info.kind}")
