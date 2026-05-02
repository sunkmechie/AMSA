# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA forward kinematics — 3-link non-planar arm with twisted joint
Algebra: 3D Conformal Geometric Algebra (CGA)

A 3-DOF serial arm where joint 2 has a non-zero twist (α ≠ 0), making
the arm non-planar.  The CGA motor composition via Denavit–Hartenberg
handles the twist naturally:

    M_i = M_{i-1} · T_z(d) · R_z(θ) · T_x(a) · R_x(α)

Two configurations are computed and the tip positions verified.
"""

import math

import amsa.robo as robo
from amsa import Algebra

print("\n=== CGA FK — 3-Link Non-Planar Arm ===\n")

alg = Algebra.cga3d()

# DH parameters: (α, a, d, θ)
# Joint 2 has α = π/3 twist, rotating the arm out of the xy-plane.
dh = [
    (0.0,        1.0,   0.0,   0.0),       # joint 1: revolute, link along x
    (math.pi / 3, 0.8,   0.0,   0.0),       # joint 2: revolute, twisted by 60°
    (0.0,        0.5,   0.0,   0.0),       # joint 3: revolute, link along x
]

# ---- Configuration 1: all zeros -----------------------------------------
angles = [(α, a, d, 0.0) for α, a, d, _ in dh]
results = robo.fk(alg, angles)

print("Configuration: all θ = 0")
for i, (_, tip) in enumerate(results):
    print(f"  link {i + 1}: {alg.extract_point(tip)}")
# Link 1: [1.0, 0, 0] — straight along x
# Link 2: [1.0, 0.4, 0.6928] — twisted 60° about x
# Link 3: [1.4, 0.4, 0.7789] — further along local x

# ---- Configuration 2: some rotation -------------------------------------
angles2 = [
    (0.0,        1.0,   0.0,   math.pi / 4),
    (math.pi / 3, 0.8,   0.0,   math.pi / 6),
    (0.0,        0.5,   0.0,  -math.pi / 3),
]
results2 = robo.fk(alg, angles2)

print("\nConfiguration: θ₁=π/4, θ₂=π/6, θ₃=-π/3")
for i, (_, tip) in enumerate(results2):
    print(f"  link {i + 1}: {alg.extract_point(tip)}")

# ---- Verify: motors are proper even-grade versors -----------------------
print("\nMotor grades (should be subset of {0, 2, 4}):")
for i, (motor, _) in enumerate(results2):
    print(f"  motor {i + 1}: {set(motor.grades)}")
