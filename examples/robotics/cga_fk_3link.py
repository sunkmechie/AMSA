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
dh = [
    (0.0,        1.0,   0.0,   0.0),
    (math.pi / 3, 0.8,   0.0,   0.0),
    (0.0,        0.5,   0.0,   0.0),
]

# ---- Configuration 1: all zeros -----------------------------------------
angles = [(α, a, d, 0.0) for α, a, d, _ in dh]
results = robo.fk(alg, angles)

print("Configuration: all θ = 0")
for i, r in enumerate(results):
    p = r['position']
    print(f"  link {i + 1}: pos ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

# ---- Configuration 2: some rotation -------------------------------------
angles2 = [
    (0.0,        1.0,   0.0,   math.pi / 4),
    (math.pi / 3, 0.8,   0.0,   math.pi / 6),
    (0.0,        0.5,   0.0,  -math.pi / 3),
]
results2 = robo.fk(alg, angles2)

print("\nConfiguration: θ₁=π/4, θ₂=π/6, θ₃=-π/3")
for i, r in enumerate(results2):
    p = r["position"]
    print(f"  link {i + 1}: pos ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

# ---- Verify: motors are proper even-grade versors -----------------------
print("\nMotor grades (should be subset of {0, 2, 4}):")
for i, r in enumerate(results2):
    print(f"  motor {i + 1}: {set(r['motor'].grades)}")
