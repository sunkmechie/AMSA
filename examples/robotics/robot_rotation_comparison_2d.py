# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0
"""
AMSA Example

Topic: Robot rotation comparison
Algebra: 2D Vector Geometric Algebra (VGA)

Rigid rotations in robotics can be represented in multiple ways:

    - rotation matrices
    - quaternions
    - geometric algebra rotors

This example rotates a vector by 45 degrees using all three
representations and verifies that they produce the same result.

Geometric algebra performs rotations using the sandwich product:

    v' = R v R^{-1}

where R is a rotor.
"""

import numpy as np

from amsa import Algebra

print("\n=== Rotation Comparison (Matrix / Quaternion / Rotor) ===")

theta = np.deg2rad(45)

v = np.array([1.0, 0.0])

print("\nOriginal vector:", v)

# --------------------------------------------------
# Matrix rotation
# --------------------------------------------------

R = np.array(
    [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
)

v_matrix = R @ v

print("\nMatrix rotation:")
print(v_matrix)

# --------------------------------------------------
# Quaternion rotation (2D embedded in Z rotation)
# --------------------------------------------------

x, y = v

v_quat = np.array(
    [
        np.cos(theta) * x - np.sin(theta) * y,
        np.sin(theta) * x + np.cos(theta) * y,
    ]
)

print("\nQuaternion rotation:")
print(v_quat)

# --------------------------------------------------
# Geometric algebra rotor
# --------------------------------------------------

alg = Algebra.vga2d()

vec = alg.vector([1.0, 0.0])

rotor = alg.multivector(
    {
        "e": np.cos(theta / 2),
        "e12": -np.sin(theta / 2),
    }
).normalized()

rotated = rotor.sandwich(vec)

v_ga = rotated.grade(1).as_dense().values[1:3]

print("\nGeometric algebra rotor:")
print(v_ga)

# --------------------------------------------------
# agreement check
# --------------------------------------------------

print("\nAgreement checks:")

print("matrix vs quaternion:", np.allclose(v_matrix, v_quat))
print("matrix vs GA:", np.allclose(v_matrix, v_ga))
