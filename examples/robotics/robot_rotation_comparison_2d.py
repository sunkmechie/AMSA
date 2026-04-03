# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])

v_matrix = R @ v

print("\nMatrix rotation:")
print(v_matrix)

# --------------------------------------------------
# Quaternion rotation (2D embedded in Z rotation)
# --------------------------------------------------

x, y = v

v_quat = np.array([
    np.cos(theta)*x - np.sin(theta)*y,
    np.sin(theta)*x + np.cos(theta)*y,
])

print("\nQuaternion rotation:")
print(v_quat)

# --------------------------------------------------
# Geometric algebra rotor
# --------------------------------------------------

alg = Algebra.vga2d()

vec = alg.vector([1.0, 0.0])

rotor = alg.multivector({
    "e": np.cos(theta / 2),
    "e12": -np.sin(theta / 2),
}).normalized()

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
