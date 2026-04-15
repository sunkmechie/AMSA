# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0


"""
AMSA Example

Topic: Planar robot heading update
Algebra: 2D Vector Geometric Algebra (VGA)

Mobile robots often rotate in the plane while maintaining
a forward body direction.

Rotations in geometric algebra are represented by rotors.

A rotor R rotates a vector v using the sandwich product:

    v' = R v R^{-1}

In this example we rotate the robot's forward axis by 30 degrees.
"""

import numpy as np

from amsa import Algebra

print("\n=== Planar Heading Update ===")

alg = Algebra.vga2d()

forward_body = alg.vector([1.0, 0.0])

theta = np.deg2rad(30)

rotor = alg.multivector({"e": np.cos(theta / 2), "e12": -np.sin(theta / 2)}).normalized()

forward_world = rotor.sandwich(forward_body)

print("rotor:", rotor.as_dense().values)
print("forward axis (world):", forward_world.grade(1).as_dense().values)
