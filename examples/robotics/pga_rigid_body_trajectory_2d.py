# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Rigid body trajectory
Algebra: 2D Projective Geometric Algebra (PGA)

Rigid body motion in the plane combines rotation and translation.

In classical robotics this is expressed using a 3x3 homogeneous
transformation matrix.

In PGA we instead use a *motor*:

    X' = M X M^{-1}

This example repeatedly applies a motor to a point,
producing a robot trajectory.
"""

import numpy as np

from amsa import Algebra

print("\n=== Rigid Body Trajectory (PGA Motor) ===")

alg = Algebra.pga2d()

# --------------------------------------------------
# motion parameters
# --------------------------------------------------

theta = np.deg2rad(10)
tx = 0.5
ty = 0.0

steps = 20

# --------------------------------------------------
# rotor (rotation)
# --------------------------------------------------

rotor = alg.multivector(
    {
        "e": np.cos(theta / 2),
        "e12": -np.sin(theta / 2),
    }
).normalized()

# --------------------------------------------------
# translator
# --------------------------------------------------

translator = alg.multivector(
    {
        "e": 1.0,
        "e01": -0.5 * ty,
        "e02": 0.5 * tx,
    }
)

# motor = translation * rotation
motor = translator * rotor

# --------------------------------------------------
# starting point
# --------------------------------------------------

point = alg.multivector(
    {
        "e01": 0.0,
        "e02": 0.0,
        "e12": 1.0,
    }
)

print("\nRobot trajectory:")

for i in range(steps):
    point = motor.sandwich(point)

    px = point.component("e01")
    py = point.component("e02")

    print(f"step {i + 1:02d} -> ({px:.3f}, {py:.3f})")
