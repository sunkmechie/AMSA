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

rotor = alg.multivector({
    "e": np.cos(theta / 2),
    "e12": -np.sin(theta / 2),
}).normalized()

# --------------------------------------------------
# translator
# --------------------------------------------------

translator = alg.multivector({
    "e": 1.0,
    "e01": -0.5 * ty,
    "e02": 0.5 * tx,
})

# motor = translation * rotation
motor = translator * rotor

# --------------------------------------------------
# starting point
# --------------------------------------------------

point = alg.multivector({
    "e01": 0.0,
    "e02": 0.0,
    "e12": 1.0,
})

print("\nRobot trajectory:")

for i in range(steps):

    point = motor.sandwich(point)

    px = point.component("e01")
    py = point.component("e02")

    print(f"step {i+1:02d} -> ({px:.3f}, {py:.3f})")
