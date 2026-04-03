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

Topic: Circular robot motion
Algebra: 2D Projective Geometric Algebra (PGA)

A differential-drive robot moving forward while turning
at a constant rate follows a circular trajectory.

In projective geometric algebra we can represent this
motion using a single motor:

    M = translator * rotor

Applying the motor repeatedly evolves the robot pose.
"""

import numpy as np
from amsa import Algebra

print("\n=== Circular Robot Motion (PGA Motor) ===")

alg = Algebra.pga2d()

# --------------------------------------------------
# motion parameters
# --------------------------------------------------

theta = np.deg2rad(10)     # turn rate
forward_step = 0.5         # forward movement per step

steps = 36                 # full circle approx

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
    "e02": 0.5 * forward_step,
})

motor = translator * rotor

# --------------------------------------------------
# starting point (robot position)
# --------------------------------------------------

point = alg.multivector({
    "e01": 0.0,
    "e02": 0.0,
    "e12": 1.0,
})

print("\nRobot path:")

for i in range(steps):

    point = motor.sandwich(point)

    x = point.component("e01")
    y = point.component("e02")

    print(f"step {i+1:02d} -> ({x:.3f}, {y:.3f})")
