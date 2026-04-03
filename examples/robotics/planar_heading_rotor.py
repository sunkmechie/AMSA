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

rotor = alg.multivector({
    "e": np.cos(theta / 2),
    "e12": -np.sin(theta / 2)
}).normalized()

forward_world = rotor.sandwich(forward_body)

print("rotor:", rotor.as_dense().values)
print("forward axis (world):", forward_world.grade(1).as_dense().values)
