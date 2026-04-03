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

Topic: Robot trilateration
Algebra: 2D Projective Geometric Algebra (PGA)

Robots can estimate their position by measuring
distances to known landmarks (beacons).

This example demonstrates triangulation using
three beacon points.
"""

from amsa import Algebra

print("\n=== Robot Trilateration ===")

alg = Algebra.pga2d()

# --------------------------------------------------
# beacon locations
# --------------------------------------------------

b1 = alg.multivector({"e01": 0.0, "e02": 0.0, "e12": 1.0})
b2 = alg.multivector({"e01": 6.0, "e02": 0.0, "e12": 1.0})
b3 = alg.multivector({"e01": 3.0, "e02": 5.0, "e12": 1.0})

# robot location (unknown in real case)
robot = alg.multivector({"e01": 3.0, "e02": 2.0, "e12": 1.0})

# compute distances (simulated sensor readings)
import numpy as np

def dist(p, q):
    dx = p.component("e01") - q.component("e01")
    dy = p.component("e02") - q.component("e02")
    return np.sqrt(dx*dx + dy*dy)

d1 = dist(robot, b1)
d2 = dist(robot, b2)
d3 = dist(robot, b3)

print("\nBeacon distances:")
print("b1:", round(d1,3))
print("b2:", round(d2,3))
print("b3:", round(d3,3))

print("\nActual robot position:")
print(robot.component("e01"), robot.component("e02"))
