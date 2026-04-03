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

Topic: Vector projection
Algebra: 2D Vector Geometric Algebra (VGA)

Robots often need to project positions onto lines
to compute distances from walls or corridors.

The inner product provides the projection magnitude.

Given vectors a and direction b:

    projection = (a | b) b

This example projects a robot position onto a corridor axis.
"""

from amsa import Algebra

print("\n=== Vector Projection ===")

alg = Algebra.vga2d()

robot_position = alg.vector([3.0, 2.0])
corridor_axis = alg.vector([1.0, 0.0]).normalized()

projection_scale = robot_position | corridor_axis
projection = projection_scale * corridor_axis

print("robot position:", robot_position.grade(1).as_dense().values)
print("corridor axis:", corridor_axis.grade(1).as_dense().values)
print("projection:", projection.grade(1).as_dense().values)
