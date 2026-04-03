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

Topic: Plane reflections
Algebra: 3D Vector Geometric Algebra (VGA)

In geometric algebra, reflecting a vector v across a plane
with unit normal n is performed using the sandwich product:

    v' = - n v n

Reflections are fundamental operations in:

    physics simulation
    ray tracing
    collision response
    robotics contact modeling

This example reflects a velocity vector across the
three coordinate planes:

    YZ plane (normal e1)
    XZ plane (normal e2)
    XY plane (normal e3)
"""

from amsa import Algebra

print("\n=== Plane Reflections ===")

alg = Algebra.vga3d()

velocity = alg.vector([1.0, -2.0, 0.5])

nx = alg.vector([1.0, 0.0, 0.0]).normalized()  # YZ plane
ny = alg.vector([0.0, 1.0, 0.0]).normalized()  # XZ plane
nz = alg.vector([0.0, 0.0, 1.0]).normalized()  # XY plane

reflect_x = -nx.sandwich(velocity)
reflect_y = -ny.sandwich(velocity)
reflect_z = -nz.sandwich(velocity)

print("original velocity:", velocity.grade(1).as_dense().values)

print("\nReflection across YZ plane:")
print(reflect_x.grade(1).as_dense().values)

print("\nReflection across XZ plane:")
print(reflect_y.grade(1).as_dense().values)

print("\nReflection across XY plane:")
print(reflect_z.grade(1).as_dense().values)