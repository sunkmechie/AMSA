# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0


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
