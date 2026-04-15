# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Corridor corner detection
Algebra: 2D Projective Geometric Algebra (PGA)

In projective geometric algebra, lines intersect using
the regressive product.

Two lines:

    l1
    l2

produce their intersection point:

    P = l1 ∨ l2

This is useful in robotics when detecting corners
from wall boundaries.
"""

from amsa import Algebra

print("\n=== Corridor Corner Detection ===")

alg = Algebra.pga2d()

left_wall = alg.multivector({"e01": 1.0, "e12": -1.0})
front_wall = alg.multivector({"e02": 1.0, "e12": -2.0})

corner = left_wall.regress(front_wall)

print("left wall:", left_wall.as_dense().values)
print("front wall:", front_wall.as_dense().values)

print("corner point:", corner.as_dense().values)
print("corner bulk part:", corner.bulk().as_dense().values)
print("corner weight part:", corner.weight().as_dense().values)
