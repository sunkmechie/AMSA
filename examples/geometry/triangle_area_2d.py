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

Topic: Signed area of a triangle using the wedge product
Algebra: 2D Vector Geometric Algebra (VGA)

Given three points p, q, r in the plane we construct edge vectors:

    u = q - p
    v = r - p

The wedge product of two vectors produces an oriented area bivector:

    B = u ∧ v

In 2D vector geometric algebra the bivector basis is e12, so the result
can be written as:

    B = (u ∧ v) = A * e12

where A is the signed parallelogram area coefficient.

The signed triangle area is therefore:

    Area = (u ∧ v)_{e12} / 2

Interpretation of the sign:

    Area > 0  → counter-clockwise orientation
    Area < 0  → clockwise orientation
    Area = 0  → degenerate triangle (collinear points)

This example demonstrates three cases:

    1. A right triangle at the origin
    2. The same triangle with reversed orientation
    3. A skew triangle translated away from the origin

This illustrates two key properties of the wedge product:

    • antisymmetry:  u ∧ v = - (v ∧ u)
    • translation invariance: area depends only on edge vectors
"""

from amsa import Algebra

# select the algebra 
alg = Algebra.vga2d()

# === RIGHT TRAINGLE ===
print("\n--- Right triangle example ---")

# define the coordinate vectors of all the three points forming the triangle
p = alg.vector([0.0, 0.0])
q = alg.vector([5.0, 0.0])
r = alg.vector([0.0, 3.0])

u = q - p
v = r - p

# wedge product produces an oriented area bivector
area_bivector = u ^ v
area = area_bivector.component("e12") / 2.0

print(f"Area is {area}")

# === SAME RIGHT TRIANGLE, ORIENTED IN OPP DIRECTION===
print("\n--- Orientation flip ---")

flipped_area_bivector = v ^ u
area_flipped = flipped_area_bivector.component("e12") / 2.0

print(f"Flipped area is {area_flipped}")

# === SKEW TRIANGLE ===
print("\n --- Skew triangle example ---")

p = alg.vector([1.5, 2.0])
q = alg.vector([4.0, 3.0])
r = alg.vector([2.0, 7.0])

u = q - p
v = r - p

# wedge product
skew_area_bivector = u ^ v
area_skew = skew_area_bivector.component("e12") / 2.0
print(f"Area is {area_skew}")

