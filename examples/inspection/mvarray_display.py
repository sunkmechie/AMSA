# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: MVArray display and pretty-printing
Algebra: 2D Vector Geometric Algebra (VGA)

The MVArray.__repr__ method provides human-readable multivector
representations for debugging and understanding multivector structure.
"""

from amsa import Algebra

print("\n=== Simple Vector ===")

alg = Algebra.vga2d()
u = alg.vector([1.0, 2.0])

print("u =", u)

print("\n=== Bivector ===")

v = alg.bivector([3.0])

print("v =", v)

print("\n=== Mixed Grades ===")

w = alg.multivector({0: 1.0, 1: 2.0, 3: 3.0})

print("w =", w)

print("\n=== Zero Multivector ===")

z = alg.zeros()

print("z =", z)

print("\n=== Batched Multivector ===")

batch = alg.zeros(batch_shape=(2, 3))

print("batch =", batch)
