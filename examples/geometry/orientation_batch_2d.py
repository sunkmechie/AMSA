# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Batched triangle orientation via the wedge product
Algebra: 2D Vector Geometric Algebra (VGA)

For three points p, q, r we construct edge vectors:

    u = q - p
    v = r - p

The wedge product produces an oriented area bivector:

    B = u ∧ v

In 2D VGA the bivector basis is e12, so:

    B = (u ∧ v) = A * e12

where A is the signed area coefficient.

The orientation test is therefore:

    signed = (u ∧ v)_{e12}

Interpretation:

    signed > 0  → counter-clockwise triangle
    signed < 0  → clockwise triangle
    signed = 0  → degenerate (collinear points)

This example demonstrates batched orientation testing for many triangles
using AMSA's vectorized multivector operations.
"""

import numpy as np

from amsa import Algebra

print("\n=== Batched Triangle Orientation ===")

# reproducible randomness for example
np.random.seed(0)

alg = Algebra.vga2d()

# small batch for demonstration
N = 10

# random triangle vertices
p = np.random.randn(N, 2)
q = np.random.randn(N, 2)
r = np.random.randn(N, 2)

# edge vectors
u = q - p
v = r - p

# lift into AMSA multivectors
u_mv = alg.vector(u)
v_mv = alg.vector(v)

# wedge product gives oriented area bivector
# sign of the e12 coefficient determines orientation
signed = (u_mv ^ v_mv).component("e12")

# classify orientation
ccw = np.sum(signed > 0)
cw = np.sum(signed < 0)
degenerate = np.sum(np.isclose(signed, 0))

# show a few signed area values
print("\nSigned wedge coefficients (first 5):")
print(signed[:5])

print(f"\nTriangles generated: {N}")
print(f"Counter-clockwise: {ccw}")
print(f"Clockwise: {cw}")
print(f"Degenerate: {degenerate}")
