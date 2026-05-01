# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA geometry primitives overview
Algebra: 2D and 3D Conformal Geometric Algebra (CGA)

CGA embeds Euclidean geometry into a higher-dimensional conformal space.
The key null basis vectors are:

    n_o  = 0.5 (e_- - e_+)    null origin (squares to 0)
    n_inf = e_- + e_+          null infinity (squares to 0)
    n_o · n_inf = -1

All geometric primitives are blades in this algebra:

    point:   X = n_o + x + 0.5 x^2 n_inf           (null 1-vector)
    sphere:  S = C - 0.5 r^2 n_inf                  (dual, 1-vector)
    plane:   P = n + d n_inf                        (dual, 1-vector)
    line:    L = A ^ B ^ n_inf                      (direct, 3-vector)
    circle:  C = A ^ B ^ C                          (direct, 3-vector)

This example constructs each primitive and prints its blade decomposition.
"""

import amsa

print("\n=== CGA3D Primitives ===")

alg = amsa.Algebra.cga3d()

print(f"\nAlgebra: cga3d  |  signature: {alg.signature}  |  dimension: {alg.dimension}")

# -- Null basis --
print("\n--- Null basis vectors ---")
no = alg.origin()
ninf = alg.infinity()
print(f"n_o    = {no}")
print(f"n_inf  = {ninf}")
print(f"n_o^2  = {(no * no).component(0):.1f}")
print(f"n_inf^2 = {(ninf * ninf).component(0):.1f}")
print(f"n_o · n_inf = {(no.inner(ninf)).component(0):.1f}")

# -- Euclidean vector --
print("\n--- Euclidean vector ---")
x = alg.euclidean_vector([3.0, 0.0, 4.0])
print(f"x = {x}")
print(f"x^2 = {(x * x).component(0):.1f}  (should be 25)")

# -- Conformal point --
print("\n--- Conformal point ---")
p = alg.point([1.0, 2.0, 3.0])
print(f"P = {p}")
print(f"P^2 = {(p * p).component(0):.1f}  (should be 0, null)")

# -- Dual sphere --
print("\n--- Dual sphere ---")
s = alg.sphere([0.0, 0.0, 0.0], 2.0)
print(f"S = {s}")
print(f"S^2 = {(s * s).component(0):.1f}  (should be r^2 = 4)")

# -- Dual plane --
print("\n--- Dual plane ---")
plane = alg.plane([0.0, 0.0, 1.0], 2.0)
print(f"P = {plane}")
print(f"P^2 = {(plane * plane).component(0):.1f}  (should be |n|^2 = 1)")

# -- Direct line through two points --
print("\n--- Direct line (through two points) ---")
a = alg.point([0.0, 0.0, 0.0])
b = alg.point([3.0, 0.0, 0.0])
L = alg.line_through_points(a, b)
print(f"L = {L}")
print(f"|L|^2 = {abs((L * L).component(0)):.1f}  (should be distance^2 = 9)")

# -- Direct circle through three points --
print("\n--- Direct circle (through three points) ---")
c1 = alg.point([1.0, 0.0, 0.0])
c2 = alg.point([0.0, 1.0, 0.0])
c3 = alg.point([-1.0, 0.0, 0.0])
C = alg.circle_through_points(c1, c2, c3)
print(f"C = {C}")
print(f"|C|^2 = {abs((C * C).component(0)):.1f}  (should be 4 * r^2 = 4)")

# -- Translator --
print("\n--- Translator ---")
T = alg.translate([5.0, 0.0, 0.0])
print(f"T = {T}")

# -- Apply translator --
print("\n--- Apply translator to point ---")
p_orig = alg.point([0.0, 0.0, 0.0])
p_moved = amsa.sandwich(T, p_orig)
print(f"Original:  {p_orig}")
print(f"Translated: {p_moved}")
print(f"Expected:  {alg.point([5.0, 0.0, 0.0])}")

print("\n=== CGA2D Primitives ===\n")

alg2d = amsa.Algebra.cga2d()
print(f"\nAlgebra: cga2d  |  signature: {alg2d.signature}  |  dimension: {alg2d.dimension}")

no2 = alg2d.origin()
ninf2 = alg2d.infinity()
print(f"n_o    = {no2}")
print(f"n_inf  = {ninf2}")

p2 = alg2d.point([3.0, 4.0])
print(f"Point [3,4] = {p2}")
print(f"P^2 = {(p2 * p2).component(0):.1f}  (should be 0)")

T2 = alg2d.translate([2.0, 1.0])
p2_moved = amsa.sandwich(T2, p2)
print(f"Translated [3,4] → [5,5]: {p2_moved.to_layout(alg2d.point([5.0, 5.0]).layout).values}")
