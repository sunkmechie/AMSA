# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA versor actions — classify before and after geometric operations
Algebra: 2D and 3D Conformal Geometric Algebra (CGA)

CGA versors (translators, planes) act on multivectors via the sandwich
product ``sandwich(versor, target)``.  ``alg.classify()`` confirms what
the result is — a point with updated coordinates, still null, possibly
unnormalized.

This example demonstrates:
    - translation: verify the point moves to the expected location
    - reflection: verify the reflected coordinates match the plane
"""

import amsa
from amsa import Algebra

print("\n=== CGA Versor Actions ===\n")

# ---- 3D: translate ----------------------------------------------------------

print("--- 3D: Translate ---\n")

alg = Algebra.cga3d()

T = alg.translate([5.0, -1.0, 3.0])
print("Translator:")
print(alg.classify(T))

p = alg.point([1.0, 2.0, 3.0])
print("Point before:")
print(alg.classify(p))

p_moved = amsa.sandwich(T, p)
print("Point after sandwich(T, P):")
print(alg.classify(p_moved))

# ---- 3D: reflect ------------------------------------------------------------

print("--- 3D: Reflect in plane z=2 ---\n")

plane = alg.plane([0.0, 0.0, 1.0], 2.0)
print("Plane:")
print(alg.classify(plane))

q = alg.point([1.0, 2.0, 3.0])
print("Point before:")
print(alg.classify(q))

q_reflected = amsa.sandwich(plane, q)
print("Point after sandwich(plane, Q):")
print(alg.classify(q_reflected))

# ---- 2D: translate ----------------------------------------------------------

print("--- 2D: Translate ---\n")

alg2 = Algebra.cga2d()

T2 = alg2.translate([10.0, -5.0])
print("Translator:")
print(alg2.classify(T2))

p2 = alg2.point([0.0, 0.0])
print("Point before:")
print(alg2.classify(p2))

p2_moved = amsa.sandwich(T2, p2)
print("Point after sandwich(T, P):")
print(alg2.classify(p2_moved))

# ---- 2D: reflect ------------------------------------------------------------

print("--- 2D: Reflect in plane y=3 ---\n")

plane2 = alg2.plane([0.0, 1.0], 3.0)
print("Plane:")
print(alg2.classify(plane2))

q2 = alg2.point([4.0, 8.0])
print("Point before:")
print(alg2.classify(q2))

q2_reflected = amsa.sandwich(plane2, q2)
print("Point after sandwich(plane, Q):")
print(alg2.classify(q2_reflected))
