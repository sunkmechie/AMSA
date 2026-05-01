# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA classification overview — every primitive through the classifier
Algebra: 2D and 3D Conformal Geometric Algebra (CGA)

``alg.classify(mv)`` inspects a multivector and returns an
:class:`~amsa.algebra.EntityInfo` with its geometric interpretation,
invariants, and storage metadata.

This example constructs every CGA primitive (cga3d and cga2d) and
prints the classification output for each.
"""

import amsa

print("\n=== CGA3D Classification Overview ===\n")

alg = amsa.Algebra.cga3d()

entities = [
    ("origin", alg.origin()),
    ("point at infinity", alg.infinity()),
    ("normalized point [1, 2, 3]", alg.point([1.0, 2.0, 3.0])),
    ("dual sphere (center [1,0,0], r=3)", alg.sphere([1.0, 0.0, 0.0], 3.0)),
    ("dual plane (n=[0,0,1], d=2)", alg.plane([0.0, 0.0, 1.0], 2.0)),
    ("direct line through [0,0,0]→[1,0,0]", alg.line_through_points(
        alg.point([0.0, 0.0, 0.0]),
        alg.point([1.0, 0.0, 0.0]),
    )),
    ("direct circle through [1,0,0],[0,1,0],[-1,0,0]", alg.circle_through_points(
        alg.point([1.0, 0.0, 0.0]),
        alg.point([0.0, 1.0, 0.0]),
        alg.point([-1.0, 0.0, 0.0]),
    )),
    ("translator [1, 2, 3]", alg.translate([1.0, 2.0, 3.0])),
    ("scalar 5.0", alg.scalar(5.0)),
    ("euclidean vector [4, 5, 6]", alg.euclidean_vector([4.0, 5.0, 6.0])),
]

for label, mv in entities:
    print(f"--- {label} ---")
    print(alg.classify(mv))
    print()

print("=== CGA2D Classification Overview ===\n")

alg2 = amsa.Algebra.cga2d()

entities_2d = [
    ("origin", alg2.origin()),
    ("point at infinity", alg2.infinity()),
    ("point [3, 4]", alg2.point([3.0, 4.0])),
    ("dual sphere (center [1,1], r=2)", alg2.sphere([1.0, 1.0], 2.0)),
    ("dual plane (n=[0,1], d=3)", alg2.plane([0.0, 1.0], 3.0)),
    ("translator [5, -1]", alg2.translate([5.0, -1.0])),
]

for label, mv in entities_2d:
    print(f"--- {label} ---")
    print(alg2.classify(mv))
    print()
