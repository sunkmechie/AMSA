# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Batched pairwise distance via the CGA inner product
Algebra: 3D Conformal Geometric Algebra (CGA)

AMSA's CGA embeds Euclidean geometry into Clifford algebra. Points become
null vectors, and distance is recovered from the inner product:

    d(A, B)^2 = -2 (A · B)

The same inner product expresses the metric between many geometric types:

    object      |  inner product meaning
    ------------+----------------------------------------
    A · B       |  point-to-point metric
    A · S       |  point-to-sphere metric
    A · P       |  point-to-plane metric (signed)

No separate formulas, no switching between coordinate systems. AMSA's
``alg.distance_squared()`` applies this uniformly with batch broadcasting.

This example:
    1. Constructs a batch of 3D conformal points
    2. Computes pairwise distance matrix via the CGA inner product
    3. Shows the same result via Euclidean coordinate subtraction
"""

import numpy as np

import amsa

print("\n=== Batched CGA Point Distance ===\n")

np.random.seed(42)

N = 5
alg = amsa.Algebra.cga3d()

coords = np.random.randn(N, 3)
for i in range(N):
    print(f"  p{i} = [{coords[i, 0]:7.3f}, {coords[i, 1]:7.3f}, {coords[i, 2]:7.3f}]")

# -- CGA: points as blades, distance from the inner product --------------------
print("\n--- CGA pairwise distance matrix ---")

points = [alg.point(coords[i]) for i in range(N)]
dense_layout = points[0].as_dense().layout
all_vals = np.stack([p.as_dense().values for p in points], axis=0)
cga_points = amsa.MVArray.from_array(alg.spec, dense_layout, all_vals)

dist2_via_cga = alg.distance_squared(
    cga_points[:, np.newaxis],
    cga_points[np.newaxis, :],
)

print(f"\n  d(A, B)^2 = -2 (A . B)  —  {N} x {N} via broadcast inner product:\n")
for i in range(N):
    row_str = "  ".join(f"{dist2_via_cga[i, j]:7.2f}" for j in range(N))
    print(f"  [{row_str}]")

# -- Euclidean: coordinate subtraction (familiar check) ------------------------
print("\n--- Euclidean coordinate subtraction (same result) ---")

diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2_euclidean = np.sum(diffs * diffs, axis=-1)

print("\n  d^2 = (x_A - x_B)^2 + (y_A - y_B)^2 + (z_A - z_B)^2:\n")
for i in range(N):
    row_str = "  ".join(f"{dist2_euclidean[i, j]:7.2f}" for j in range(N))
    print(f"  [{row_str}]")

assert np.allclose(dist2_via_cga, dist2_euclidean, atol=1e-10)

# -- Why CGA: same operation, different objects ---------------------------------
print("\n--- Why Clifford? Unified metric across entity types ---")
print("""
  The inner product A . B is the *universal* CGA metric operator.
  Changing what A and B represent changes the geometric meaning,
  but the algebraic operation stays the same:

      A · B   for point A, point B   →  point-to-point metric
      A · P   for point A, plane P   →  signed point-to-plane metric
      A · S   for point A, sphere S  →  point-to-sphere metric

  No separate formulas. No switching between point math, sphere math,
  and plane math. The algebra carries the geometry.
""")

plane = alg.plane([0.0, 0.0, 1.0], 2.0)
q = alg.point([0.5, 0.5, 3.0])
sq_dist = (q.inner(plane)).component(0) ** 2 / (plane.inner(plane)).component(0)
expected = 1.0
print("  Plane: normal [0,0,1], signed distance from origin = 2.0")
print(f"  Point: [{0.5}, {0.5}, {3.0}]")
print(f"    CGA: (Q . P)^2 / (P . P) = {sq_dist:.4f}")
print(f"    Expected: |z - d|^2 = |3 - 2|^2 = {expected:.1f}")
