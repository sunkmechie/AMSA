"""
Challenge 8 — PGA Null-Blade Behaviour

Examines algebra behaviour involving e0 in PGA.
"""

import _bootstrap

from amsa import Algebra
from tests._utils import assert_allclose

print("\n=== Challenge 8: PGA Null Blade Behaviour ===")

alg = Algebra.pga2d()

e0 = alg.blade("e0")
e1 = alg.blade("e1")

print("\ne0 * e0")
print((e0 * e0).layout.blades)

print("\ne0 ^ e1")
print((e0 ^ e1).values)

print("\ne0 | e0")
print((e0 | e0).values)

# expected property in projective GA
metric = e0 * e0

assert_allclose(metric.values, 0.0, tol=1e-12)

print("\nVerified: e0 is null (e0*e0 ≈ 0)")
