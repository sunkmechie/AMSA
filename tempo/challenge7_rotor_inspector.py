"""
Challenge 7 — Rotor-Like Element Inspector

Explores even multivectors that behave like rotors.
Checks that R * ~R is approximately scalar.
"""

import _bootstrap

from amsa import Algebra
from tests._utils import assert_allclose
import numpy as np

print("\n=== Challenge 7: Rotor-Like Inspector ===")

alg = Algebra.vga3d()

for _ in range(5000):
    coeffs = np.random.randn(4)
    
    R = alg.even(coeffs)
    R_rev = R.reverse()

    metric = R * R_rev
    dense = metric.as_dense()

    scalar = dense.component("e")
    others = dense.values.copy()
    others[0] = 0

    assert_allclose(others, 0.0, tol = 1e-15)

print("Rotor-like elements collapse to scalar norm")
    
