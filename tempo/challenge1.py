"""
Challenge 1 — 2D Signed Triangle Area Tool

Demonstrates oriented area using the bivector e12.

Outputs:
- edge vectors
- oriented bivector
- signed area
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from amsa import Algebra
import numpy as np

alg = Algebra.vga2d()

print("\n=== Deterministic Example ===")

p = alg.vector([0, 0])
q = alg.vector([5, 0])
r = alg.vector([0, 3])

u = q - p
v = r - p

biv = u ^ v
area = biv.component("e12") / 2

print("p:", p.values)
print("q:", q.values)
print("r:", r.values)

print("edge u:", u.values)
print("edge v:", v.values)

print("bivector:", biv.values)
print("signed area:", area)

print("\n=== Random Torture Test ===")

for _ in range(10):
    p = alg.vector(np.random.randn(2))
    q = alg.vector(np.random.randn(2))
    r = alg.vector(np.random.randn(2))

    u = q - p
    v = r - p

    biv = u ^ v
    area = biv.component("e12") / 2

    print("area:", area)
