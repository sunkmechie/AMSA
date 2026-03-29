"""
Challenge 1 — 2D Signed Triangle Area Tool

Demonstrates oriented area using the bivector e12.

Outputs:
- edge vectors
- oriented bivector
- signed area
"""

import _bootstrap

from amsa import Algebra
from tests._utils import assert_allclose
import numpy as np

print("\n=== Challenge 1: 2D Signed Triangle Area ===")

alg = Algebra.vga2d()

# deterministic example
p = alg.vector([0,0])
q = alg.vector([5,0])
r = alg.vector([0,3])

u = q - p
v = r - p

biv = u ^ v
area = biv.component("e12") / 2

print("edge u:", u.values)
print("edge v:", v.values)
print("bivector:", biv.values)
print("area:", area)

# sanity check using determinant
det_area = 0.5*(5*3 - 0*0)

assert_allclose(area, det_area)

print("Deterministic example verified.")

print("\nRandom torture test")

for _ in range(10000):

    pts = np.random.randn(3,2)

    p = alg.vector(pts[0])
    q = alg.vector(pts[1])
    r = alg.vector(pts[2])

    u = q - p
    v = r - p

    biv = u ^ v
    area = biv.component("e12") / 2

    ux,uy = (pts[1]-pts[0])
    vx,vy = (pts[2]-pts[0])

    expected = 0.5*(ux*vy - uy*vx)

    assert_allclose(area, expected)

print("Random triangle tests passed.")
