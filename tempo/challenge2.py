"""
Challenge 2 — Batched orientation classification.
"""

import numpy as np
from amsa import Algebra

alg = Algebra.vga2d()

N = 1_000_000

p = np.random.randn(N,2)
q = np.random.randn(N,2)
r = np.random.randn(N,2)

u = q - p
v = r - p

u_mv = alg.vector(u)
v_mv = alg.vector(v)

signed = (u_mv ^ v_mv).component("e12")

ccw = np.sum(signed > 0)
cw = np.sum(signed < 0)
deg = np.sum(np.isclose(signed,0))

print("Triangles:",N)
print("CCW:",ccw)
print("CW:",cw)
print("Degenerate:",deg)
