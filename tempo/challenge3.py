"""
Challenge 3 — 3D orientation / signed volume.
"""

from amsa import Algebra
import numpy as np

alg = Algebra.vga3d()

u = np.random.randn(3)
v = np.random.randn(3)
w = np.random.randn(3)

mv_u = alg.vector(u)
mv_v = alg.vector(v)
mv_w = alg.vector(w)

triv = mv_u ^ mv_v ^ mv_w
volume = triv.component("e123")

print("u:",u)
print("v:",v)
print("w:",w)

print("trivector:",triv.values)
print("signed volume:",volume)
