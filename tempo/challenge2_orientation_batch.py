import _bootstrap

from amsa import Algebra
from tests._utils import assert_allclose
import numpy as np

print("\n=== Challenge 2: Batched Orientation ===")

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

# numpy determinant
det = u[:,0]*v[:,1] - u[:,1]*v[:,0]

assert_allclose(signed, det)

ccw = np.sum(signed > 0)
cw = np.sum(signed < 0)
deg = np.sum(np.isclose(signed,0))

print("Triangles:", N)
print("CCW:", ccw)
print("CW:", cw)
print("Degenerate:", deg)

print("Batch orientation test passed.")
