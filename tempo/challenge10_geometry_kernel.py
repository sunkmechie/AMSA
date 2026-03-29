"""
Challenge 10 — Mini Geometry Kernel Without Matrices

Implements small reusable geometric functions using AMSA.
"""

import _bootstrap

from amsa import Algebra
from tests._utils import assert_allclose
import numpy as np

print("\n=== Challenge 10: Mini Geometry Kernel ===")

alg2 = Algebra.vga2d()
alg3 = Algebra.vga3d()


def signed_area_2d(a,b,c):

    p = alg2.vector(a)
    q = alg2.vector(b)
    r = alg2.vector(c)

    return ( (q-p) ^ (r-p) ).component("e12") / 2


def signed_volume_3d(u,v,w):

    return (alg3.vector(u) ^ alg3.vector(v) ^ alg3.vector(w)).component("e123")


def are_orthogonal(u,v,tol=1e-9):

    u = alg3.vector(u)
    v = alg3.vector(v)

    return abs((u | v).component("e")) < tol


def bivector_plane(u,v):

    return alg3.vector(u) ^ alg3.vector(v)


print("\nTesting kernels")

for _ in range(2000):

    pts = np.random.randn(3,2)

    area = signed_area_2d(*pts)

    ux,uy = pts[1]-pts[0]
    vx,vy = pts[2]-pts[0]

    expected = 0.5*(ux*vy - uy*vx)

    assert_allclose(area, expected, tol=1e-12)

print("Area kernel verified")


print("\nOrthogonality check")

u = np.array([1,0,0])
v = np.array([0,1,0])

assert are_orthogonal(u,v)

print("Orthogonality test passed")
