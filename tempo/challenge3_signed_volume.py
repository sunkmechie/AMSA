import _bootstrap

from amsa import Algebra
from tests._utils import assert_allclose
import numpy as np

print("\n=== Challenge 3: 3D Signed Volume ===")
alg = Algebra.vga3d()

for _ in range(10000):
    u = np.random.randn(3)
    v = np.random.randn(3)
    w = np.random.randn(3)

    mv_u = alg.vector(u)
    mv_v = alg.vector(v)
    mv_w = alg.vector(w)

    triv = mv_u ^ mv_v ^ mv_w
    volume = triv.component("e123")

    expected = np.linalg.det(np.stack([u, v, w]))

    assert_allclose(volume, expected, tol=1e-12)

print("Signed volume tests passed, 10000 trails, 1e-12 tolerance")
