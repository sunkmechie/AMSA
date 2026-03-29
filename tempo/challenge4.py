"""
Challenge 4 — Vector product decomposition.

Verifies:

uv = u·v + u∧v
"""
import sys
from pathlib import Path

# add project root to Python path BEFORE imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from amsa import Algebra
import numpy as np
from tests._utils import assert_mv_allclose

alg = Algebra.vga3d()

print("\n=== Deterministic Example ===")

u = alg.vector([1,2,3])
v = alg.vector([-1,4,2])

gp = u * v
ip = u | v
op = u ^ v

print("u:",u.values)
print("v:",v.values)

print("gp:",gp.values)
print("ip:",ip.values)
print("op:",op.values)

assert_mv_allclose(gp, ip + op)

print("Decomposition verified.")

print("\n=== Random Torture Test ===")

for i in range(10000):

    u = alg.vector(np.random.randn(3))
    v = alg.vector(np.random.randn(3))

    gp = u * v
    ip = u | v
    op = u ^ v

    assert_mv_allclose(gp, ip + op)

print("Random vector tests passed.")
