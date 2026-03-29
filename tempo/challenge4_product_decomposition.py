import _bootstrap

from amsa import Algebra
from tests._utils import assert_mv_allclose
import numpy as np
import random

print("\n=== Challenge 4: Vector Product Decomposition ===")

alg = Algebra.vga3d()

pairs = 20000
for _ in range(pairs):
    u = alg.vector(np.random.randn(3))
    v = alg.vector(np.random.randn(3))

    gp = u * v
    ip = u | v
    op = u ^ v

    assert_mv_allclose(gp, ip+op)

print(f"Vector identity holds for {pairs} random pairs.")


# -----------------------------------------------------
# Test 2: Show identity does NOT hold for multivectors
# -----------------------------------------------------

print("\n--- Demonstrating non-generalizability ---")

def random_mv():
    choice = random.choice(["scalar", "vector", "bivector", "even"])

    if choice == "scalar":
        return alg.scalar(np.random.randn())

    if choice == "vector":
        return alg.vector(np.random.randn(3))

    if choice == "bivector":
        return alg.bivector(np.random.randn(3))

    if choice == "even":
        return alg.even(np.random.randn(4))


for _ in range(1000):

    A = random_mv()
    B = random_mv()

    gp = A * B
    ip = A | B
    op = A ^ B

    try:
        assert_mv_allclose(gp, ip + op, tol=1e-10)

    except AssertionError:

        print("\nIdentity fails for general multivectors (expected):")
        print("A grades:", A.grades)
        print("B grades:", B.grades)

        print("gp:", gp.as_dense().values)
        print("ip+op:", (ip + op).as_dense().values)

        break


print("\nConclusion:")
print("u*v = u|v + u^v holds for vectors,")
print("but does not generalize to arbitrary multivectors.")
