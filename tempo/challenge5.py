"""
Challenge 5 — Sparse Support Explorer
"""

from amsa import Algebra
import numpy as np
import random
from collections import Counter

alg = Algebra.vga3d()

blade_pool = ["e1","e2","e3","e12","e13","e23","e123"]

counts = Counter()

for _ in range(10000):

    k1 = random.randint(1,3)
    k2 = random.randint(1,3)

    A = {random.choice(blade_pool):np.random.randn() for _ in range(k1)}
    B = {random.choice(blade_pool):np.random.randn() for _ in range(k2)}

    mvA = alg.multivector(A)
    mvB = alg.multivector(B)

    out = mvA * mvB

    print("lhs:",mvA.layout.blades,
          "rhs:",mvB.layout.blades,
          "out:",out.layout.blades)

    for b in out.layout.blades:
        counts[b]+=1

print("\nMost common output blades:")
print(counts)
