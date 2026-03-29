import _bootstrap

from amsa import Algebra
import numpy as np
import random
from collections import Counter

print("\n=== Challenge 5: Sparse Support Explorer ===")

alg = Algebra.vga3d()

blades = ["e1","e2","e3","e12","e13","e23","e123"]

counts = Counter()

lhs_support_sizes = []
rhs_support_sizes = []
out_support_sizes = []
pair_product_sizes = []

N = 10000
for _ in range(N):
    k1 = random.randint(1,3)
    k2 = random.randint(1,3)

    A = {random.choice(blades): np.random.randn() for _ in range(k1)}
    B = {random.choice(blades): np.random.randn() for _ in range(k2)}

    mvA = alg.multivector(A)
    mvB = alg.multivector(B)

    out = mvA * mvB

    lhs = mvA.layout.blades
    rhs = mvB.layout.blades
    res = out.layout.blades

    print("lhs:", lhs,
          "rhs:", rhs,
          "out:", res)

    # blade frequency
    for b in res:
        counts[b] += 1

    # support metrics
    lhs_support_sizes.append(len(lhs))
    rhs_support_sizes.append(len(rhs))
    out_support_sizes.append(len(res))

    pair_product_sizes.append(len(lhs) * len(rhs))


print("\n=== Blade Frequency ===")
print(counts)


print("\n=== Support Size Metrics ===")

avg_lhs = np.mean(lhs_support_sizes)
avg_rhs = np.mean(rhs_support_sizes)
avg_out = np.mean(out_support_sizes)

print("avg |support(A)| :", avg_lhs)
print("avg |support(B)| :", avg_rhs)
print("avg |support(A*B)| :", avg_out)


print("\n=== Support Explosion Metric ===")

avg_pair_products = np.mean(pair_product_sizes)

explosion_factor = avg_out / avg_pair_products

print("avg pair blade products:", avg_pair_products)
print("avg resulting support:", avg_out)
print("support explosion factor:", explosion_factor)


print("\nInterpretation:")

print("""
If explosion_factor << 1:
    sparse multiplication stays sparse (good for AMSA design)

If explosion_factor ~ 1:
    multiplication becomes dense quickly

If explosion_factor > 1:
    something is very wrong
""")
