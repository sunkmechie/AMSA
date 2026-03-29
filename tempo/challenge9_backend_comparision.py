"""
Challenge 9 — Backend Comparison Harness

Runs identical workloads under dense and CSR backends.
Reports equality, support patterns, and execution time.
"""

import _bootstrap

from amsa import Algebra
from tests._utils import assert_mv_allclose
import numpy as np
import time

print("\n=== Challenge 9: Backend Comparison ===")

alg = Algebra.vga3d()

data_lhs = {"e1":1.0,"e2":2.0,"e23":-0.5}
data_rhs = {"e3":1.2,"e12":0.3}

lhs_dense = alg.multivector(data_lhs, backend="dense")
rhs_dense = alg.multivector(data_rhs, backend="dense")

lhs_csr = alg.multivector(data_lhs, backend="csr")
rhs_csr = alg.multivector(data_rhs, backend="csr")

print("\nDense * Dense")

t0 = time.perf_counter()
out_dd = lhs_dense * rhs_dense
t1 = time.perf_counter()

print("support:", out_dd.layout.blades)
print("time:", t1 - t0)

print("\nCSR * CSR")

t0 = time.perf_counter()
out_cc = lhs_csr * rhs_csr
t1 = time.perf_counter()

print("support:", out_cc.layout.blades)
print("time:", t1 - t0)

print("\nMixed")

out_dc = lhs_dense * rhs_csr
out_cd = lhs_csr * rhs_dense

assert_mv_allclose(out_dd, out_cc, tol=1e-12)
assert_mv_allclose(out_dd, out_dc, tol=1e-12)
assert_mv_allclose(out_dd, out_cd, tol=1e-12)

print("\nBackends produce identical results.")
