# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: ProductIR inspection for understanding execution structure
Algebra: 2D Vector Geometric Algebra (VGA)

The ProductIR.show() method displays the storage-aware IR to understand
how operations are executed at the backend level.
"""

from amsa import Algebra
from amsa.ir import build_product_ir
from amsa.plans import plan_binary_product

print("\n=== ProductIR for Geometric Product ===")

alg = Algebra.vga2d()
lhs_layout = alg.grade_layout(1)
rhs_layout = alg.grade_layout(1)
plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")
ir = build_product_ir(plan, "dense", "dense")

print(ir.show(alg.spec))

print("\n=== ProductIR for Outer Product ===")

plan = plan_binary_product(lhs_layout, rhs_layout, "outer")
ir = build_product_ir(plan, "dense", "dense")

print(ir.show(alg.spec))

print("\n=== ProductIR with CSR Storage ===")

plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")
ir = build_product_ir(plan, "csr", "dense")

print(ir.show(alg.spec))
