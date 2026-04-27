# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: OpPlan inspection for understanding product structure
Algebra: 2D Vector Geometric Algebra (VGA)

The OpPlan.show() method displays product plans to understand which
blades are produced and how they are computed.
"""

from amsa import Algebra
from amsa.plans import plan_binary_product

print("\n=== Geometric Product (vector * vector) ===")

alg = Algebra.vga2d()
lhs_layout = alg.grade_layout(1)
rhs_layout = alg.grade_layout(1)
plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")

print(plan.show())

print("\n=== Outer Product (vector * vector) ===")

plan = plan_binary_product(lhs_layout, rhs_layout, "outer")

print(plan.show())

print("\n=== Inner Product (vector * vector) ===")

plan = plan_binary_product(lhs_layout, rhs_layout, "inner")

print(plan.show())

print("\n=== Geometric Product (bivector * bivector) ===")

biv_layout = alg.grade_layout(2)
plan = plan_binary_product(biv_layout, biv_layout, "geometric")

print(plan.show())
