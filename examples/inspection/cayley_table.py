# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Cayley table inspection for understanding algebra structure
Algebra: Various (VGA2d, PGA2d, VGA3d)

The Algebra.show_cayley() method displays Cayley table subsets to
understand blade multiplication rules in the algebra.
"""

from amsa import Algebra

print("\n=== VGA2d Cayley Table (default subset) ===")

alg = Algebra.vga2d()

print(alg.show_cayley())

print("\n=== VGA2d Cayley Table (custom blades) ===")

print(alg.show_cayley(blades=(0, 1, 2)))

print("\n=== PGA2d Cayley Table (first 8 blades) ===")

alg = Algebra.pga2d()

print(alg.show_cayley())

print("\n=== VGA3d Cayley Table (first 8 blades) ===")

alg = Algebra.vga3d()

print(alg.show_cayley())
