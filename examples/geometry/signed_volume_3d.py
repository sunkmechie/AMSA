# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AMSA Example

Topic: Signed volume of a parallelepiped using the outer product
Algebra: 3D Vector Geometric Algebra (VGA)

Given three vectors u, v, w in 3D space we compute the trivector:

    T = u ∧ v ∧ w

In 3D VGA the highest grade basis element is e123, so:

    T = V * e123

where V is the signed volume coefficient.

Thus the signed volume is:

    Volume = (u ∧ v ∧ w)_{e123}

Interpretation:

    Volume > 0  → right-handed orientation
    Volume < 0  → left-handed orientation
    Volume = 0  → coplanar vectors

This example demonstrates the relationship between the GA trivector
and the classical determinant.
"""

from amsa import Algebra
import numpy as np

print("\n=== Signed Volume Example ===")

alg = Algebra.vga3d()

# example vectors
u = alg.vector([1.0, 0.0, 0.0])
v = alg.vector([0.0, 2.0, 0.0])
w = alg.vector([0.0, 0.0, 3.0])

trivector = u ^ v ^ w
volume = trivector.component("e123")

print("u:", u.values)
print("v:", v.values)
print("w:", w.values)

print("\nTrivector:", trivector.values)
print(f"Signed volume: {volume}")

