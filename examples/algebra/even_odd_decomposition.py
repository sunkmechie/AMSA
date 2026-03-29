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

Topic: Even and odd grade decomposition
Algebra: 3D Vector Geometric Algebra (VGA)

Any multivector M can be decomposed into even and odd grade parts:

    M = M_even + M_odd

Even grades include:

    grade 0 (scalar)
    grade 2 (bivector)

Odd grades include:

    grade 1 (vector)
    grade 3 (trivector)

This decomposition is fundamental in geometric algebra because
rotors and many geometric transformations live in the even subalgebra.
"""

from amsa import Algebra

print("\n=== Even / Odd Decomposition ===")

alg = Algebra.vga3d()

mv = alg.multivector({
    "e1": 1.0,
    "e2": -0.5,
    "e12": 2.0,
    "e123": 3.0
})

even = mv.grade(0, 2)
odd = mv.grade(1, 3)

recomposed = even + odd

print("Original multivector:", mv.values)
print("Even part:", even.values)
print("Odd part:", odd.values)

print("\nRecomposition:")
print(recomposed.values)
