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

Topic: Small geometry utilities implemented with geometric algebra

This example demonstrates how common geometric computations can be
written directly using multivector operations instead of matrices.

Functions implemented:

    signed_area_2d(a, b, c)
    signed_volume_3d(u, v, w)
    are_orthogonal(u, v)
    bivector_plane(u, v)

These operations use fundamental GA formulas:

Triangle area:
    Area = ( (q - p) ∧ (r - p) )_{e12} / 2

Signed volume:
    Volume = (u ∧ v ∧ w)_{e123}

Orthogonality:
    u ⨼ v = 0
"""

from amsa import Algebra
import numpy as np

alg2 = Algebra.vga2d()
alg3 = Algebra.vga3d()


def signed_area_2d(a, b, c):

    p = alg2.vector(a)
    q = alg2.vector(b)
    r = alg2.vector(c)

    return ((q - p) ^ (r - p)).component("e12") / 2


def signed_volume_3d(u, v, w):

    return (alg3.vector(u) ^ alg3.vector(v) ^ alg3.vector(w)).component("e123")


def are_orthogonal(u, v, tol=1e-9):

    u = alg3.vector(u)
    v = alg3.vector(v)

    return abs((u | v).component("e")) < tol


def bivector_plane(u, v):

    return alg3.vector(u) ^ alg3.vector(v)


print("\n=== Mini Geometry Kernel ===")

a = np.array([0.0, 0.0])
b = np.array([3.0, 0.0])
c = np.array([0.0, 4.0])

print("Triangle area:", signed_area_2d(a, b, c))

u = np.array([1.0, 0.0, 0.0])
v = np.array([0.0, 1.0, 0.0])
w = np.array([0.0, 0.0, 1.0])

print("Signed volume:", signed_volume_3d(u, v, w))
print("Are u and v orthogonal?", are_orthogonal(u, v))
