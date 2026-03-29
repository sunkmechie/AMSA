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

Topic: Distance from a point to a plane using geometric algebra
Algebra: 3D Vector Geometric Algebra (VGA)

A plane can be defined by two spanning vectors:

    u, v

The wedge product produces the plane bivector:

    B = u ∧ v

The dual of the bivector gives the plane normal:

    n = dual(B)

For a point p and a point p0 on the plane, the signed distance is:

    distance = ( (p − p0) · n ) / ||n||

This example shows how geometric algebra represents planes and
computes point-to-plane distance using multivector operations.
"""

from amsa import Algebra
import numpy as np

print("\n=== Point to Plane Distance ===")

alg = Algebra.vga3d()

# plane origin
p0 = alg.vector([0.0, 0.0, 0.0])

# plane spanning vectors
u = alg.vector([1.0, 0.0, 0.0])
v = alg.vector([0.0, 1.0, 0.0])

# plane bivector
B = u ^ v

# TODO: replace with B.dual() once dual() is implemented
n_vec = np.cross(u.values, v.values)
n = alg.vector(n_vec)
#n = B.dual()

# point above the plane
p = alg.vector([0.5, 0.5, 2.0])

# displacement
d = p - p0

# signed distance
distance = (d | n).component("e") / np.linalg.norm(n.values)

print("Plane bivector:", B.values)
print("Plane normal:", n.values)
print("Point:", p.values)

print(f"\nSigned distance to plane: {distance}")
