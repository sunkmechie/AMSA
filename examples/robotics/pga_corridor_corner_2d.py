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

Topic: Corridor corner detection
Algebra: 2D Projective Geometric Algebra (PGA)

In projective geometric algebra, lines intersect using
the regressive product.

Two lines:

    l1
    l2

produce their intersection point:

    P = l1 ∨ l2

This is useful in robotics when detecting corners
from wall boundaries.
"""

from amsa import Algebra

print("\n=== Corridor Corner Detection ===")

alg = Algebra.pga2d()

left_wall = alg.multivector({"e01": 1.0, "e12": -1.0})
front_wall = alg.multivector({"e02": 1.0, "e12": -2.0})

corner = left_wall.regress(front_wall)

print("left wall:", left_wall.as_dense().values)
print("front wall:", front_wall.as_dense().values)

print("corner point:", corner.as_dense().values)
print("corner bulk part:", corner.bulk().as_dense().values)
print("corner weight part:", corner.weight().as_dense().values)
