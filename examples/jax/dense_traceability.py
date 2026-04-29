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

from __future__ import annotations

import jax
import jax.numpy as jnp

import amsa


def main() -> None:
    amsa.init(use="gpu")

    alg = amsa.Algebra.vga3d()
    lhs = alg.vector(jnp.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]))
    rhs = alg.vector(jnp.array([[4.0, -2.0, 1.0], [1.5, 0.25, -0.75]]))

    @jax.jit
    def product_values(a: amsa.MVArray, b: amsa.MVArray):
        return (a * b).values

    mapped_values = jax.jit(jax.vmap(lambda a, b: (a ^ b).values))(lhs, rhs)

    vector_layout = alg.grade_layout(1)

    def objective(coefficients):
        mv = amsa.MVArray(algebra=alg.spec, layout=vector_layout, values=coefficients)
        return amsa.norm_squared(mv).values[0]

    gradient = jax.grad(objective)(jnp.array([0.5, -1.5, 2.0]))

    print("=== Dense JAX Traceability ===")
    print("jitted geometric product:")
    print(product_values(lhs, rhs))
    print()
    print("jitted vmap outer product:")
    print(mapped_values)
    print()
    print("norm-squared gradient:")
    print(gradient)


if __name__ == "__main__":
    main()
