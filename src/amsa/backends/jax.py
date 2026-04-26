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

from typing import Any

try:
    import jax.numpy as jnp
except ImportError:
    raise ImportError(
        "JAX is required for the JAX backend. "
        "Install with: uv pip install amsa-ga[jax]"
    )

from amsa.ir import ProductIR, SequenceIR, UnaryIR
from amsa.mv import MVArray


class JAXBackend:
    """JAX-based execution backend implementing the ``Executor`` protocol.

    This backend provides GPU-accelerated execution through JAX, with dense
    storage parity against the NumPy backend.
    """

    def execute_product(self, lhs: MVArray, rhs: MVArray, ir: ProductIR) -> MVArray:
        """Execute a product IR using JAX."""
        raise NotImplementedError("JAX product execution not yet implemented")

    def execute_unary(self, mv: MVArray, ir: UnaryIR) -> MVArray:
        """Execute a unary IR using JAX."""
        raise NotImplementedError("JAX unary execution not yet implemented")

    def execute_sequence(self, inputs: dict[str, Any], ir: SequenceIR) -> Any:
        """Execute a sequence IR using JAX."""
        raise NotImplementedError("JAX sequence execution not yet implemented")
