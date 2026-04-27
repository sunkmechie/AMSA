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

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import amsa
from amsa.backends.numpy import NumpyBackend
from amsa.ir import clear_backends, init, register_backend

try:
    from amsa.backends.jax import JAXBackend
    JAX_AVAILABLE = True
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    JAX_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    operation: Callable[[], object]


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(case.operation, number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<32} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def build_numpy_cases() -> list[BenchCase]:
    """Build benchmark cases for NumPy backend."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")
    
    vga2d = amsa.Algebra.vga2d()
    pga2d = amsa.Algebra.pga2d()
    pga3d = amsa.Algebra.pga3d()
    
    # VGA2D operations (single)
    u = vga2d.vector([1.0, 2.0])
    v = vga2d.vector([3.0, -4.0])
    
    # VGA2D operations (batched)
    u_batch = vga2d.vector([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    v_batch = vga2d.vector([[3.0, -4.0], [5.0, -6.0], [7.0, -8.0]])
    
    # VGA2D operations (large batch - throughput test)
    import numpy as np
    np.random.seed(42)
    n_large = 10000
    u_large = vga2d.vector(np.random.randn(n_large, 2))
    v_large = vga2d.vector(np.random.randn(n_large, 2))
    
    # PGA2D operations
    pga2d_generator = pga2d.multivector({"e12": -0.35, "e01": 0.1, "e02": -0.2})
    
    # PGA3D operations
    pga3d_generator = pga3d.multivector({"e12": -0.3, "e03": 0.2, "e01": 0.05})
    
    return [
        BenchCase(
            name="numpy: vga2d geometric_product (single)",
            operation=lambda: u * v,
        ),
        BenchCase(
            name="numpy: vga2d geometric_product (batch)",
            operation=lambda: u_batch * v_batch,
        ),
        BenchCase(
            name="numpy: vga2d geometric_product (large batch)",
            operation=lambda: u_large * v_large,
        ),
        BenchCase(name="numpy: vga2d outer_product", operation=lambda: u ^ v),
        BenchCase(name="numpy: vga2d inner_product", operation=lambda: u | v),
        BenchCase(name="numpy: pga2d exp", operation=lambda: pga2d_generator.exp()),
        BenchCase(
            name="numpy: pga3d motor_exp",
            operation=lambda: amsa.motor_exp(pga3d_generator),
        ),
    ]


def build_jax_cases() -> list[BenchCase]:
    """Build benchmark cases for JAX backend."""
    if not JAX_AVAILABLE:
        return []
    
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")
    
    vga2d = amsa.Algebra.vga2d()
    pga2d = amsa.Algebra.pga2d()
    pga3d = amsa.Algebra.pga3d()
    
    # VGA2D operations (single)
    u = vga2d.vector([1.0, 2.0])
    v = vga2d.vector([3.0, -4.0])
    
    # VGA2D operations (batched)
    u_batch = vga2d.vector([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    v_batch = vga2d.vector([[3.0, -4.0], [5.0, -6.0], [7.0, -8.0]])
    
    # VGA2D operations (large batch - throughput test)
    import numpy as np
    np.random.seed(42)
    n_large = 10000
    u_large = vga2d.vector(np.random.randn(n_large, 2))
    v_large = vga2d.vector(np.random.randn(n_large, 2))
    
    # PGA2D operations
    pga2d_generator = pga2d.multivector({"e12": -0.35, "e01": 0.1, "e02": -0.2})
    
    # PGA3D operations
    pga3d_generator = pga3d.multivector({"e12": -0.3, "e03": 0.2, "e01": 0.05})
    
    return [
        BenchCase(
            name="jax: vga2d geometric_product (single)",
            operation=lambda: u * v,
        ),
        BenchCase(
            name="jax: vga2d geometric_product (batch)",
            operation=lambda: u_batch * v_batch,
        ),
        BenchCase(
            name="jax: vga2d geometric_product (large batch)",
            operation=lambda: u_large * v_large,
        ),
        BenchCase(name="jax: vga2d outer_product", operation=lambda: u ^ v),
        BenchCase(name="jax: vga2d inner_product", operation=lambda: u | v),
        BenchCase(name="jax: pga2d exp", operation=lambda: pga2d_generator.exp()),
        BenchCase(
            name="jax: pga3d motor_exp",
            operation=lambda: amsa.motor_exp(pga3d_generator),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs JAX backends.")
    parser.add_argument("--number", type=int, default=2000, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=5, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA backend comparison benchmarks")
    print(f"number={args.number} repeat={args.repeat}")
    print()

    print("NumPy backend:")
    for case in build_numpy_cases():
        print(_summarize(case, number=args.number, repeat=args.repeat))
    
    print()
    
    if JAX_AVAILABLE:
        print("JAX backend:")
        for case in build_jax_cases():
            print(_summarize(case, number=args.number, repeat=args.repeat))
    else:
        print("JAX backend: not available (install with: uv pip install amsa-ga[jax])")


if __name__ == "__main__":
    main()
