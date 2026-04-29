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

import numpy as np

import amsa
from amsa.backends.numpy import NumpyBackend
from amsa.ir import clear_backends, init, register_backend

try:
    import jax
    import jax.numpy as jnp

    from amsa.backends.jax import JAXBackend

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    operation: Callable[[], object]


def _block(value: object) -> object:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, amsa.MVArray):
        values = value.values
        if hasattr(values, "block_until_ready"):
            values.block_until_ready()
    return value


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(lambda: _block(case.operation()), number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<44} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def _register_numpy() -> None:
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")


def _register_jax() -> None:
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")


def build_numpy_cases(batch_size: int) -> list[BenchCase]:
    _register_numpy()
    rng = np.random.default_rng(42)
    alg = amsa.Algebra.vga3d()
    lhs = alg.vector(rng.normal(size=(batch_size, 3)))
    rhs = alg.vector(rng.normal(size=(batch_size, 3)))

    return [
        BenchCase("numpy eager: vga3d gp batch", lambda: lhs * rhs),
        BenchCase("numpy eager: vga3d outer batch", lambda: lhs ^ rhs),
        BenchCase("numpy eager: vga3d norm_squared batch", lambda: amsa.norm_squared(lhs)),
    ]


def build_jax_cases(batch_size: int) -> list[BenchCase]:
    if not JAX_AVAILABLE:
        return []

    _register_jax()
    rng = np.random.default_rng(42)
    alg = amsa.Algebra.vga3d()
    lhs = alg.vector(jnp.asarray(rng.normal(size=(batch_size, 3))))
    rhs = alg.vector(jnp.asarray(rng.normal(size=(batch_size, 3))))

    gp_jit = jax.jit(lambda a, b: (a * b).values)
    outer_jit = jax.jit(lambda a, b: (a ^ b).values)
    norm_jit = jax.jit(lambda a: amsa.norm_squared(a).values)
    gp_vmap_jit = jax.jit(jax.vmap(lambda a, b: (a * b).values))

    coeff_layout = alg.grade_layout(1)

    def scalar_objective(coefficients):
        mv = amsa.MVArray(algebra=alg.spec, layout=coeff_layout, values=coefficients)
        return amsa.norm_squared(mv).values[0]

    grad_jit = jax.jit(jax.grad(scalar_objective))
    grad_input = jnp.asarray([0.5, -1.5, 2.0])

    for warmed in (
        gp_jit(lhs, rhs),
        outer_jit(lhs, rhs),
        norm_jit(lhs),
        gp_vmap_jit(lhs, rhs),
        grad_jit(grad_input),
    ):
        _block(warmed)

    return [
        BenchCase("jax eager: vga3d gp batch", lambda: lhs * rhs),
        BenchCase("jax jit: vga3d gp batch", lambda: gp_jit(lhs, rhs)),
        BenchCase("jax jit: vga3d outer batch", lambda: outer_jit(lhs, rhs)),
        BenchCase("jax jit: vga3d norm_squared batch", lambda: norm_jit(lhs)),
        BenchCase("jax jit+vmap: vga3d gp batch", lambda: gp_vmap_jit(lhs, rhs)),
        BenchCase("jax jit+grad: vga3d norm objective", lambda: grad_jit(grad_input)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AMSA dense JAX traceability.")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--number", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    print("AMSA dense JAX traceability benchmarks")
    print(f"batch_size={args.batch_size} number={args.number} repeat={args.repeat}")
    print()

    for case in build_numpy_cases(args.batch_size):
        print(_summarize(case, number=args.number, repeat=args.repeat))

    print()
    if JAX_AVAILABLE:
        for case in build_jax_cases(args.batch_size):
            print(_summarize(case, number=args.number, repeat=args.repeat))
    else:
        print("JAX unavailable. Install with: uv pip install amsa-ga[jax]")


if __name__ == "__main__":
    main()
