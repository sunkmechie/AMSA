"""AMSA Fused vs Dynamic Operator Benchmarks."""
from __future__ import annotations

import timeit
import numpy as np

try:
    import jax
except ImportError:
    print("Benchmarking requires JAX installed.")
    exit(1)

from amsa import Algebra
from amsa.ops import (
    sandwich,
    geometric_product,
    outer_product,
    inner_product,
    inverse,
    add,
)


def bench_operator(name, op, args, number=500, repeat=5):
    # Warmup and exact dynamic execution once
    _ = op(*args)

    def run_dynamic():
        return op(*args)

    dyn_times = timeit.repeat("run_dynamic()", globals=locals(), number=number, repeat=repeat)
    dyn_best = min(dyn_times) / number * 1e6

    # JIT Compile
    jitted_op = jax.jit(op)
    # Warmup JIT
    jitted_result = jitted_op(*args)
    if hasattr(jitted_result, "storage"):
        jitted_result.storage.array.block_until_ready()

    def run_fused():
        res = jitted_op(*args)
        if hasattr(res, "storage"):
            res.storage.array.block_until_ready()
        return res

    fused_times = timeit.repeat("run_fused()", globals=locals(), number=number, repeat=repeat)
    fused_best = min(fused_times) / number * 1e6

    print(f"{name:<20} | {dyn_best:8.1f} | {fused_best:8.1f} | {dyn_best / fused_best:7.1f}x")


def main():
    vga3d = Algebra.vga3d()
    number = 500
    repeat = 5

    R = vga3d.multivector({"e": np.cos(np.pi / 4), "e12": -np.sin(np.pi / 4)}, backend="jax")
    v = vga3d.multivector({"e1": 1.0, "e2": 0.5, "e3": 2.0}, backend="jax")

    print(f"AMSA JAX Benchmarks (number={number}, repeat={repeat})")
    print("-" * 55)
    print(f"{'Operator':<20} | {'Dyn (us)':>8} | {'Fused (us)':>8} | Speedup")
    print("-" * 55)

    bench_operator("geometric_product", geometric_product, (R, v), number, repeat)
    bench_operator("outer_product", outer_product, (R, v), number, repeat)
    bench_operator("inner_product", inner_product, (R, v), number, repeat)
    bench_operator("add", add, (R, v), number, repeat)
    bench_operator("inverse", inverse, (R,), number, repeat)
    bench_operator("sandwich", sandwich, (R, v), number, repeat)
    print("-" * 55)


if __name__ == "__main__":
    main()
