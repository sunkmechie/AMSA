from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from amsa import Algebra, geometric_product, reverse, dual, poincare_dual, normalize
from amsa.storage import to_csr_storage, to_dense_storage


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    operation: Callable[[], object]


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(case.operation, number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<50} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def build_cases() -> list[BenchCase]:
    """Benchmark CSR vs dense storage for common operations.

    These benchmarks compare performance of dense and CSR storage
    backends on operations that preserve or transform storage.
    """
    vga3d = Algebra.vga3d()
    pga3d = Algebra.pga3d()

    # Dense multivectors
    vga3d_dense = vga3d.multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0, "e12": 0.5, "e23": 0.5, "e13": 0.5})
    pga3d_dense = pga3d.multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0, "e0": 4.0})

    # CSR multivectors (sparse support)
    vga3d_csr = vga3d.multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0})  # Only grade-1
    pga3d_csr = pga3d.multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0})  # Only grade-1

    # Force CSR storage
    vga3d_csr_mv = vga3d_csr.with_storage("csr")
    pga3d_csr_mv = pga3d_csr.with_storage("csr")

    # Binary product operands
    vga3d_a = vga3d.multivector({"e1": 1.0, "e2": 2.0})
    vga3d_b = vga3d.multivector({"e1": 3.0, "e2": 4.0})

    vga3d_a_csr = vga3d_a.with_storage("csr")
    vga3d_b_csr = vga3d_b.with_storage("csr")

    return [
        # Binary products - dense vs CSR
        BenchCase(
            name="vga3d geometric_product (dense, dense)",
            operation=lambda: geometric_product(vga3d_a, vga3d_b),
        ),
        BenchCase(
            name="vga3d geometric_product (csr, csr)",
            operation=lambda: geometric_product(vga3d_a_csr, vga3d_b_csr),
        ),
        BenchCase(
            name="vga3d geometric_product (dense, csr)",
            operation=lambda: geometric_product(vga3d_a, vga3d_b_csr),
        ),
        # Unary operations - dense vs CSR
        BenchCase(
            name="vga3d reverse (dense)",
            operation=lambda: reverse(vga3d_dense),
        ),
        BenchCase(
            name="vga3d reverse (csr)",
            operation=lambda: reverse(vga3d_csr_mv),
        ),
        BenchCase(
            name="vga3d dual (dense)",
            operation=lambda: dual(vga3d_dense),
        ),
        BenchCase(
            name="vga3d dual (csr)",
            operation=lambda: dual(vga3d_csr_mv),
        ),
        BenchCase(
            name="pga3d poincare_dual (dense)",
            operation=lambda: poincare_dual(pga3d_dense),
        ),
        BenchCase(
            name="pga3d poincare_dual (csr)",
            operation=lambda: poincare_dual(pga3d_csr_mv),
        ),
        # Normalization - preserves CSR when possible
        BenchCase(
            name="vga3d normalize (dense)",
            operation=lambda: normalize(vga3d_dense),
        ),
        BenchCase(
            name="vga3d normalize (csr)",
            operation=lambda: normalize(vga3d_csr_mv),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AMSA storage backends (dense vs CSR).")
    parser.add_argument("--number", type=int, default=2000, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=7, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA storage backend benchmarks (dense vs CSR)")
    print(f"number={args.number} repeat={args.repeat}")
    print()

    for case in build_cases():
        print(_summarize(case, number=args.number, repeat=args.repeat))


if __name__ == "__main__":
    main()
