from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

from amsa import Algebra, geometric_product, outer_product, reverse, dual, poincare_dual


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    operation: Callable[[], object]


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(case.operation, number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<40} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def build_cases() -> list[BenchCase]:
    """Benchmark IR routing overhead for common operations.

    These benchmarks measure the full IR execution path through
    the NumPy backend, including IR construction, backend dispatch,
    and coefficient computation.
    """
    vga2d = Algebra.vga2d()
    vga3d = Algebra.vga3d()
    pga2d = Algebra.pga2d()
    pga3d = Algebra.pga3d()

    # Binary products
    vga2d_a = vga2d.multivector({"e1": 1.0, "e2": 2.0})
    vga2d_b = vga2d.multivector({"e1": 3.0, "e2": 4.0})

    vga3d_a = vga3d.multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0})
    vga3d_b = vga3d.multivector({"e1": 4.0, "e2": 5.0, "e3": 6.0})

    # Unary operations
    vga3d_mv = vga3d.multivector({"e1": 1.0, "e2": 2.0, "e12": 3.0, "e123": 4.0})
    pga3d_mv = pga3d.multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0, "e0": 4.0})

    return [
        # Binary products - these go through ProductIR
        BenchCase(
            name="vga2d geometric_product",
            operation=lambda: geometric_product(vga2d_a, vga2d_b),
        ),
        BenchCase(
            name="vga2d outer_product",
            operation=lambda: outer_product(vga2d_a, vga2d_b),
        ),
        BenchCase(
            name="vga3d geometric_product",
            operation=lambda: geometric_product(vga3d_a, vga3d_b),
        ),
        BenchCase(
            name="vga3d outer_product",
            operation=lambda: outer_product(vga3d_a, vga3d_b),
        ),
        # Unary operations - these go through UnaryIR
        BenchCase(
            name="vga3d reverse",
            operation=lambda: reverse(vga3d_mv),
        ),
        BenchCase(
            name="vga3d dual",
            operation=lambda: dual(vga3d_mv),
        ),
        BenchCase(
            name="pga3d poincare_dual",
            operation=lambda: poincare_dual(pga3d_mv),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AMSA IR routing overhead.")
    parser.add_argument("--number", type=int, default=5000, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=7, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA IR routing benchmarks")
    print(f"number={args.number} repeat={args.repeat}")
    print()

    for case in build_cases():
        print(_summarize(case, number=args.number, repeat=args.repeat))


if __name__ == "__main__":
    main()
