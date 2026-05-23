from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from amsa import Algebra, geometric_product


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    operation: Callable[[], object]


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(case.operation, number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<58} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def _sparse_batch(
    algebra: Algebra,
    *,
    batch_size: int,
    offset: int,
    backend: str,
):
    values = {
        "e1": np.where(np.arange(batch_size) % 4 == offset % 4, 1.0 + offset, 0.0),
        "e2": np.where(np.arange(batch_size) % 5 == offset % 5, 2.0 + offset, 0.0),
        "e3": np.where(np.arange(batch_size) % 7 == offset % 7, 3.0 + offset, 0.0),
        "e12": np.where(np.arange(batch_size) % 11 == offset % 11, 0.5 + offset, 0.0),
        "e23": np.where(np.arange(batch_size) % 13 == offset % 13, -0.25 - offset, 0.0),
    }
    return algebra.multivector(values, backend=backend)


def build_cases(batch_size: int) -> list[BenchCase]:
    algebra = Algebra.vga3d()

    lhs_csr = _sparse_batch(algebra, batch_size=batch_size, offset=0, backend="csr")
    rhs_csr = _sparse_batch(algebra, batch_size=batch_size, offset=1, backend="csr")
    lhs_dense = lhs_csr.with_storage("dense")
    rhs_dense = rhs_csr.with_storage("dense")

    scalar_csr = algebra.multivector(
        {"e1": 1.0, "e12": -2.0},
        backend="csr",
    )
    scalar_dense = scalar_csr.with_storage("dense")

    return [
        BenchCase(
            name="getitem batch slice (csr native)",
            operation=lambda: lhs_csr[::2],
        ),
        BenchCase(
            name="getitem batch slice (dense baseline)",
            operation=lambda: lhs_dense[::2],
        ),
        BenchCase(
            name="getitem batch slice (old densify fallback)",
            operation=lambda: lhs_csr.with_storage("dense")[::2],
        ),
        BenchCase(
            name="add broadcast csr+csr (csr native)",
            operation=lambda: lhs_csr + scalar_csr,
        ),
        BenchCase(
            name="add broadcast dense+dense baseline",
            operation=lambda: lhs_dense + scalar_dense,
        ),
        BenchCase(
            name="add broadcast old densify fallback",
            operation=lambda: lhs_csr.with_storage("dense") + scalar_csr.with_storage("dense"),
        ),
        BenchCase(
            name="sub broadcast csr-csr (csr native)",
            operation=lambda: lhs_csr - scalar_csr,
        ),
        BenchCase(
            name="sub broadcast dense-dense baseline",
            operation=lambda: lhs_dense - scalar_dense,
        ),
        BenchCase(
            name="sub broadcast old densify fallback",
            operation=lambda: lhs_csr.with_storage("dense") - scalar_csr.with_storage("dense"),
        ),
        BenchCase(
            name="geometric_product csr*csr (csr native)",
            operation=lambda: geometric_product(lhs_csr, rhs_csr),
        ),
        BenchCase(
            name="geometric_product dense*dense baseline",
            operation=lambda: geometric_product(lhs_dense, rhs_dense),
        ),
        BenchCase(
            name="geometric_product old densify fallback",
            operation=lambda: geometric_product(
                lhs_csr.with_storage("dense"),
                rhs_csr.with_storage("dense"),
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CSR-native paths against dense baselines and old densify fallbacks."
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch rows to benchmark.")
    parser.add_argument("--number", type=int, default=200, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=7, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA CSR-native benchmarks")
    print(f"batch_size={args.batch_size} number={args.number} repeat={args.repeat}")
    print()

    for case in build_cases(args.batch_size):
        print(_summarize(case, number=args.number, repeat=args.repeat))


if __name__ == "__main__":
    main()
