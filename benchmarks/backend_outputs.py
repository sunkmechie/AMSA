from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from amsa import Algebra


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    storage_kind: str
    operation: Callable[[], object]


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(case.operation, number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<28} "
        f"out={case.storage_kind:<5} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def build_cases() -> list[BenchCase]:
    algebra = Algebra.vga3d()
    batch = np.linspace(0.5, 2.0, 256)

    dense_lhs = algebra.multivector({"e1": batch, "e23": 1.0}, backend="dense")
    dense_rhs = algebra.multivector({"e2": 2.0, "e12": batch}, backend="dense")

    csr_lhs = algebra.multivector({"e1": batch, "e23": 1.0}, backend="csr")
    csr_rhs = algebra.multivector({"e2": 2.0, "e12": batch}, backend="csr")

    cases: list[BenchCase] = []

    dense_result = dense_lhs * dense_rhs
    cases.append(
        BenchCase(
            name="dense gp",
            storage_kind=dense_result.storage_kind,
            operation=lambda: dense_lhs * dense_rhs,
        )
    )

    csr_result = csr_lhs * csr_rhs
    cases.append(
        BenchCase(
            name="csr->csr gp",
            storage_kind=csr_result.storage_kind,
            operation=lambda: csr_lhs * csr_rhs,
        )
    )

    mixed_result = dense_lhs * csr_rhs
    cases.append(
        BenchCase(
            name="mixed dense/csr gp",
            storage_kind=mixed_result.storage_kind,
            operation=lambda: dense_lhs * csr_rhs,
        )
    )

    try:
        import jax  # noqa: F401
    except Exception:
        return cases

    jax_lhs = algebra.multivector({"e1": batch, "e23": 1.0}, backend="jax")
    jax_rhs = algebra.multivector({"e2": 2.0, "e12": batch}, backend="jax")
    jax_result = jax_lhs * jax_rhs
    cases.append(
        BenchCase(
            name="jax->jax gp",
            storage_kind=jax_result.storage_kind,
            operation=lambda: jax_lhs * jax_rhs,
        )
    )

    mixed_jax_result = jax_lhs * dense_rhs
    cases.append(
        BenchCase(
            name="mixed jax/dense gp",
            storage_kind=mixed_jax_result.storage_kind,
            operation=lambda: jax_lhs * dense_rhs,
        )
    )

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AMSA backend-preserving binary execution paths."
    )
    parser.add_argument("--number", type=int, default=500, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=5, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA backend output benchmarks")
    print(f"number={args.number} repeat={args.repeat}")
    print()

    for case in build_cases():
        print(_summarize(case, number=args.number, repeat=args.repeat))


if __name__ == "__main__":
    main()
