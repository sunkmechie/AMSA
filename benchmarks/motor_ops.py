from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

from amsa import Algebra, motor_exp, motor_log


@dataclass(frozen=True, slots=True)
class BenchCase:
    name: str
    operation: Callable[[], object]


def _summarize(case: BenchCase, *, number: int, repeat: int) -> str:
    timings = timeit.repeat(case.operation, number=number, repeat=repeat)
    per_call_us = [elapsed * 1_000_000.0 / number for elapsed in timings]
    return (
        f"{case.name:<24} "
        f"best={min(per_call_us):9.3f} us  "
        f"median={statistics.median(per_call_us):9.3f} us  "
        f"mean={statistics.mean(per_call_us):9.3f} us"
    )


def build_cases() -> list[BenchCase]:
    pga2d = Algebra.pga2d()
    pga3d = Algebra.pga3d()

    pga2d_generator = pga2d.multivector({"e12": -0.35, "e01": 0.1, "e02": -0.2})
    pga2d_motor = pga2d_generator.exp()

    pga3d_generator = pga3d.multivector({"e12": -0.3, "e03": 0.2, "e01": 0.05})
    pga3d_motor = motor_exp(pga3d_generator)

    return [
        BenchCase(name="pga2d exp", operation=lambda: pga2d_generator.exp()),
        BenchCase(name="pga2d motor_log", operation=lambda: motor_log(pga2d_motor)),
        BenchCase(name="pga3d motor_exp", operation=lambda: motor_exp(pga3d_generator)),
        BenchCase(name="pga3d motor_log", operation=lambda: motor_log(pga3d_motor)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AMSA motor exp/log operations.")
    parser.add_argument("--number", type=int, default=2000, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=5, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA motor benchmarks")
    print(f"number={args.number} repeat={args.repeat}")
    print()

    for case in build_cases():
        print(_summarize(case, number=args.number, repeat=args.repeat))


if __name__ == "__main__":
    main()
