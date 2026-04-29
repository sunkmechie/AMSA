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


def build_scale_product_cases() -> list[BenchCase]:
    """Build benchmark cases for scale + product fusion."""
    import numpy as np

    from amsa.backends.numpy import _execute_fused_scale_product, execute_product_ir, scale_storage
    from amsa.ir import ProductIR, TermIR
    from amsa.layouts import MVLayout
    from amsa.mv import MVArray
    from amsa.specs import AlgebraSpec
    from amsa.storage import DenseStorage, NumPyPayload

    algebra = AlgebraSpec(signature=(1, 1), start_index=1, basis_prefix='e')
    layout = MVLayout.grade(algebra, 1)
    
    # Small batch
    u_values = np.array([1.0, 2.0])
    v_values = np.array([3.0, -4.0])
    
    u_storage = DenseStorage(NumPyPayload(u_values.reshape(1, 2)))
    v_storage = DenseStorage(NumPyPayload(v_values.reshape(1, 2)))
    
    u = MVArray(algebra=algebra, layout=layout, storage=u_storage)
    v = MVArray(algebra=algebra, layout=layout, storage=v_storage)

    # Create ProductIR
    ir = ProductIR(
        kind="geometric",
        lhs_storage="dense",
        rhs_storage="dense",
        lhs_width=2,
        rhs_width=2,
        out_blades=(1, 2, 3),
        terms=(
            TermIR(lhs_col=0, rhs_col=0, out_col=0, coefficient=1),
            TermIR(lhs_col=0, rhs_col=1, out_col=2, coefficient=1),
            TermIR(lhs_col=1, rhs_col=0, out_col=2, coefficient=-1),
            TermIR(lhs_col=1, rhs_col=1, out_col=0, coefficient=-1),
        ),
    )

    factor = 2.0

    return [
        BenchCase(
            name="scale+product (non-fused)",
            operation=lambda: execute_product_ir(
                MVArray(algebra=algebra, layout=layout, storage=scale_storage(u.storage, factor)),
                v,
                ir
            )
        ),
        BenchCase(
            name="scale+product (fused)",
            operation=lambda: _execute_fused_scale_product(u, v, ir, factor)
        ),
    ]


def build_large_batch_cases() -> list[BenchCase]:
    """Build benchmark cases for large batch operations."""
    import numpy as np

    from amsa.backends.numpy import _execute_fused_scale_product, execute_product_ir, scale_storage
    from amsa.ir import ProductIR, TermIR
    from amsa.layouts import MVLayout
    from amsa.mv import MVArray
    from amsa.specs import AlgebraSpec
    from amsa.storage import DenseStorage, NumPyPayload

    algebra = AlgebraSpec(signature=(1, 1), start_index=1, basis_prefix='e')
    layout = MVLayout.grade(algebra, 1)
    
    # Large batch
    n = 10000
    u_values = np.random.randn(n, 2)
    v_values = np.random.randn(n, 2)
    
    u_storage = DenseStorage(NumPyPayload(u_values))
    v_storage = DenseStorage(NumPyPayload(v_values))
    
    u = MVArray(algebra=algebra, layout=layout, storage=u_storage)
    v = MVArray(algebra=algebra, layout=layout, storage=v_storage)

    # Create ProductIR
    ir = ProductIR(
        kind="geometric",
        lhs_storage="dense",
        rhs_storage="dense",
        lhs_width=2,
        rhs_width=2,
        out_blades=(1, 2, 3),
        terms=(
            TermIR(lhs_col=0, rhs_col=0, out_col=0, coefficient=1),
            TermIR(lhs_col=0, rhs_col=1, out_col=2, coefficient=1),
            TermIR(lhs_col=1, rhs_col=0, out_col=2, coefficient=-1),
            TermIR(lhs_col=1, rhs_col=1, out_col=0, coefficient=-1),
        ),
    )

    factor = 2.0

    return [
        BenchCase(
            name="scale+product large batch (non-fused)",
            operation=lambda: execute_product_ir(
                MVArray(algebra=algebra, layout=layout, storage=scale_storage(u.storage, factor)),
                v,
                ir
            )
        ),
        BenchCase(
            name="scale+product large batch (fused)",
            operation=lambda: _execute_fused_scale_product(u, v, ir, factor)
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fusion performance.")
    parser.add_argument("--number", type=int, default=1000, help="Calls per timing sample.")
    parser.add_argument("--repeat", type=int, default=5, help="Timing samples to collect.")
    args = parser.parse_args()

    print("AMSA fusion performance benchmarks")
    print(f"number={args.number} repeat={args.repeat}")
    print()

    print("Scale + Product (small):")
    for case in build_scale_product_cases():
        print(_summarize(case, number=args.number, repeat=args.repeat))

    print()
    print("Scale + Product (large batch):")
    for case in build_large_batch_cases():
        print(_summarize(case, number=max(1, args.number // 10), repeat=args.repeat))


if __name__ == "__main__":
    main()
