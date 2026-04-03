# AMSA

AMSA is a Clifford algebra library focused on high-performance numerical computation for robotics, engineering, and science.

AMSA is inspired by Kingdon and Look-Ma-No-Matrices, it is still under active development and doesn't have a stable implementation yet.

## Package Layout

- `src/amsa/specs.py`: algebra signatures, blade naming, blade products, presets
- `src/amsa/layouts.py`: dense, grade, and sparse layout descriptors
- `src/amsa/storage.py`: dense and CSR storage backends plus storage helpers
- `src/amsa/mv.py`: storage-backed multivector array type
- `src/amsa/plans.py`: cached operator plans
- `src/amsa/reference.py`: reference execution of plans
- `src/amsa/ops.py`: public operator layer
- `src/amsa/algebra.py`: user-facing algebra handle and constructors

## Quick Start

```python
from amsa import Algebra

alg = Algebra.vga2d()
u = alg.vector([1.0, 2.0])
v = alg.vector([3.0, -4.0])

gp = u * v
ip = u | v
op = u ^ v

print(gp.as_dense().values)  # [-5.0, 0.0, 0.0, -10.0]
print(ip.values)             # [-5.0]
print(op.values)             # [-10.0]
```

Sparse construction keeps support explicit:

```python
from amsa import Algebra

alg = Algebra.vga3d()
mv = alg.multivector({"e1": 1.0, "e12": 2.0, "e123": 3.0})

print(mv.layout.blades)          # (1, 3, 7)
print(mv.grade(1, 3).values)     # [1.0, 3.0]
print((2.0 - mv).as_dense().values)
```

Scalar construction is intentionally explicit:

```python
from amsa import Algebra

alg = Algebra.vga2d()
s = alg.scalar(1.0)
```

Use `alg.scalar(1.0)`, not `alg.multivector(1.0)`.

## What Works Today

- geometric product
- outer product
- inner product
- scalar product
- left contraction
- right contraction
- regressive product
- sandwich / conjugation
- bulk dual and weight dual on degenerate/projective algebras
- addition and subtraction
- inverse and division for the current reverse-scalar-norm cases
- reverse-based `norm_squared`, `norm`, and `normalize`
- bulk/weight norms plus `bulk_normalize` and `unitize` for PGA-style work
- reverse, involute, conjugate, dual, undual, poincare_dual, and poincare_undual
- scalar arithmetic
- grade projection and component lookup
- lazy basis-product tables and on-demand Cayley tables via `AlgebraSpec`
- dense/CSR conversion
- dense and CSR-backed input execution in the reference backend


## Development

Install and verify with `uv`:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Current Operations

| Category | Available now |
| --- | --- |
| Binary arithmetic | `add`, `sub`, `mv + other`, `mv - other` |
| Scalar arithmetic | `scalar * mv`, `mv * scalar`, `mv / scalar`, multivector-scalar add/sub |
| Geometric products | geometric product `*`, outer product `^`, inner product `\|`, `scalar_product`, `left_contraction`, `right_contraction`, `regressive_product`, `sandwich`, `bulk_dual`, `weight_dual` |
| Unary operations | `neg`, `reverse`, `involute`, `conjugate`, `dual`, `undual`, `poincare_dual`, `poincare_undual`, `inverse`, `norm_squared`, `norm`, `normalize`, `bulk_norm_squared`, `bulk_norm`, `weight_norm_squared`, `weight_norm`, `bulk_normalize`, `unitize`, unary `-mv` |
| Projection / inspection | `grade(...)`, `project_grades(...)`, `component(...)`, `as_dense()`, `to_layout(...)` |
| Storage operations | dense/CSR construction, `with_storage(...)`, `to_dense_storage(...)`, `to_csr_storage(...)` |
| Constructors | `scalar`, `blade`, `multivector`, `vector`, `bivector`, `trivector`, `even`, `odd`, `pseudoscalar`, `zeros` |
| Presets | `vga`, `vga2d`, `vga3d`, `pga2d`, `pga3d`, `Algebra.from_name(...)` |

`dual()` / `undual()` currently use the metric pseudoscalar transform, while
`poincare_dual()` / `poincare_undual()` use the metric-free basis complement.
That makes the Poincare pair available on degenerate algebras such as the PGA presets.

`inverse()` is currently a restricted reverse-based inverse: it succeeds when
`reverse(mv) * mv` and `mv * reverse(mv)` both collapse to the same nonzero scalar,
and raises otherwise.

`norm_squared()` returns the signed reverse norm scalar `<mv * reverse(mv)>_0`.
`norm()` takes `sqrt(abs(norm_squared))` so it stays real on indefinite signatures, and
`normalize()` divides by that magnitude.

For the current PGA presets, AMSA also exposes explicit bulk/weight helpers:
- `bulk()` and `weight()` split components by whether they carry the null basis factor
- `bulk_dual()` / `weight_dual()` apply Poincare complement duality to those parts
- `bulk_norm*` and `weight_norm*` keep the two normalization notions separate
- `bulk_normalize()` and `unitize()` are explicit PGA-facing normalization paths
