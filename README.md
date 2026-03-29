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
- addition and subtraction
- reverse, involute, and conjugate
- scalar arithmetic
- grade projection and component lookup
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

## Exploratory Probes

The repo currently also contains a temporary `tempo/` folder with exploratory challenge scripts and probes.

These are useful for:

- validating the current API on small real tasks
- stress-testing batch workflows
- comparing dense and CSR behavior
- collecting ideas for future examples, notebooks, and benchmarks

Run them from the repo root with:

```bash
uv run tempo/challenge1_triangle_area.py
uv run tempo/challenge2_orientation_batch.py
uv run tempo/challenge10_geometry_kernel.py
```

## Current Operations

| Category | Available now |
| --- | --- |
| Binary arithmetic | `add`, `sub`, `mv + other`, `mv - other` |
| Scalar arithmetic | `scalar * mv`, `mv * scalar`, multivector-scalar add/sub |
| Geometric products | geometric product `*`, outer product `^`, inner product `\|` |
| Unary operations | `neg`, `reverse`, `involute`, `conjugate`, unary `-mv` |
| Projection / inspection | `grade(...)`, `project_grades(...)`, `component(...)`, `as_dense()`, `to_layout(...)` |
| Storage operations | dense/CSR construction, `with_storage(...)`, `to_dense_storage(...)`, `to_csr_storage(...)` |
| Constructors | `scalar`, `blade`, `multivector`, `vector`, `bivector`, `trivector`, `even`, `odd`, `pseudoscalar`, `zeros` |
| Presets | `vga`, `vga2d`, `vga3d`, `pga2d`, `pga3d`, `Algebra.from_name(...)` |
