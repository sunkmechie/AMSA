# AMSA

![PyPI](https://img.shields.io/pypi/v/amsa-ga)

AMSA is a Clifford algebra library for high-performance numerical computation in robotics, engineering, and science.

**Beta on PyPI:** `uv pip install amsa-ga==0.1.0b0`

## Table of Contents

1. [What is AMSA?](#what-is-amsa)
2. [Package Layout](#package-layout)
3. [Quick Start](#quick-start)
4. [Documentation](#documentation)
5. [Notebooks](#notebooks)
6. [License and Acknowledgments](#license-and-acknowledgments)
7. [What Works Today](#what-works-today)
8. [Development](#development)
9. [Current Operations](#current-operations)
10. [Notes](#notes)

## What is AMSA?

AMSA (Advanced Multivector Symbolic Architecture Engine) is a Clifford algebra library focused on high-performance numerical computation for robotics, engineering, and science.

## Installation

```bash
uv pip install amsa-ga==0.1.0b0
```

Or with visualization extras:

```bash
uv pip install amsa-ga[viz]==0.1.0b0
```

## Package Layout

- `src/amsa/specs.py`: algebra signatures, blade naming, blade products, presets
- `src/amsa/layouts.py`: dense, grade, and sparse layout descriptors
- `src/amsa/storage.py`: dense and CSR storage backends plus storage helpers
- `src/amsa/mv.py`: storage-backed multivector array type
- `src/amsa/plans.py`: cached operator plans
- `src/amsa/reference.py`: reference execution of plans
- `src/amsa/ops.py`: public operator layer
- `src/amsa/algebra.py`: user-facing algebra handle and constructors
- `src/amsa/viz/`: visualization adapters, neutral primitives, and optional backends

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

Scalar construction:

```python
from amsa import Algebra

alg = Algebra.vga2d()
s = alg.scalar(1.0)
```

Use `alg.scalar(1.0)`, not `alg.multivector(1.0)`.

## Documentation

Full documentation is in `docs/` 

```bash
uv run sphinx-build docs docs/_build
```

You can also browse the source directly:

- `docs/quickstart.rst` — installation and first steps
- `docs/algebra.rst` — `AlgebraSpec`, presets, and blade products
- `docs/layouts.rst` — `MVLayout` and sparse support
- `docs/storage.rst` — dense and CSR backends
- `docs/operators.rst` — product semantics, duality, and normalization
- `docs/viz.rst` — visualization adapters, primitives, and optional matplotlib/VisPy backends
- `docs/examples.rst` — index of runnable example scripts
- `docs/probes.rst` — visual debugger probe (`amsa_lab`)

## Notebooks

Introductory notebooks are in `notebooks/`:

- `01_vga_rotors.ipynb` — VGA vector products, rotors, and sandwich conjugation
- `02_pga_rigid_body.ipynb` — PGA2d lines, meet/join, motors, and bulk/weight splits

## License and Acknowledgements

The AMSA source code is licensed under Apache 2.0.

AMSA's development has been made possible and was inspired by the following open-source projects:

- Kingdon
- Look-Ma-No-Matrices
- Ganja.js

## What Works Today

- geometric product
- outer product
- inner product
- scalar product
- commutator
- anticommutator
- left contraction
- right contraction
- regressive product
- sandwich / conjugation
- exponential / logarithm support (for robotics-friendly motor slices)
- bulk dual and weight dual on degenerate/projective algebras
- addition and subtraction
- inverse and division for the current reverse-scalar-norm cases
- reverse-based `norm_squared`, `norm`, and `normalize`
- bulk/weight norms plus `bulk_normalize`, `unitize`, and `rigid_body_normalize` (for PGA-style work)
- reverse, involute, conjugate, dual, undual, poincare_dual, and poincare_undual
- scalar arithmetic
- grade projection and component lookup
- lazy basis-product tables and on-demand Cayley tables via `AlgebraSpec`
- dense/CSR conversion
- dense and CSR-backed input execution in the reference backend
- neutral visualization primitives, point adapters, and backend modules in `amsa.viz`


## Development

For local development after cloning the repository:

```bash
uv sync --extra dev --extra viz
uv run pytest -q
uv run ruff check .
uv run mypy
```

Build the documentation:

```bash
uv run sphinx-build docs docs/_build
```

## Current Operations

| Category | Available now |
| --- | --- |
| Binary arithmetic | `add`, `sub`, `mv + other`, `mv - other` |
| Scalar arithmetic | `scalar * mv`, `mv * scalar`, `mv / scalar`, multivector-scalar add/sub |
| Geometric products | geometric product `*`, outer product `^`, inner product `\|`, `scalar_product`, `commutator_product`, `anticommutator_product`, `left_contraction`, `right_contraction`, `regressive_product`, `sandwich`, `bulk_dual`, `weight_dual` |
| Unary operations | `neg`, `reverse`, `involute`, `conjugate`, `dual`, `undual`, `poincare_dual`, `poincare_undual`, `inverse`, `exp`, `motor_exp`, `motor_log`, `norm_squared`, `norm`, `normalize`, `bulk_norm_squared`, `bulk_norm`, `weight_norm_squared`, `weight_norm`, `bulk_normalize`, `unitize`, `rigid_body_normalize`, unary `-mv` |
| Projection / inspection | `grade(...)`, `project_grades(...)`, `component(...)`, `as_dense()`, `to_layout(...)` |
| Storage operations | dense/CSR construction, `with_storage(...)`, `to_dense_storage(...)`, `to_csr_storage(...)` |
| Constructors | `scalar`, `blade`, `multivector`, `vector`, `bivector`, `trivector`, `even`, `odd`, `pseudoscalar`, `zeros` |
| Presets | `vga`, `vga2d`, `vga3d`, `pga2d`, `pga3d`, `Algebra.from_name(...)` |

## Notes

`dual()` / `undual()` currently use the metric pseudoscalar transform, while
`poincare_dual()` / `poincare_undual()` use the metric-free basis complement.
That makes the Poincare pair available on degenerate algebras such as the PGA presets.

`inverse()` is currently a restricted reverse-based inverse: it succeeds when
`reverse(mv) * mv` and `mv * reverse(mv)` both collapse to the same nonzero scalar,
and raises otherwise.

`norm_squared()` returns the signed reverse norm scalar `<mv * reverse(mv)>_0`.
`norm()` takes `sqrt(abs(norm_squared))` so it stays real on indefinite signatures, and
`normalize()` divides by that magnitude.

`commutator_product(a, b)` and `anticommutator_product(a, b)` expose the Lie/Jordan
splits of the geometric product:
- `0.5 * (a * b - b * a)`
- `0.5 * (a * b + b * a)`

`exp()` is currently defined for simple elements whose square collapses to a scalar.
That covers the common circular, hyperbolic, and nilpotent generator cases used for
rotors, boosts, and translators.

For robotics-oriented PGA3d work, AMSA also supports `motor_exp()` for pure bivector
twist generators, and `exp()` now dispatches to that same closed form when given a
PGA3d bivector whose square is scalar + pseudoscalar valued.

`motor_log()` is the inverse-side companion for the currently supported robotics cases.
Today it supports:
- PGA2d motor-like even multivectors after rigid-body normalization
- PGA3d unit-motor style multivectors with scalar, bivector, and optional pseudoscalar terms


For the current PGA presets, AMSA also exposes explicit bulk/weight helpers:
- `bulk()` and `weight()` split components by whether they carry the null basis factor
- `bulk_dual()` / `weight_dual()` apply Poincare complement duality to those parts
- `bulk_norm*` and `weight_norm*` keep the two normalization notions separate
- `bulk_normalize()` and `unitize()` are explicit PGA-facing normalization paths
- `rigid_body_normalize()` is a motor-oriented PGA helper that currently bulk-normalizes
  even grade-`0/2` multivectors without pretending to be a universal projective normalization

Visualization note:
- `amsa.viz`provides neutral primitives, point adapters for PGA points, and optional matplotlib/VisPy backends


