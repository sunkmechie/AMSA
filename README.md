# AMSA

![PyPI](https://img.shields.io/pypi/v/amsa-ga)

AMSA is a matrix-free Clifford algebra engine for high-performance numerical
computation in robotics, engineering, and science. Supports **VGA** (vector),
**PGA** (projective), and **CGA** (conformal) geometric algebras.

**Install:** `uv pip install amsa-ga`

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Algebras](#supported-algebras)
3. [Inspection & Classification](#inspection--classification)
4. [Examples](#examples)
5. [Execution Backends](#execution-backends)
6. [Package Layout](#package-layout)
7. [Documentation](#documentation)
8. [Notebooks](#notebooks)
9. [License and Acknowledgments](#license-and-acknowledgments)
10. [Current Operations](#current-operations)
11. [Notes](#notes)
12. [Development](#development)

## Quick Start

### VGA — vectors, wedge products, rotors

```python
from amsa import Algebra

alg = Algebra.vga3d()
u = alg.vector([1.0, 2.0, 3.0])
v = alg.vector([4.0, -2.0, 1.0])

print(u ^ v)                  # bivector: -10.0 e12 + -11.0 e13 + 8.0 e23
print(u | v)                  # inner product: 3.0
```

### PGA — motors, meet/join, rigid motion

```python
from amsa import Algebra

alg = Algebra.pga2d()
motor = alg.multivector({"e": 0.7071, "e12": 0.7071})
point = alg.multivector({"e12": 1.0, "e01": 2.0, "e02": 1.0})

moved = amsa.sandwich(motor, point)
```

### CGA — points, spheres, translators, classification

```python
from amsa import Algebra

alg = Algebra.cga3d()
p = alg.point([1.0, 2.0, 3.0])
s = alg.sphere([0.0, 0.0, 0.0], 5.0)
T = alg.translate([3.0, 0.0, 0.0])

moved = amsa.sandwich(T, p)
print(alg.classify(moved))    # EntityInfo: conformal point, coordinates [4, 2, 3]
```

See `docs/quickstart.rst` and `examples/` for more.

## Supported Algebras

| Algebra | Presets | Signature | Highlights |
|---|---|---|---|
| **VGA** (vector) | `vga2d()`, `vga3d()` | `(1,1)`, `(1,1,1)` | Rotors, wedge products, projections |
| **PGA** (projective) | `pga2d()`, `pga3d()` | `(0,1,1)`, `(0,1,1,1)` | Motors, meet/join, bulk/weight splits |
| **CGA** (conformal) | `cga2d()`, `cga3d()` | `(1,1,1,-1)`, `(1,1,1,1,-1)` | Points, spheres, planes, translators, `classify()` |

All presets are available as `Algebra.vga3d()`, `Algebra.pga2d()`, `Algebra.cga3d()`,
etc., or via `Algebra.from_name("cga3d")`.

## Inspection & Classification

AMSA provides inspection helpers for debugging and understanding algebraic operations.

**MVArray display:**

```python
from amsa import Algebra
alg = Algebra.vga2d()
u = alg.vector([1.0, 2.0])
print(u)  # 1.0 e1 + 2.0 e2
```

**Product plan and IR inspection:**

```python
from amsa.plans import plan_binary_product
from amsa.ir import build_product_ir

plan = plan_binary_product(alg.grade_layout(1), alg.grade_layout(1), "geometric")
print(plan.show())  # OpPlan: blades, terms, coefficients
```

**Cayley table subsets:**

```python
print(alg.show_cayley())
# Cayley table for (1, 1) (4 blades)
#        e     e1    e2   e12
#      e  +     e  +  e1  +  e2  + e12
#     e1  +    e1  +   e  + e12  +  e2
#     e2  +    e2 -1 e12  +   e -1  e1
#    e12  +   e12 -1  e2  +  e1 -1   e
```

**Geometric classification (CGA):**

```python
alg = Algebra.cga3d()
info = alg.classify(alg.sphere([1.0, 0.0, 0.0], 3.0))
print(info)
# CGA3D Classification
# kind:           dual sphere
# representation: dual
# geometric data:
#   center: [1. 0. 0.]
#   radius: 3.0
```

See `docs/inspection.rst`, `docs/cga.rst`, and `examples/inspection/`.

## Examples

Runnable example scripts live in `examples/`:

| Directory | Topic |
|---|---|
| [`cga/`](examples/cga/) | CGA primitives, batched distance, classify overview, versor actions |
| [`geometry/`](examples/geometry/) | Triangle area, signed volume, orientation testing |
| [`robotics/`](examples/robotics/) | PGA motors, kinematic chain, trilateration, rigid body trajectory |
| [`inspection/`](examples/inspection/) | Cayley tables, product plans, IR, MVArray display |
| [`jax/`](examples/jax/) | JAX traceability: `jax.jit`, `jax.vmap`, `jax.grad` |
| [`algebra/`](examples/algebra/) | Even/odd grade decomposition |
| [`planes/`](examples/planes/) | Point-to-plane distance using duals |
| [`kernels/`](examples/kernels/) | Geometric kernel functions |

Run any example with:

```bash
uv run python examples/cga/cga_classify_overview.py
```

## Execution Backends

AMSA supports pluggable execution backends for coefficient computation. Backends are selected by device type:

```python
import amsa

# CPU execution (NumPy) - default
amsa.init(use="cpu")

# GPU execution (JAX) - requires amsa-ga[jax] extra
# amsa.init(use="gpu")

# Check current device
print(amsa.get_device())  # "cpu"
```

**JAX Installation:**

For CPU execution:

```bash
uv pip install amsa-ga[jax]
```

For GPU execution (CUDA), install JAX with CUDA support:

```bash
uv pip install "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then select GPU execution:

```python
import amsa
amsa.init(use="gpu")
```

Dense AMSA multivectors can be used directly with JAX transforms:

```python
import jax
import jax.numpy as jnp
import amsa

amsa.init(use="gpu")

alg = amsa.Algebra.vga3d()
u = alg.vector(jnp.array([1.0, 2.0, 3.0]))
v = alg.vector(jnp.array([4.0, -2.0, 1.0]))

product_values = jax.jit(lambda a, b: (a * b).values)
print(product_values(u, v))
```

See the documentation for details on execution backends.

## Package Layout

- `src/amsa/specs.py` — algebra signatures, blade naming, blade products, presets
- `src/amsa/layouts.py` — dense, grade, and sparse layout descriptors
- `src/amsa/storage.py` — dense and CSR storage backends plus storage helpers
- `src/amsa/mv.py` — storage-backed multivector array type
- `src/amsa/plans.py` — cached operator plans
- `src/amsa/ir.py` — IR definitions and backend registry
- `src/amsa/backends/` — execution backend implementations
- `src/amsa/ops.py` — public operator layer
- `src/amsa/algebra.py` — user-facing algebra handle, constructors, `classify()` routing
- `src/amsa/inspection.py` — `EntityInfo` dataclass and geometric classification (CGA, future PGA/VGA)
- `src/amsa/cga.py` — CGA geometry helpers (point, sphere, plane, translator, extraction)
- `src/amsa/viz/` — visualization adapters, neutral primitives, and optional backends

## Documentation

Full documentation is in `docs/`:

```bash
uv run sphinx-build docs docs/_build
```

You can also browse the source directly:

- `docs/quickstart.rst` — installation and first steps
- `docs/algebra.rst` — `AlgebraSpec`, presets, and blade products
- `docs/layouts.rst` — `MVLayout` and sparse support
- `docs/storage.rst` — dense and CSR backends
- `docs/backends.rst` — execution backend selection (CPU/GPU)
- `docs/operators.rst` — product semantics, duality, and normalization
- `docs/cga.rst` — CGA presets, constructors, extraction, classification
- `docs/viz.rst` — visualization adapters, primitives, and optional matplotlib/VisPy backends
- `docs/examples.rst` — index of runnable example scripts
- `docs/probes.rst` — visual debugger probe (`amsa_lab`)

## Notebooks

Introductory notebooks are in `notebooks/`:

- `01_vga_rotors.ipynb` — VGA vector products, rotors, and sandwich conjugation
- `02_pga_rigid_body.ipynb` — PGA2d lines, meet/join, motors, and bulk/weight splits

## License and Acknowledgments

The AMSA source code is licensed under Apache 2.0.

AMSA's development has been made possible and was inspired by the following open-source projects:

- Kingdon
- Look-Ma-No-Matrices
- Ganja.js

## Current Operations

| Category | Operations |
|---|---|
| Binary arithmetic | `add`, `sub`, `mv + other`, `mv - other` |
| Scalar arithmetic | `scalar * mv`, `mv * scalar`, `mv / scalar`, multivector-scalar add/sub |
| Geometric products | `*` (geometric), `^` (outer), `\|` (inner), `scalar_product`, `commutator_product`, `anticommutator_product`, `left_contraction`, `right_contraction`, `regressive_product`, `sandwich`, `bulk_dual`, `weight_dual` |
| Unary operations | `neg`, `reverse`, `involute`, `conjugate`, `dual`, `undual`, `poincare_dual`, `poincare_undual`, `inverse`, `exp`, `motor_exp`, `motor_log`, `norm_squared`, `norm`, `normalize`, `bulk_norm_squared`, `bulk_norm`, `weight_norm_squared`, `weight_norm`, `bulk_normalize`, `unitize`, `rigid_body_normalize`, `-mv` |
| Projection / inspection | `grade(...)`, `project_grades(...)`, `component(...)`, `as_dense()`, `to_layout(...)`, `show_cayley()`, `classify()` |
| Storage operations | dense/CSR construction, `with_storage(...)`, `to_dense_storage(...)`, `to_csr_storage(...)` |
| Constructors | `scalar`, `blade`, `multivector`, `vector`, `bivector`, `trivector`, `even`, `odd`, `pseudoscalar`, `zeros` |
| CGA constructors | `origin`, `infinity`, `euclidean_vector`, `point`, `sphere`, `plane`, `translate`, `line_through_points`, `circle_through_points`, `distance_squared` |
| CGA extraction | `extract_point`, `extract_sphere`, `extract_plane`, `extract_euclidean_vector` |
| Presets | `vga`, `vga2d`, `vga3d`, `pga2d`, `pga3d`, `cga2d`, `cga3d`, `Algebra.from_name(...)` |

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

**CGA classification:** `alg.classify(mv)` inspects a multivector and returns an
`EntityInfo` describing its geometric interpretation (kind, grades, nullity, normalization,
invariants, geometric data, and storage metadata). Currently CGA-first; PGA and VGA
classifiers are planned for a future pass. See `docs/cga.rst` and
`examples/cga/cga_classify_overview.py`.

Visualization note:
- `amsa.viz` provides neutral primitives, point adapters for PGA points, and optional
  matplotlib/VisPy backends

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
