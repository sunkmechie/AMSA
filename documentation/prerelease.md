# AMSA Prerelease Snapshot

## Status

AMSA is currently in a prerelease stage.

The project now has:

- a portable algebra core based on bit-pattern blade identifiers
- lazy numeric basis-product tables for small-basis algebras
- dense, grade-packed, and sparse layout descriptors
- dense and CSR storage backends behind a shared storage contract
- a reference multivector array type
- cached operator plans for binary products
- a first reference backend split between planning and execution
- storage-aware binary execution that can consume dense or CSR inputs
- explicit constructor-level backend selection for dense and CSR storage
- an external visual debugger probe that traces binary product plans and renders HTML reports
- a tested public API for the current reference semantics

Current verification status:

- `uv run pytest -q` passes
- `uv run ruff check .` passes
- `uv run mypy` passes
- the test suite currently covers layout behavior, algebra presets, product planning, geometric product equivalence, outer product, inner product, and the current convenience constructors

## Project Intent

AMSA is meant to become a high-performance geometric algebra library for robotics, engineering, and scientific computation.

The immediate direction is:

- keep algebra semantics separate from storage and execution
- treat layout-sensitive specialization as a core design choice
- make sparse support-pattern reasoning a first-class concept
- keep the Python/NumPy path as the reference backend
- prepare for future optimized backends without changing algebra semantics
- keep dense as the default fresh-construction backend until benchmarks justify anything more aggressive


## Current Architecture

The codebase is currently organized around these roles:

- `amsa.specs`: algebra signatures, blade naming, grade helpers, basis-blade products, preset specs
- `amsa.layouts`: layout descriptors for coefficient ordering and support
- `amsa.storage`: storage protocol plus dense and CSR coefficient backends
- `amsa.mv`: storage-backed multivectors tied to an algebra and layout
- `amsa.plans`: immutable cached product plans for binary operators
- `amsa.reference`: reference execution of precomputed plans
- `amsa.ops`: public operator layer for arithmetic and involutions
- `amsa.algebra`: user-facing constructors and convenience helpers

Binary products now use a two-phase reference path:

1. Build or fetch a cached `OpPlan` keyed by algebra plus the exact blade tuples of the input layouts.
2. Gather the coefficient slots referenced by that plan from dense or CSR storage, then execute into a dense result buffer for the output layout.

This is the current boundary between the pure reference backend and future optimized backend work.

For small-basis algebras, plan construction can also reuse a lazy numeric basis-product table from `amsa.specs`
instead of recomputing basis-blade products term by term.

Exploratory probes that inspect or visualize this execution path intentionally live outside `src/amsa`.
The current visual debugger prototype is `probes/amsa_lab.py`.

## Public API

The top-level package currently exports:

- `Algebra`
- `AlgebraSpec`
- `MVArray`
- `MVLayout`
- `add`
- `sub`
- `neg`
- `geometric_product`
- `outer_product`
- `inner_product`
- `scalar_product`
- `left_contraction`
- `right_contraction`
- `regressive_product`
- `inverse`
- `divide`
- `reverse`
- `involute`
- `conjugate`
- `dual`
- `undual`
- `poincare_dual`
- `poincare_undual`
- `project_grades`
- `grade_of_blade`
- `vga`
- `vga2d`
- `vga3d`
- `pga2d`
- `pga3d`

## Algebra Specs

`AlgebraSpec` currently provides:

- signature validation
- `dimension`
- `blade_count`
- `p`, `q`, `r`
- `grades()`
- `blade_name(blade)`
- `blade_names()`
- `blade_from_key(key)`
- `blades_of_grade(grade)`
- `grades_of_blades()`
- `pseudoscalar_blade`
- `basis_product_table`
- `blade_product(lhs, rhs)`
- `cayley_table()`
- `from_pqr(...)`

Preset spec constructors currently available:

- `vga(dimension)`
- `vga2d()`
- `vga3d()`
- `pga2d()`
- `pga3d()`

## Layouts

`MVLayout` currently supports:

- `MVLayout.dense(algebra)`
- `MVLayout.grade(algebra, *grades)`
- `MVLayout.sparse_pattern(algebra, blades, name=...)`

Layout metadata currently available:

- `blades`
- `kind`
- `name`
- `size`
- `grades`
- `blade_names()`
- `index_of(blade)`
- `contains(blade)`

## Storage

`amsa.storage` separates coefficient storage from layouts.

`MVStorage` is a typing `Protocol`, not a partially implemented base class. The `...` method bodies in
that protocol are interface signatures only; the concrete behavior lives in storage backends such as
`DenseStorage` and `CSRStorage`.

Storage backends currently available:

- `DenseStorage`
- `CSRStorage`

Storage conversion helpers currently available:

- `to_dense_storage(storage)`
- `to_csr_storage(storage)`

Internal storage execution helpers currently available:

- `storage_component(storage, column)`
- `gather_storage_columns(storage, columns, batch_shape=...)`

## Algebra Handle

`Algebra` is the main user-facing entry point and currently provides:

- preset constructors:
  - `Algebra.vga2d()`
  - `Algebra.vga3d()`
  - `Algebra.pga2d()`
  - `Algebra.pga3d()`
  - `Algebra.from_name(name)`
- layout helpers:
  - `dense_layout()`
  - `grade_layout(*grades)`
  - `even_layout()`
  - `odd_layout()`
  - `sparse_layout(blades, name=...)`
- constructors:
  - `zeros(..., backend="auto")`
  - `blade(key, value=1.0, backend="auto")`
  - `multivector(data, layout=None, backend="auto")` for mappings, arrays, and existing `MVArray` values
  - `scalar(value=0.0, backend="auto")`
  - `kvector(grade, values, backend="auto")`
  - `vector(values, backend="auto")`
  - `bivector(values, backend="auto")`
  - `trivector(values, backend="auto")`
  - `even(values, backend="auto")`
  - `odd(values, backend="auto")`
  - `pseudoscalar(value=0.0, backend="auto")`
- operator helpers:
  - `gp(lhs, rhs)`
  - `outer(lhs, rhs)`
  - `inner(lhs, rhs)`
  - `scalar_product(lhs, rhs)`
  - `left_contract(lhs, rhs)`
  - `right_contract(lhs, rhs)`
  - `regress(lhs, rhs)`
  - `inverse(mv)`
  - `add(lhs, rhs)`
  - `sub(lhs, rhs)`
  - `div(lhs, rhs)`

`Algebra.from_name(...)` currently recognizes:

- `vga2d`
- `vga3d`
- `pga2d`
- `2dpga`
- `pga3d`
- `3dpga`

## Naming Notes

AMSA currently uses three related names on purpose:

- `multivector`: the full mathematical and user-facing term
- `mv`: local shorthand used in code for an `MVArray` value or parameter
- `kvector`: a multivector restricted to a single grade

So `Algebra.multivector(...)` is the general constructor, while `kvector(...)`, `vector(...)`,
`bivector(...)`, and `trivector(...)` are more specific helpers layered on top of it.

Current backend policy:

- `backend="auto"` means dense for fresh construction today
- `backend="csr"` opts into CSR storage explicitly
- importing an existing `MVArray` preserves its current backend unless a different backend is requested

## Multivectors

`MVArray` currently provides:

- storage metadata:
  - `algebra`
  - `layout`
  - `storage_kind`
  - `values`
  - `batch_shape`
  - `dtype`
  - `grades`
- constructors:
  - `MVArray.zeros(...)`
  - `MVArray.from_array(...)`
- layout and inspection helpers:
  - `copy()`
  - `with_storage(kind)`
  - `to_layout(layout)`
  - `as_dense()`
  - `component(key)`
  - `grade(*grades)`
- unary operations:
  - `reverse()`
  - `involute()`
  - `conjugate()`
  - `dual()`
  - `undual()`
  - `poincare_dual()`
  - `poincare_undual()`
  - `inverse()`
  - unary negation via `-mv`
- binary operations:
  - `mv + other`
  - `mv - other`
  - `mv * other`
  - `mv / other`
  - `mv ^ other`
  - `mv | other`
  - scalar multiplication via `scalar * mv` and `mv * scalar`
  - scalar division and reverse-order division via `scalar / mv`
- named methods:
  - `outer(other)`
  - `inner(other)`
  - `scalar_product(other)`
  - `left_contract(other)`
  - `right_contract(other)`
  - `regress(other)`

## Exact Operations Available Today

These are the exact algebraic operations currently implemented in the reference backend:

### Binary multivector operations

- addition
- subtraction
- geometric product
- outer product
- inner product
- scalar product
- left contraction
- right contraction
- regressive product

### Unary multivector operations

- negation
- reverse
- involute
- conjugate
- dual
- undual
- poincare dual
- poincare undual
- inverse

### Scalar interactions

- multivector-scalar addition
- scalar-multivector addition
- multivector-scalar subtraction
- scalar-multivector subtraction
- left scalar multiplication
- right scalar multiplication
- multivector-scalar division
- scalar-multivector division through `inverse()`

### Projection and storage operations

- projection into a target layout via `to_layout(...)`
- backend conversion over the current layout via `with_storage(...)`
- dense conversion via `as_dense()`
- dense/CSR storage conversion via `amsa.storage.to_dense_storage(...)` and `amsa.storage.to_csr_storage(...)`
- grade selection via `grade(...)` and `project_grades(...)`
- component lookup by blade id or blade name

## Operator Semantics

The current binary product semantics are:

- geometric product:
  - includes every nonzero blade-pair contribution produced by `blade_product`
- outer product:
  - includes only terms whose output grade equals the sum of the input grades
- inner product:
  - includes only terms whose output grade equals the absolute difference of the input grades
- scalar product:
  - includes only terms whose output grade equals `0`
- left contraction:
  - includes only terms whose output grade equals `grade(rhs) - grade(lhs)` with `grade(lhs) <= grade(rhs)`
- right contraction:
  - includes only terms whose output grade equals `grade(lhs) - grade(rhs)` with `grade(lhs) >= grade(rhs)`
- regressive product:
  - is the Poincare-dual complement of the outer product:
    `regressive_product(a, b) = poincare_undual(poincare_dual(a) ^ poincare_dual(b))`

All seven products:

- respect the algebra metric, including degenerate signatures
- preserve sparse support when possible
- return dense output only when the implied support spans the full algebra basis
- can consume dense or CSR-backed operands in the reference backend
- broadcast over batch dimensions using NumPy broadcasting rules
- currently materialize the result as dense storage over the chosen output layout

## Current Limitations

The following are not implemented yet:

- sandwich operators
- normalization helpers
- symbolic backends
- JAX, Triton, or PyTorch execution paths
- density-based backend auto-selection beyond the current explicit policy
- CSR output emission for binary reference execution

The following API edge is important right now:

- `alg.multivector(1.0)` is not supported yet
- use `alg.scalar(1.0)` for scalar construction

## Getting Started

The safest way to use AMSA today is:

1. Construct an algebra preset with `Algebra.vga2d()`, `Algebra.vga3d()`, `Algebra.pga2d()`, `Algebra.pga3d()`, or `Algebra.from_name(...)`.
2. Build multivectors with `scalar`, `vector`, `bivector`, `trivector`, `even`, `odd`, `pseudoscalar`, or mapping-based `multivector({...})`.
3. Use `*`, `/`, `^`, `|`, `scalar_product(...)`, `left_contract(...)`, `right_contract(...)`, `regress(...)`, `+`, `-`, `inverse()`, `dual()`, `undual()`, `poincare_dual()`, and `poincare_undual()` for the currently implemented operators.
4. Use `component(...)`, `grade(...)`, and `as_dense()` to inspect results.

Duality note:

- `dual()` and `undual()` are metric pseudoscalar duals.
- They require an invertible pseudoscalar, so they raise on degenerate algebras such as the PGA presets.
- `poincare_dual()` and `poincare_undual()` are metric-free complement duals.
- The Poincare pair works on degenerate algebras, so it is the current duality path for PGA-style use cases.

Inverse note:

- `inverse()` is currently a restricted reverse-based inverse.
- It succeeds when `reverse(mv) * mv` and `mv * reverse(mv)` both reduce to the same nonzero scalar.
- That covers scalars, invertible blades, and common rotor-like/versor-like cases.
- It raises on null elements, degenerate zero-norm cases, and multivectors whose reverse norms do not collapse to a scalar.

### Example: 2D VGA vectors

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

### Example: sparse construction and scalar arithmetic

```python
from amsa import Algebra

alg = Algebra.vga2d()
x = alg.multivector({"e1": 1.0, "e12": -3.0})

print(x.layout.blades)         # (1, 3)
print((x + 2.0).as_dense().values)
print((3.0 - x).as_dense().values)
```

### Example: grade-aware construction in 3D VGA

```python
from amsa import Algebra

alg = Algebra.vga3d()
rotor_like = alg.even([1.0, 0.0, 0.5, -0.25])
mixed = alg.multivector({"e1": 1.0, "e12": 2.0, "e123": 3.0})

print(rotor_like.grades)           # (0, 2)
print(mixed.grade(1, 3).values)    # [1.0, 3.0]
print(mixed.grade(1, 3).layout.blades)  # (1, 7)
```

### Example: degenerate PGA behavior

```python
from amsa import Algebra

alg = Algebra.pga2d()
e0 = alg.blade("e0")
e1 = alg.blade("e1")

print((e0 * e0).layout.size)       # 0
print((e0 ^ e1).component("e01"))  # 1.0
```

### Example: batched coefficients

```python
import numpy as np
from amsa import Algebra

alg = Algebra.pga2d()
mv = alg.multivector({"e0": np.array([1.0, 2.0]), "e1": 3.0})

print(mv.batch_shape)  # (2,)
print(mv.values)       # [[1.0, 3.0], [2.0, 3.0]]
```

## Examples

`/examples` currently contains:

- `examples/algebra/even_odd_decomposition.py`
- `examples/geometry/orientation_batch_2d.py`
- `examples/geometry/signed_volume_3d.py`
- `examples/geometry/triangle_area_2d.py`
- `examples/kernels/geometric_kernels.py`
- `examples/planes/point_plane_distance_3d.py`

These are the scripts that should reflect the current public API most directly.

## Probes

`/probes` is the right place for exploratory tooling that depends on internal plan or tracing details without turning
those details into stable package API.

Current probe:

- `probes/amsa_lab.py`

What it does today:

- runs a trusted local AMSA expression
- captures binary product steps from the existing operator path
- uses `OpPlan` terms as the structural blade-interaction graph
- renders a self-contained HTML report

What it does not claim yet:

- a hardened expression parser
- a stable tracing API inside `src/amsa`
- full expression visualization for unary operations or addition/subtraction
- a geometry overlay for PGA object semantics

Minimal usage:

```bash
uv run python probes/amsa_lab.py \
  --algebra vga2d \
  --stmt "u = alg.vector([1.0, 2.0])" \
  --stmt "v = alg.vector([3.0, 4.0])" \
  --expr "u * v" \
  --output tempo/amsa_lab.html
```

That prototype is intentionally external to the core package so AMSA can explore debugger UX without blurring the
boundaries between algebra semantics, execution, and visualization.
