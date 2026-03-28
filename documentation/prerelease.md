# AMSA Prerelease Snapshot

## Status

AMSA is currently in a prerelease stage.

The project now has:

- a portable algebra core based on bit-pattern blade identifiers
- dense, grade-packed, and sparse layout descriptors
- dense and CSR storage backends behind a shared storage contract
- a reference multivector array type
- cached operator plans for binary products
- a first reference backend split between planning and execution
- storage-aware binary execution that can consume dense or CSR inputs
- explicit constructor-level backend selection for dense and CSR storage
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

- `amsa.specs`: algebra signatures, blade naming, grade helpers, preset specs
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
- `reverse`
- `involute`
- `conjugate`
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
- `blade_product(lhs, rhs)`
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
  - `add(lhs, rhs)`
  - `sub(lhs, rhs)`

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
  - unary negation via `-mv`
- binary operations:
  - `mv + other`
  - `mv - other`
  - `mv * other`
  - `mv ^ other`
  - `mv | other`
  - scalar multiplication via `scalar * mv` and `mv * scalar`
- named methods:
  - `outer(other)`
  - `inner(other)`

## Exact Operations Available Today

These are the exact algebraic operations currently implemented in the reference backend:

### Binary multivector operations

- addition
- subtraction
- geometric product
- outer product
- inner product

### Unary multivector operations

- negation
- reverse
- involute
- conjugate

### Scalar interactions

- multivector-scalar addition
- scalar-multivector addition
- multivector-scalar subtraction
- scalar-multivector subtraction
- left scalar multiplication
- right scalar multiplication

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

All three products:

- respect the algebra metric, including degenerate signatures
- preserve sparse support when possible
- return dense output only when the implied support spans the full algebra basis
- can consume dense or CSR-backed operands in the reference backend
- broadcast over batch dimensions using NumPy broadcasting rules
- currently materialize the result as dense storage over the chosen output layout

## Current Limitations

The following are not implemented yet:

- left contraction
- right contraction
- scalar product as a separate operator
- regressive product
- dual / undual helpers
- inverse / division
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
3. Use `*`, `^`, `|`, `+`, and `-` for the currently implemented operators.
4. Use `component(...)`, `grade(...)`, and `as_dense()` to inspect results.

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

There is no dedicated examples directory yet.
For now, use the snippets in this prerelease snapshot and the test suite as the most accurate usage references.
