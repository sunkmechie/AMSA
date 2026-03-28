# AMSA Agent Guide

## Core Philosophy

AMSA is a matrix-free Clifford algebra engine.

Every implementation decision should preserve these invariants:

- Stay in pure Clifford terms: blades, grades, signatures, layouts, supports, products.
- Do not introduce matrix formulations, matrix kernels, basis-change tables, or linear-algebra-style operator abstractions as the primary execution model.
- Keep bit-pattern blade identifiers as the core algebra representation.
- Treat layout as blade ordering and support metadata only.
- Treat storage as coefficient representation only.
- Treat execution as a separate concern from algebra, layout, and storage.

If a change makes the code feel more like sparse matrix algebra than multivector algebra, it is probably moving AMSA in the wrong direction.

## Architectural Boundaries

- `amsa.specs`: algebra semantics, signatures, blade naming, blade products, presets.
- `amsa.layouts`: supported blade sets and coefficient orderings.
- `amsa.storage`: backend representation of coefficients; currently dense and CSR.
- `amsa.mv`: multivector container binding algebra + layout + storage.
- `amsa.plans`: cached product plans derived from blade support.
- `amsa.reference`: reference execution of product plans.
- `amsa.ops`: public algebraic operators and involutions.
- `amsa.algebra`: user-facing constructors and convenience helpers.

Keep these layers separate. Do not let storage rules leak into algebra semantics, and do not let layout objects become execution engines.

## Sparse / CSR Rules

- CSR is a storage backend, not a new layout kind.
- CSR rows map to flattened batch elements.
- CSR columns map to layout-local coefficient slots.
- Preserve public multivector semantics across dense and CSR backends.
- Prefer backend-aware coefficient operations over silent dense fallback when the operation is naturally storage-local.
- Do not change operator semantics just to suit a sparse backend.


## Implementation Preferences

- Prefer support-driven reasoning over full dense-basis work when semantics allow it.
- Preserve canonical blade ordering from layouts.
- Keep zero-support and exact-zero behavior correct without ad hoc special casing.
- Favor immutable descriptors and explicit helpers over hidden mutation.
- Add small, composable storage helpers instead of branching backend logic throughout the codebase.

## Testing Expectations

- Cover both dense and CSR behavior whenever touching storage-aware paths.
- Test empty sparse layouts, exact zeros, and batched broadcasting.
- Verify results through algebra semantics, not implementation details alone.
- Prefer regression tests that lock down blade support, coefficient values, backend preservation, and layout transitions.

Run from the repo root with:

- `uv run pytest -q`
- `uv run ruff check .`
- `uv run mypy`

## Contributor Guardrails

- Keep public semantics stable unless the change explicitly targets API behavior.
- Do not use Git commands as part of automated repo analysis workflows unless explicitly requested by the user.
- Keep documentation aligned with the current implementation stage; do not claim execution support beyond what exists.
