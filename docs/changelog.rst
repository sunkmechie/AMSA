Changelog
=========

0.1.0 (2026-04-04)
------------------

Initial release.

- Portable algebra core based on bit-pattern blade identifiers
- Lazy numeric basis-product tables for small-basis algebras (≤512 blades)
- Dense, grade-packed, and sparse layout descriptors
- Dense and CSR storage backends behind a shared storage contract
- Reference multivector array type with batch broadcasting
- Cached operator plans for binary products
- Storage-aware binary execution consuming dense or CSR inputs
- Explicit constructor-level backend selection (``backend="auto" | "dense" | "csr"``)
- Public API covering geometric, outer, inner, scalar, left/right contraction, and regressive products
- Reverse, involute, conjugate, metric dual/undual, and Poincaré dual/undual
- Reverse-based norm, normalization, and restricted inverse
- PGA-specific bulk/weight split, duals, and normalization helpers
- Visual debugger probe (``amsa_lab.py``) that traces plans to interactive HTML
- Comprehensive test suite with dense and CSR coverage
