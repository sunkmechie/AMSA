# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- CSR-native NumPy execution paths for CSR/CSR binary products, preserving CSR
  output while still using support-driven Clifford product plans.
- Regression coverage for CSR product output preservation, including
  multidimensional broadcasted batches.

### Changed
- CSR storage-local operations now preserve CSR representation across component
  extraction, layout projection, scaling, unary transformations, coefficient
  magnitude squared, add/sub, and batch indexing where applicable.
- Internal CSR helpers now use trusted construction for helper-produced arrays,
  avoiding repeated validation in hot CSR indexing, add/sub, and product paths.
- Mixed dense/CSR binary products are documented as dense-output operations.

## [0.2.3] - 2026-05-23

### Added
- PGA and VGA support in `alg.classify()` — ideal points, normalized Euclidean
  points, lines, planes, motors, rotors, translators, even versors
- VGA even-versor detection in classification
- Naive forward-mode autodiff: `DualMV`, `directional_derivative()`, `forward_grad()`
- `motor_exp()` and `motor_log()` CGA support for scalar+bivector Euclidean motors

### Removed
- **`amsa.robo` subpackage — split into the standalone `amsa-robo` package.**


## [0.2.2] - 2026-05-02

### Added
- CGA geometry methods on `Algebra`: `point`, `sphere`, `plane`, `translate`,
  `line_through_points`, `circle_through_points`, `origin`, `infinity`,
  `euclidean_vector`, `distance_squared`
- CGA extraction utilities: `extract_point`, `extract_sphere`, `extract_plane`,
  `extract_euclidean_vector` — recover Euclidean parameters from CGA MVs
- Geometric classification: `alg.classify(mv)` returns `EntityInfo` with
  structured geometric interpretation (12 recognized CGA entity types,
  invariants, geometric data, storage metadata)
- `EntityInfo` dataclass with pretty-printed output and programmatic access
- 4 new CGA examples in `examples/cga/`: primitives, batched distance,
  classify overview, versor actions
- `examples/inspection/README.md`; fixed `examples/robotics/README.md`
- `docs/cga.rst` covering CGA constructors, extraction, and classification
- Initial CGA operations roadmap (superseded by `docs/roadmap.rst`)

### Changed
- README : VGA/PGA/CGA quick start snippets, Supported Algebras
  table, Examples directory listing


## [0.2.1] - 2026-04-26

Performance optimization with IR fusion for NumPy backend.

### Added
- IR fusion pattern detection (scale_product, unary_product, elementwise_chain)
- Fusion metadata field to IRStep for marking fusible operations
- Fused kernels in NumPy backend for scale+product and unary+product
- Fused elementwise chain execution
- Fusion analysis module (`amsa.fusion`)
- Fusion benchmarks showing 12-21% speedup for scale+product operations
- Comprehensive fusion tests (15 tests, all passing)

### Changed
- NumPy backend now uses fused execution paths when fusion metadata is present
- CHANGELOG.md is now the single source of truth (docs/changelog.rst links to it)

### Fixed
- Infinite loop bug in fusion execution path (missing increment in non-fused branch)

## [0.2.0] - 2026-04-26

### Added

- **JAX Backend**: Full JAX backend implementation with dense storage parity
- **Device Selection**: `amsa.init(use="cpu")` and `amsa.init(use="gpu")` for backend selection
- **Backend Registry**: Pluggable backend system via `amsa.ir.register_backend()`
- **JAX GPU Support**: CUDA installation instructions for GPU execution
- **Backend Benchmarks**: NumPy vs JAX performance comparison script
- **Storage/Backend Separation**: `StorageDescriptor` and `BackendPayload` protocols for backend-agnostic storage

### Changed

- JAX moved to optional dependencies (`amsa-ga[jax]`)
- NumPy is now a backend, not embedded in storage classes
- All coefficient execution routes through IR layer
- Backend selection affects coefficient execution only, not algebra semantics

### Fixed

- JAX x64 enabled to avoid float64 truncation warnings
- Conditional JAX backend registration for graceful degradation

## [0.1.0b0] - 2026-04-21

### Added

- **Core Algebra**: Clifford algebra engine with blade-based representation and signature-driven semantics
- **Presets**: VGA (2D/3D), PGA (2D/3D) via `Algebra.from_name()` or convenience constructors
- **Products**: geometric, outer, inner, scalar, commutator, anticommutator, left/right contraction, regressive, sandwich
- **Unary Operations**: reverse, involute, conjugate, dual/undual, poincare_dual/undual, inverse
- **Exp/Log**: `exp()`, `motor_exp()`, `motor_log()` for robotics motor slices
- **Norms**: `norm_squared`, `norm`, `normalize`, plus bulk/weight variants and `rigid_body_normalize`
- **Projection**: `grade()`, `project_grades()`, `component()`
- **Storage Backends**: Dense and CSR (sparse) with explicit support tracking
- **Visualization**: Adapter layer for matplotlib/VisPy, geometric primitives (Point, Line, Circle, Rotor)
- **Documentation**: Sphinx docs, README, and tutorial notebooks (VGA rotors, PGA rigid body)
- **Testing**: Probe-based semantic validation suite for algebra identities and backend parity

### Changed

- Backend routing now uses an IR layer for numpy - enables future JAX/other backends without changing operator semantics
