# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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