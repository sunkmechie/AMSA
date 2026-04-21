# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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