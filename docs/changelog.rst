Changelog
=========

For full release notes, see the project's `CHANGELOG.md <https://github.com/amsa/amsa/blob/main/CHANGELOG.md>`_ on GitHub.

0.1.0b0 (2026-04-21)
--------------------

Beta release.

- Portable algebra core based on bit-pattern blade identifiers
- VGA (2D/3D) and PGA (2D/3D) presets via ``Algebra.from_name()``
- Dense and CSR storage backends behind a shared storage contract
- Reference multivector array type with batch broadcasting
- Cached operator plans for binary products
- Products: geometric, outer, inner, scalar, commutator, anticommutator, left/right contraction, regressive, sandwich
- Unary: reverse, involute, conjugate, dual/undual, poincare_dual/undual, inverse
- Exp/log: ``exp()``, ``motor_exp()``, ``motor_log()`` for robotics motor slices
- Norms: ``norm_squared``, ``norm``, ``normalize``, plus bulk/weight variants and ``rigid_body_normalize``
- Projection: ``grade()``, ``project_grades()``, ``component()``
- In-package visualization layer (``amsa.viz``) with geometric primitives and optional matplotlib/VisPy backends
- Visual debugger probe (``amsa_lab.py``) that traces plans to interactive HTML
- Comprehensive test suite with dense and CSR coverage
