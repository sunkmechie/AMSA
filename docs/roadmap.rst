Roadmap
=======

AMSA's northstar is a matrix-free Clifford algebra engine that makes rigorous
geometric computation pleasant to use. Blades remain bit-pattern identifiers,
layouts remain ordering and support metadata, storage remains coefficient
representation, and execution remains separate from all three. Future work
must strengthen those boundaries rather than introducing matrix kernels or
operator abstractions as the primary execution model.

Current capability matrix
-------------------------

.. list-table::
   :header-rows: 1

   * - Capability
     - NumPy
     - JAX
   * - Dense storage
     - Supported
     - Supported
   * - CSR storage
     - Supported
     - Not supported
   * - Dense binary and unary ops
     - Supported
     - Supported
   * - CSR/CSR products
     - Native CSR output
     - Not supported
   * - Mixed dense/CSR products
     - Dense output
     - Not supported
   * - Dense ``jax.jit`` / ``vmap``
     - Not applicable
     - Supported subset
   * - Dense ``jax.grad`` objectives
     - Not applicable
     - Supported subset
   * - Value-checked operations in JIT
     - Not applicable
     - Deferred

JAX execution accepts dense ``MVArray`` storage only. Convert a CSR
multivector using ``with_storage("dense")`` before selecting the JAX backend.
AMSA preserves the configured JAX precision; applications that require x64
must enable it before creating JAX arrays.

Milestone 0: contract lock-down
-------------------------------

- Publish and test the storage/backend capability boundary.
- Keep JAX CSR explicitly unsupported until a device-resident sparse payload
  and valid pytree model exist.
- Define traceability by operation: dense products, involutions, projections,
  coefficient-local arithmetic, ``sandwich()``, and scalar-objective paths are
  supported; operations with value-dependent validation such as
  ``normalize()`` and ``inverse()`` remain deferred in ``jax.jit``.
- Keep documentation, examples, changelog, and runtime behavior synchronized.

Milestone 1: execution reliability
----------------------------------

- Treat blade plans and IR as stable execution contracts.
- Add dense/CSR NumPy and dense JAX differential tests for every supported
  operation, including empty support, exact zeros, and broadcasted batches.
- Establish reproducible benchmarks for plan construction, dense products,
  CSR products, and fused sequences.

Milestone 2: sparse performance depth
--------------------------------------

- Improve NumPy CSR only through storage-local or support-driven algorithms.
- Preserve canonical layout order and the explicit mixed dense/CSR output
  policy.
- Do not introduce matrix representations to accelerate sparse products.

Milestone 3: differentiable Clifford execution
----------------------------------------------

- Retain ``DualMV`` as the small-scale forward-mode reference.
- Lower bilinear tangent propagation beside product-plan and IR construction.
- Add nonlinear derivative rules only after documenting their domains and
  singularity behavior.
- Keep reverse-mode and compiled differentiation in the dense JAX path; defer
  CSR-on-JAX until its representation is sound.

Milestone 4: complete geometric slices
--------------------------------------

- Extend VGA, PGA, and CGA through complete slices: construction, operation,
  extraction or classification where meaningful, batching, and semantic tests.
- Keep robotics domain workflows in ``amsa-robo`` and visualization as neutral
  adapters plus user-selected backends.
