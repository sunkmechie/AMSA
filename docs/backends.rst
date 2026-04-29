Execution backends
==================

AMSA supports pluggable execution backends for coefficient computation.
Backends are selected by device type rather than library name, making the API
more intuitive for users who think in terms of CPU/GPU execution.

Device selection
---------------

Use ``amsa.init()`` to select the execution device:

.. code-block:: python

   import amsa
   
   # CPU execution (NumPy) - default
   amsa.init(use="cpu")
   
   # GPU execution (JAX) - requires JAX backend
   # amsa.init(use="gpu")
   
   # Check current device
   print(amsa.get_device())  # "cpu"

Backend mapping
---------------

AMSA maps device types to specific backend implementations:

- ``cpu`` → NumPy backend (always available)
- ``gpu`` → JAX backend (requires ``amsa-ga[jax]`` extra)

The NumPy backend is registered and set as the default on import, so most
users do not need to call ``amsa.init()`` unless they want to explicitly
switch devices.

JAX backend
-----------

To use the JAX backend for CPU execution, install the JAX extra:

.. code-block:: bash

   uv pip install amsa-ga[jax]

For GPU execution (CUDA), install JAX with CUDA support:

.. code-block:: bash

   uv pip install "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Then select GPU execution:

.. code-block:: python

   import amsa
   amsa.init(use="gpu")

**JAX-specific notes:**

- JAX uses functional array updates (``.at[].set()``) instead of in-place mutations
- The JAX backend currently supports dense storage only; CSR support is planned for a future release
- JAX may truncate float64 to float32 by default. Enable float64 with the ``JAX_ENABLE_X64=1`` environment variable or ``jax.config.update("jax_enable_x64", True)`` in your code
- JIT compilation can be enabled on individual backend functions for performance, but is not enabled by default to maintain debugging tractability

JAX traceability contract
-------------------------

AMSA's JAX integration treats algebraic structure as static metadata and
coefficient arrays as dynamic data. This keeps tracing aligned with AMSA's
matrix-free architecture: JAX transforms coefficient execution, while blades,
layouts, supports, and product plans remain Clifford metadata.

Static metadata:

- ``AlgebraSpec`` values, including signature and basis naming policy
- ``MVLayout`` values, including blade ordering and support
- product plans and IR objects such as ``ProductIR`` and ``UnaryIR``
- storage descriptors such as storage kind, width, and batch rank

Dynamic values:

- coefficient arrays
- scalar coefficient inputs used by scale, row-scale, and coefficient helpers
- batch contents within a fixed traced shape

Each compiled JAX trace specializes to the static algebra/layout/IR metadata and
to the array shapes seen by JAX. Changing coefficients should not require a new
trace; changing layouts, blade support, or output shape may require one.

Traceability targets
~~~~~~~~~~~~~~~~~~~~

The dense JAX path is expected to become traceable for these core operations:

- dense binary products: geometric, outer, inner, scalar, left contraction,
  right contraction, and regressive product
- unary involutions and duals: reverse, involute, conjugate, Poincare dual, and
  Poincare undual
- coefficient-local operations: add, sub, scale, row_scale, and grade projection
- composed Clifford operations that do not require value-dependent validation,
  such as norm_squared
- scalar-objective autodiff paths built from differentiable Clifford operations
- coefficient helper kernels for ``exp()``, ``motor_exp()``, and motor-log
  coefficient calculations

Deferred traceability targets:

- CSR storage on JAX
- value-dependent output support or value-dependent output shapes
- Python exceptions triggered from traced coefficient values
- singular normalization branches inside ``jax.jit``
- predicate helpers that intentionally return Python ``bool`` values
- validation-backed public operations such as ``normalize()``, ``inverse()``,
  and ``sandwich()`` until their value checks have a trace-safe validation model

Implementation rules for traceable paths:

- register AMSA containers as JAX pytrees only when array payloads are leaves and
  algebra/layout metadata is static auxiliary data
- avoid Python ``bool(...)`` conversions of traced values
- avoid value-dependent boolean indexing that changes array size
- prefer shape-preserving ``jax.numpy.where`` expressions in coefficient kernels
- keep validation that raises Python exceptions outside jitted numeric kernels

Benchmarking note:

Traceability should be verified before performance claims are made. Add or
refresh JAX benchmarks after the dense core operation suite has explicit
``jax.jit`` coverage, so measurements reflect stable supported behavior rather
than isolated helper kernels.

Important notes
---------------

- Backend selection affects **coefficient execution only**
- Algebra semantics, blade identity, and layout ordering are unchanged
- Storage backend (dense/CSR) is independent of execution backend
- Device selection is global for the current Python process

Future backends
--------------

Additional backends can be registered via the low-level ``amsa.ir`` module:

.. code-block:: python

   from amsa.ir import register_backend
   
   # Register a custom backend
   register_backend("my_backend", MyExecutor())
   amsa.init(use="cpu")  # Switches to registered backend

See the ``amsa.ir`` module documentation for the ``Executor`` protocol and
backend registration details.
