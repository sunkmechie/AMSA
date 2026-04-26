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

To use the JAX backend for GPU execution, install the JAX extra:

.. code-block:: bash

   uv pip install amsa-ga[jax]

Then select GPU execution:

.. code-block:: python

   import amsa
   amsa.init(use="gpu")

**JAX-specific notes:**

- JAX uses functional array updates (``.at[].set()``) instead of in-place mutations
- The JAX backend currently supports dense storage only; CSR support is planned for a future release
- JAX may truncate float64 to float32 by default. Enable float64 with the ``JAX_ENABLE_X64=1`` environment variable or ``jax.config.update("jax_enable_x64", True)`` in your code
- JIT compilation can be enabled on individual backend functions for performance, but is not enabled by default to maintain debugging tractability

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
