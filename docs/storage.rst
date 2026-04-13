Storage backends
================

AMSA separates coefficient storage from layout metadata. The :class:`amsa.storage.MVStorage`
protocol is currently implemented by ``DenseStorage``, ``CSRStorage``, and the Beta
``JAXStorage``.

Dense storage
-------------

A thin wrapper around a NumPy ndarray. The last axis corresponds to the layout size; all preceding axes are batch dimensions.

.. code-block:: python

   from amsa.storage import DenseStorage
   import numpy as np

   storage = DenseStorage.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

CSR storage
-----------

NumPy-backed compressed-row storage for flattened multivector batches.

- ``data`` — nonzero values
- ``indices`` — column indices within the layout
- ``indptr`` — row offsets (flattened batch elements)
- ``batch_shape`` — original logical batch shape
- ``width`` — layout size (number of columns)

.. code-block:: python

   from amsa.storage import CSRStorage

   storage = CSRStorage(
       data=np.array([1.0, 2.0]),
       indices=np.array([0, 2]),
       indptr=np.array([0, 2]),
       batch_shape=(2,),
       width=3,
   )

JAX storage
-----------

A Beta JAX-backed dense storage type for multivector batches.

- construction uses ``jax.numpy.asarray`` / ``jax.numpy.zeros``
- storage-local helpers such as projection, scaling, reweighting, and row scaling work on JAX arrays
- ``as_dense()`` currently converts back to a NumPy ndarray for inspection

This is a storage-layer feature today, not a finished accelerated execution backend.

Backend policy
--------------

- ``backend="auto"`` resolves to ``dense`` for fresh construction.
- ``backend="csr"`` opts into CSR storage explicitly.
- ``backend="jax"`` opts into JAX (Beta) dense storage explicitly.
- Importing an existing ``MVArray`` preserves its current backend unless a different backend is requested.

Conversion helpers
------------------

- ``to_dense_storage(storage)``
- ``to_csr_storage(storage)``
- ``to_jax_storage(storage)``

Acceleration and Fusion
-----------------------

The JAX backend is fully integrated with XLA via ``jax.jit``. 

- **JIT Compilation**: Binary operations (geometric, outer, inner products) are dynamically compiled into optimized XLA kernels.
- **Trace Fusion**: ``MVArray`` and ``JAXStorage`` are registered as PyTree nodes. This allows you to apply ``@jax.jit`` to entire Python functions containing multiple AMSA operations. JAX will trace through the multivector objects and fuse the entire algebraic expression into a single optimized kernel.

Performance Guide
-----------------

When using the JAX backend (especially on GPU), there is a significant trade-off between **latency** and **throughput**.

- **Latency (Single Operations)**: Each JAX kernel launch incurs a small overhead (~100μs on GPU). For single, unbatched multivector operations, the NumPy ``dense`` backend may be faster.
- **Throughput (Batched Operations)**: JAX excels at massive parallelism. For batches of 10,000+ multivectors, JAX can be **10,000x faster** than NumPy.

| Workload | Backend | Time (total) | Time (per element) |
| --- | --- | --- | --- |
| Single GP | NumPy | 40μs | 40μs |
| Single GP | JAX (GPU) | 92μs | 92μs |
| Batch 100k | NumPy | ~4,000,000μs | 40μs |
| **Batch 100k** | **JAX (GPU)** | **163μs** | **0.0016μs** |

For robotics simulations or neural networks, represent your data as batched ``MVArray`` objects to leverage hardware acceleration.
