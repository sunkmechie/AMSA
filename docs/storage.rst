Storage backends
================

AMSA separates coefficient storage from layout metadata. The :class:`amsa.storage.MVStorage`
protocol is currently implemented by ``DenseStorage``, ``CSRStorage``, and the experimental
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

An experimental JAX-backed dense storage type for multivector batches.

- construction uses ``jax.numpy.asarray`` / ``jax.numpy.zeros``
- storage-local helpers such as projection, scaling, reweighting, and row scaling work on JAX arrays
- ``as_dense()`` currently converts back to a NumPy ndarray for inspection

This is a storage-layer feature today, not a finished accelerated execution backend.

Backend policy
--------------

- ``backend="auto"`` resolves to ``dense`` for fresh construction.
- ``backend="csr"`` opts into CSR storage explicitly.
- ``backend="jax"`` opts into experimental JAX dense storage explicitly.
- Importing an existing ``MVArray`` preserves its current backend unless a different backend is requested.

Conversion helpers
------------------

- ``to_dense_storage(storage)``
- ``to_csr_storage(storage)``
- ``to_jax_storage(storage)``

Current limitation
------------------

Binary reference execution is now backend-aware for same-backend binary inputs:

- dense inputs produce dense output
- CSR inputs produce CSR output
- JAX inputs produce JAX output

Mixed-backend binary execution still falls back to dense output.

JAX limitation
--------------

The current JAX path is still a reference path, not a compiled kernel backend:

- binary plan execution still runs through the same reference planning/execution structure
- mixed JAX/non-JAX binary execution falls back to dense output
- no fused or JIT-specialized JAX kernels exist yet

So the current JAX path is useful for backend-preserving execution and future kernel work, but it
is not yet a full optimized JAX operator backend.
