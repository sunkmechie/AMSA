Storage backends
================

AMSA separates coefficient storage from layout metadata. The :class:`amsa.storage.MVStorage` protocol is implemented by ``DenseStorage`` and ``CSRStorage``.

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

Backend policy
--------------

- ``backend="auto"`` resolves to ``dense`` for fresh construction.
- ``backend="csr"`` opts into CSR storage explicitly.
- Importing an existing ``MVArray`` preserves its current backend unless a different backend is requested.

Conversion helpers
------------------

- ``to_dense_storage(storage)``
- ``to_csr_storage(storage)``

Current limitation
------------------

Binary reference execution can consume dense or CSR inputs, but currently materializes the result as dense storage over the output layout. CSR output emission is planned for a future release.
