Storage backends
================

AMSA separates coefficient storage from layout metadata. The
:class:`amsa.storage.MVStorage` protocol is implemented by ``DenseStorage``
and ``CSRStorage``.

Dense storage
-------------

A thin wrapper around a NumPy ndarray. The last axis corresponds to the layout
size; all preceding axes are batch dimensions.

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
   import numpy as np

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

Storage-aware execution
-----------------------

Storage remains coefficient representation only. Layouts still define blade
ordering and support, plans still define Clifford product terms, and execution
backends decide how to consume the coefficient payloads.

The NumPy backend currently preserves CSR storage for storage-local operations
where doing so is natural:

- component extraction
- layout projection
- scalar and row scaling
- unary sign/permutation operations such as reverse, involute, conjugate, and
  dual variants
- coefficient magnitude squared
- CSR/CSR addition and subtraction, including broadcasted scalar-batch cases
- CSR batch indexing

CSR/CSR binary products also emit CSR output in the NumPy backend. The product
still comes from the same support-driven Clifford product plans as dense
execution; the CSR path only changes how coefficient rows are traversed and
stored.

Mixed dense/CSR products follow an explicit dense-output policy. They can
consume CSR inputs without first materializing the CSR operand as a full dense
multivector, but the result uses dense storage. This keeps the storage rule
simple: when a binary product combines dense and CSR payloads, the dense side
determines the output representation.

Current limitations
-------------------

- ``backend="auto"`` does not infer CSR from sparsity; users must request
  ``backend="csr"`` or call ``with_storage("csr")``.
- CSR support is NumPy-backed. Dense JAX execution is supported separately, but
  CSR-on-JAX is intentionally deferred.
- Mixed dense/CSR binary products intentionally return dense output.
