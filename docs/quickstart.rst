Quickstart
==========

Installation
------------

AMSA requires Python 3.12 or newer. Install with ``uv`` (recommended) or ``pip``:

.. code-block:: bash

   uv sync --extra dev

Verify the installation:

.. code-block:: bash

   uv run pytest -q
   uv run ruff check .
   uv run mypy

Basic usage
-----------

Construct an algebra preset and build multivectors:

.. code-block:: python

   from amsa import Algebra

   alg = Algebra.vga2d()
   u = alg.vector([1.0, 2.0])
   v = alg.vector([3.0, -4.0])

   gp = u * v
   ip = u | v
   op = u ^ v

   print(gp.as_dense().values)  # [-5.0, 0.0, 0.0, -10.0]
   print(ip.values)             # [-5.0]
   print(op.values)             # [-10.0]

Sparse construction keeps support explicit:

.. code-block:: python

   from amsa import Algebra

   alg = Algebra.vga3d()
   mv = alg.multivector({"e1": 1.0, "e12": 2.0, "e123": 3.0})

   print(mv.layout.blades)          # (1, 3, 7)
   print(mv.grade(1, 3).values)     # [1.0, 3.0]
   print((2.0 - mv).as_dense().values)

Scalar construction is intentionally explicit:

.. code-block:: python

   from amsa import Algebra

   alg = Algebra.vga2d()
   s = alg.scalar(1.0)

Use ``alg.scalar(1.0)``, not ``alg.multivector(1.0)``.

Batched coefficients
--------------------

All constructors broadcast over NumPy arrays:

.. code-block:: python

   import numpy as np
   from amsa import Algebra

   alg = Algebra.pga2d()
   mv = alg.multivector({"e0": np.array([1.0, 2.0]), "e1": 3.0})

   print(mv.batch_shape)  # (2,)
   print(mv.values)       # [[1.0, 3.0], [2.0, 3.0]]

Choosing a storage backend
--------------------------

By default, fresh construction uses dense storage. You can opt into CSR explicitly:

.. code-block:: python

   alg = Algebra.vga3d()
   mv = alg.multivector({"e1": np.array([0.0, 2.0]), "e23": 3.0}, backend="csr")

   print(mv.storage_kind)  # csr

There is also an experimental JAX storage mode:

.. code-block:: python

   alg = Algebra.vga2d()
   mv = alg.vector([1.0, 2.0], backend="jax")

   print(mv.storage_kind)  # jax

This is currently a storage-layer option. The reference binary executor still materializes NumPy
results, so ``backend="jax"`` is not yet a full accelerated operator backend.

Visualization
-------------

AMSA includes a lightweight ``amsa.viz`` layer for converting selected multivectors into
plot-friendly primitives.

.. code-block:: python

   import matplotlib.pyplot as plt
   from amsa import Algebra
   from amsa.viz.adapters import to_point
   from amsa.viz.backends import mpl

   alg = Algebra.pga2d()
   point = alg.multivector({"e01": 3.0, "e02": 4.0, "e12": 1.0})

   fig, ax = plt.subplots()
   mpl.plot(ax, to_point(point, color="red", label="robot"))
   ax.set_aspect("equal", "box")
   ax.legend()
   mpl.show()
