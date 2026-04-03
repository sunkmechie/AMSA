Algebra specs
=============

AMSA represents a Clifford algebra with an immutable :class:`amsa.specs.AlgebraSpec`. It stores the signature, dimension, blade naming rules, and lazy basis-product tables.

Core properties
---------------

- ``signature`` — tuple of metric values (``-1``, ``0``, ``1``)
- ``dimension`` — number of basis vectors
- ``blade_count`` — total basis blades, ``2**dimension``
- ``p``, ``q``, ``r`` — counts of positive, negative, and null metric directions

Blade naming and lookup
-----------------------

.. code-block:: python

   from amsa import Algebra

   alg = Algebra.vga3d()
   spec = alg.spec

   print(spec.blade_name(0))     # e
   print(spec.blade_name(7))     # e123
   print(spec.blade_from_key("e12"))  # 3

Presets
-------

The following presets are built in:

- ``Algebra.vga2d()`` — Euclidean 2D
- ``Algebra.vga3d()`` — Euclidean 3D
- ``Algebra.pga2d()`` — 2D Projective Geometric Algebra
- ``Algebra.pga3d()`` — 3D Projective Geometric Algebra

You can also construct arbitrary signatures:

.. code-block:: python

   from amsa import AlgebraSpec

   spec = AlgebraSpec.from_pqr(p=3, q=1, r=1)

Products and tables
-------------------

- ``blade_product(lhs, rhs)`` returns ``(coefficient, output_blade)`` for a single pair.
- ``basis_product_table`` returns a precomputed numeric table for small algebras (≤512 blades).
- ``cayley_table()`` returns a human-readable dict mapping ``(lhs_name, rhs_name) → result``.

For large algebras (dimension > 9) the precomputed table is skipped and products are computed on demand with caching.
