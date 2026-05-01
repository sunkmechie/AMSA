CGA
===

AMSA includes first-pass conformal geometric algebra presets and constructors.
The implementation keeps the core algebra diagonal and matrix-free:
``cga3d`` uses Euclidean axes followed by two conformal axes with squares
``+1`` and ``-1``. The public helpers build the standard null vectors
``n_o = 0.5(e_- - e_+)`` and ``n_inf = e_- + e_+``.

.. code-block:: python

   import amsa
   from amsa import cga

   alg = amsa.Algebra.cga3d()
   a = cga.point(alg, [1.0, 2.0, 3.0])
   b = cga.point(alg, [2.0, 2.0, 3.0])

   print(cga.distance_squared(a, b))  # 1.0

Available helpers
-----------------

- ``cga.origin(alg)`` and ``cga.infinity(alg)``
- ``cga.euclidean_vector(alg, coordinates)``
- ``cga.point(alg, coordinates)``
- ``cga.sphere(alg, center, radius)`` as a dual sphere
- ``cga.plane(alg, normal, distance)`` as a dual plane
- ``cga.line_through_points(a, b)``
- ``cga.circle_through_points(a, b, c)``
- ``cga.translate(alg, displacement)``
- ``cga.distance_squared(a, b)``

The point embedding is ``X = n_o + x + 0.5 (x*x) n_inf``. For normalized
conformal points, ``-2 (A · B)`` gives squared Euclidean distance.

Operator status
---------------

The constructors use existing AMSA products and layouts. NumPy execution is
covered by tests; dense JAX parity follows the same operation layer where the
underlying operation is already traceable. There is no matrix representation or
basis-change table in the CGA implementation.
