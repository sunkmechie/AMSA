CGA
===

AMSA includes first-pass conformal geometric algebra presets and constructors.
The implementation keeps the core algebra diagonal and matrix-free:
``cga3d`` uses Euclidean axes followed by two conformal axes with squares
``+1`` and ``-1``. The public helpers build the standard null vectors
``n_o = 0.5(e_- - e_+)`` and ``n_inf = e_- + e_+``.

Algebra methods (primary API)
-----------------------------

CGA geometry constructors are available as methods on ``Algebra``. This is
consistent with how VGA and PGA constructors work
(e.g. ``alg.vector(values)``, ``alg.bivector(values)``):

.. code-block:: python

   import amsa

   alg = amsa.Algebra.cga3d()
   a = alg.point([1.0, 2.0, 3.0])
   b = alg.point([2.0, 2.0, 3.0])

   print(alg.distance_squared(a, b))  # 1.0

Available ``Algebra`` methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``alg.origin(*, backend)`` — conformal null origin ``n_o``
- ``alg.infinity(*, backend)`` — conformal null infinity ``n_inf``
- ``alg.euclidean_vector(coordinates, *, backend)`` — embed Euclidean coordinates
- ``alg.point(coordinates, *, backend)`` — conformal point
  ``X = n_o + x + 0.5 (x·x) n_inf``
- ``alg.sphere(center, radius, *, backend)`` — dual sphere ``S = C - 0.5 r^2 n_inf``
- ``alg.plane(normal, distance, *, backend)`` — dual plane ``P = n + d n_inf``
- ``alg.translate(displacement, *, backend)`` — CGA translator
  ``T = 1 - 0.5 t n_inf``
- ``alg.line_through_points(a, b)`` — direct line through two conformal points
- ``alg.circle_through_points(a, b, c)`` — direct circle through three conformal points
- ``alg.distance_squared(a, b)`` — Euclidean squared distance from normalized points

Standalone ``amsa.cga`` module (secondary API)
----------------------------------------------

The same functions are available as standalone helpers in ``amsa.cga`` for users
who prefer explicit imports. All functions take the algebra as the first argument:

.. code-block:: python

   import amsa
   from amsa.cga import origin, point, distance_squared

   alg = amsa.Algebra.cga3d()
   a = point(alg, [1.0, 2.0, 3.0])
   b = point(alg, [2.0, 2.0, 3.0])
   print(distance_squared(alg, a, b))  # 1.0

Available standalone helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``origin(alg, *, backend)``
- ``infinity(alg, *, backend)``
- ``euclidean_vector(alg, coordinates, *, backend)``
- ``point(alg, coordinates, *, backend)``
- ``sphere(alg, center, radius, *, backend)``
- ``plane(alg, normal, distance, *, backend)``
- ``translate(alg, displacement, *, backend)``
- ``line_through_points(alg, a, b)``
- ``circle_through_points(alg, a, b, c)``
- ``distance_squared(alg, a, b)``

CGA identities
--------------

The point embedding is ``X = n_o + x + 0.5 (x*x) n_inf``. For normalized
conformal points, ``-2 (A · B)`` gives squared Euclidean distance.

Null basis identities: ``n_o^2 = 0``, ``n_inf^2 = 0``, ``n_o · n_inf = -1``.

Operator status
---------------

The constructors use existing AMSA products and layouts. NumPy execution is
covered by tests; dense JAX parity follows the same operation layer where the
underlying operation is already traceable. There is no matrix representation or
basis-change table in the CGA implementation.
