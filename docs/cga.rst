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

Extraction utilities
^^^^^^^^^^^^^^^^^^^^

Extract Euclidean parameters from CGA multivectors.  ``extract_point``
normalizes via ``-(X · n_inf)`` so it works correctly after versor actions.

.. code-block:: python

   import amsa

   alg = amsa.Algebra.cga3d()

   p = alg.point([1.0, 2.0, 3.0])
   print(alg.extract_point(p))  # [1. 2. 3.]

   s = alg.sphere([1.0, 0.0, 0.0], 3.0)
   center, radius = alg.extract_sphere(s)
   print(center, radius)  # [1. 0. 0.] 3.0

   plane = alg.plane([0.0, 0.0, 1.0], 2.0)
   normal, distance = alg.extract_plane(plane)
   print(normal, distance)  # [0. 0. 1.] 2.0

    # After versor action — still extracts correctly
    reflected = amsa.sandwich(plane, p)
    print(alg.extract_point(reflected))  # [1. 2. 1.]  (reflected across z=2)

Available extraction methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``alg.extract_point(mv)`` — Euclidean point coordinates (normalizes by ``-(X·n_inf)``)
- ``alg.extract_sphere(mv)`` — ``(center, radius)`` from a dual sphere
- ``alg.extract_plane(mv)`` — ``(normal, signed_distance)`` from a dual plane
- ``alg.extract_euclidean_vector(mv)`` — Euclidean coordinates from a subspace vector

Classification
^^^^^^^^^^^^^^

``alg.classify(mv)`` inspects a multivector and returns an
:class:`~amsa.algebra.EntityInfo` with its geometric interpretation.

.. code-block:: python

   import amsa

   alg = amsa.Algebra.cga3d()

   print(alg.classify(alg.point([1.0, 2.0, 3.0])))

.. code-block:: text

   CGA3D Classification
   --------------------
   kind:           normalized conformal point
   representation: direct

   grades:        {1}
   null:          yes
   normalized:    yes

   geometric data:
     coordinates: [1. 2. 3.]

   invariants:
     X²   = 0
     X·n∞ = -1

   storage:
     layout       dense
     backend      numpy
     batch_shape  ()
     dtype        float64

Recognized CGA entities
~~~~~~~~~~~~~~~~~~~~~~~

- ``normalized conformal point`` — grade 1, null, ``X·n∞ = -1``
- ``conformal point`` — grade 1, null, unnormalized
- ``point at infinity`` — grade 1, null, ``X·n∞ = 0``
- ``dual sphere`` — grade 1, not null, contains n_o
- ``dual plane`` — grade 1, not null, no n_o component
- ``direct line`` — grade 3, null
- ``direct circle`` — grade 3, null
- ``translator candidate`` — grades {0, 2}, contains conformal axes
- ``even versor`` — grades {0, 2}, no conformal axes
- ``generic blade`` — single grade
- ``zero multivector`` — empty or all-zero coefficients
- ``unknown multivector`` — everything else

.. warning::

   ``classify()`` provides a *structured geometric interpretation*, not a
   mathematical proof.  It uses numerical tolerance (1e-10) and may classify
   near-null or near-degenerate objects approximately.  It never mutates the
   input multivector.

.. note::

   Extraction is documented in Dorst, Fontijne, Mann (2007), *Geometric Algebra
   for Computer Science*, Morgan Kaufmann, Tables 13.1–13.4.
   ``extract_point`` normalizes using the inverse mapping from Perwass (2009),
   §4.3.2.

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
- ``extract_point(mv)`` — Euclidean point coordinates
- ``extract_sphere(mv)`` — ``(center, radius)``
- ``extract_plane(mv)`` — ``(normal, signed_distance)``
- ``extract_euclidean_vector(mv)`` — Euclidean coordinates

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
