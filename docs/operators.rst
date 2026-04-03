Operators and semantics
=======================

AMSA provides a public operator layer in :mod:`amsa.ops` and matching methods on :class:`amsa.mv.MVArray`. All operators respect the algebra metric, broadcast over batch dimensions, and preserve sparse support whenever possible.

Binary products
---------------

Geometric product ``*``
^^^^^^^^^^^^^^^^^^^^^^^

Includes every nonzero blade-pair contribution.

Outer product ``^``
^^^^^^^^^^^^^^^^^^^

Includes only terms whose output grade equals the sum of the input grades.

Inner product ``|``
^^^^^^^^^^^^^^^^^^^

Includes only terms whose output grade equals the absolute difference of the input grades.

Scalar product
^^^^^^^^^^^^^^

Includes only terms whose output grade equals ``0``.

Left contraction
^^^^^^^^^^^^^^^^

Includes only terms with ``grade(lhs) <= grade(rhs)`` and output grade ``grade(rhs) - grade(lhs)``.

Right contraction
^^^^^^^^^^^^^^^^^

Includes only terms with ``grade(lhs) >= grade(rhs)`` and output grade ``grade(lhs) - grade(rhs)``.

Regressive product
^^^^^^^^^^^^^^^^^^

Defined via the Poincaré-dual complement of the outer product:

.. code-block:: text

   regressive_product(a, b) = poincare_undual(poincare_dual(a) ^ poincare_dual(b))

Involutions and duality
-----------------------

- ``reverse()`` — grade-dependent sign flip
- ``involute()`` — parity flip by grade
- ``conjugate()`` — ``reverse(involute(mv))``
- ``dual()`` / ``undual()`` — metric pseudoscalar duals (require invertible pseudoscalar)
- ``poincare_dual()`` / ``poincare_undual()`` — metric-free basis complement duals (work on degenerate algebras)

Normalization
-------------

- ``norm_squared()`` — signed reverse norm scalar ``<mv * reverse(mv)>_0``
- ``norm()`` — ``sqrt(abs(norm_squared()))``, kept real on indefinite signatures
- ``normalize()`` — divides by ``norm()``

Inverse
-------

``inverse()`` is currently a restricted reverse-based inverse. It succeeds when ``reverse(mv) * mv`` and ``mv * reverse(mv)`` both reduce to the same nonzero scalar. This covers scalars, invertible blades, and common rotor-like cases.

PGA helpers
-----------

For degenerate / projective algebras:

- ``bulk()`` — coefficients whose blades do **not** contain the null basis factor
- ``weight()`` — coefficients whose blades **do** contain the null basis factor
- ``bulk_dual()`` / ``weight_dual()`` — Poincaré duals applied to the split parts
- ``bulk_norm()``, ``weight_norm()`` — separate normalization magnitudes
- ``bulk_normalize()`` / ``unitize()`` — PGA-facing normalization paths

Sandwich
--------

``sandwich(actor, target)`` computes ``actor * target * inverse(actor)``. It inherits the current restricted ``inverse()`` support and is most useful with normalized versor-like operands.
