Visualization
=============

AMSA now includes a lightweight in-package visualization layer under ``amsa.viz``.

This layer is intentionally separate from the core algebra engine:

- the algebra core still works in blades, grades, layouts, and supports
- ``amsa.viz`` converts selected multivectors into neutral geometric primitives
- optional backends render those primitives without changing algebra semantics

Current structure
-----------------

- ``amsa.viz.primitives`` — neutral visualization dataclasses such as ``Point`` and ``Rotor``
- ``amsa.viz.adapters`` — multivector-to-primitive adapters, currently focused on PGA points
- ``amsa.viz.backends.mpl`` — optional matplotlib backend for plotting primitives

Current API
-----------

- ``to_point(mv, *, color=None, label=None)``
- ``Point``
- ``Line``
- ``Plane``
- ``Rotor``
- ``VizPrimitive``

The current matplotlib backend provides:

- ``plot(ax, primitive, **kwargs)``
- ``show()``

Point extraction
----------------

``to_point(...)`` currently supports:

- ``pga2d`` points encoded as ``x e01 + y e02 + w e12``
- ``pga3d`` points encoded in the canonical trivector basis used by AMSA

Weighted points are normalized by their homogeneous weight when possible. Ideal points
(``w = 0``) are returned as direction-like coordinates rather than raising.

Design boundary
---------------

The visualization layer is intentionally allowed to use derived numeric representations
that AMSA core would avoid. For example, the ``Rotor`` primitive may store a matrix for
backend convenience, but that representation stays isolated inside ``amsa.viz`` and does
not affect the matrix-free algebra core.

Relationship to probes
----------------------

``amsa.viz`` is not the same as the external probe tooling.

- ``amsa.viz`` is for lightweight plotting and geometric adapters
- ``probes/amsa_lab.py`` is still the richer operator-plan visual debugger

That split keeps normal visualization support available in-package while allowing the
debugging UX to evolve more freely.
